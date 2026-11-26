import csv
import shutil
from pathlib import Path

import yaml

from detectit.data.dataset import check_records, custom_ground_truth, load_manifest
from detectit.eval.validation import score
from detectit.models.yolo import Detector
from detectit.utils.runs import git_state, is_reportable, sha256, write_json


def training_data(manifest, run):
    staging = (run / "training-data").resolve()
    paths = {"path": str(staging), "names": dict(enumerate(manifest["names"]))}
    for split in ("train", "val"):
        records = check_records(manifest, split)
        paths[split] = f"{split}/images"
        for record in records:
            relative_image = Path(record["image"])
            relative_image = Path(*relative_image.parts[relative_image.parts.index("images") + 1:])
            image = staging / split / "images" / relative_image
            label = staging / split / "labels" / relative_image.with_suffix(".txt")
            for key, destination in (("image", image), ("label", label)):
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Ultralytics may repair JPEGs in place while it builds its cache.
                shutil.copyfile(manifest["root"] / record[key], destination)
    config_path = (run / "dataset.yaml").resolve()
    config_path.write_text(yaml.safe_dump(paths, sort_keys=False), encoding="utf-8")
    return config_path


def finetune(config, run, device, workers, git_start):
    manifest_path = config["dataset"]["manifest"]
    manifest = load_manifest(manifest_path)
    split_hash = sha256(manifest_path)
    training_yaml = training_data(manifest, run)
    detector = Detector(config)
    initial_hash = sha256(detector.weights)
    detector.model.train(
        data=str(training_yaml),
        imgsz=config["imgsz"],
        seed=config["seed"],
        device=device,
        workers=workers,
        project=str(run.resolve()),
        name="training",
        val=True,
        plots=True,
        **config["training"],
    )
    training = detector.model.trainer
    source = Path(training.best)
    if not source.is_file():
        raise FileNotFoundError("Training did not produce a best-validation checkpoint.")
    shutil.copyfile(training.csv, run / "learning-curve.csv")
    with (run / "learning-curve.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{name.strip(): value for name, value in row.items()} for row in csv.DictReader(stream)]
    best = max(rows, key=lambda row: (float(row["metrics/mAP50-95(B)"]), int(row["epoch"])))

    checkpoint = run / "best.pt"
    shutil.copyfile(source, checkpoint)
    candidate = {
        "model": f"{config['model']} · {manifest['name']}",
        "weights": checkpoint.as_posix(),
        "imgsz": config["imgsz"],
        "confidence": config["confidence"],
        "nms_iou": config["nms_iou"],
        "matching_iou": config["matching_iou"],
        "seed": config["seed"],
        "dataset": {
            **{key: manifest[key] for key in ("name", "version", "license", "source")},
            "manifest": manifest_path,
        },
        "fixed_test": {"weights_sha256": sha256(checkpoint), "manifest_sha256": split_hash},
    }
    (run / "checkpoint.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")

    tuned = Detector(candidate)
    # Use the same AP protocol on both splits to expose the train/validation gap.
    split_scores = {}
    for split in ("train", "val"):
        print(f"Scoring the selected checkpoint on {split}", flush=True)
        ground_truth, paths = custom_ground_truth(manifest, split)
        split_scores[split] = score(tuned, ground_truth, paths, sorted(paths), candidate, device, batch=8)

    train_ap = split_scores["train"]["map_50_95"]
    val_ap = split_scores["val"]["map_50_95"]
    revision, clean = git_state()
    write_json(run / "metrics.json", {
        "model": candidate["model"],
        "weights": checkpoint.name,
        "weights_sha256": candidate["fixed_test"]["weights_sha256"],
        "initial_weights": detector.weights.name,
        "initial_weights_sha256": initial_hash,
        "dataset": {key: manifest[key] for key in ("name", "version", "license", "source")},
        "split": "train_and_val",
        "split_sha256": split_hash,
        "git_revision": git_start[0],
        "best_epoch": int(best["epoch"]),
        "best_ultralytics_val_map_50_95": float(best["metrics/mAP50-95(B)"]),
        "train_minus_val_map_50_95": train_ap - val_ap,
        "scores": split_scores,
        "reportable": is_reportable(
            full_split=True, clean_start=git_start[1], clean_end=clean,
            same_revision=revision == git_start[0],
        ),
    })
    print(f"Saved training evidence to {run}")
