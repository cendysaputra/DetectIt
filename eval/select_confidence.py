import argparse
from itertools import batched
from pathlib import Path

from detectit.config import load_config
from detectit.eval.metrics import operating_annotations, operating_counts
from detectit.eval.validation import validation_data
from detectit.models.yolo import Detector
from detectit.utils.runs import git_state, sha256, write_json

THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
BATCH = 8


def main():
    parser = argparse.ArgumentParser(description="Select confidence by micro F1 on validation only.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device")
    args = parser.parse_args()
    config = load_config(args.config)
    if "manifest" not in config["dataset"]:
        parser.error("Use a custom dataset with a frozen validation manifest.")
    ground_truth, paths, split_hash, _ = validation_data(config, "val")
    detector = Detector(config)
    category_ids = {category["name"]: key for key, category in ground_truth.cats.items()}
    categories = {class_id: category_ids[name] for class_id, name in detector.names.items()}
    totals = {threshold: [0, 0, 0] for threshold in THRESHOLDS}
    for ids in batched(sorted(paths), BATCH):
        predictions = detector.predict_many(
            [str(paths[image_id]) for image_id in ids],
            confidence=min(THRESHOLDS), device=args.device,
        )
        for image_id, detections in zip(ids, predictions, strict=True):
            annotations = operating_annotations(ground_truth, image_id, categories)
            for threshold, total in totals.items():
                tp, fp, fn = operating_counts(detections, annotations, config["matching_iou"], threshold)
                total[0] += tp
                total[1] += fp
                total[2] += fn

    results = []
    for threshold, (tp, fp, fn) in totals.items():
        results.append({
            "confidence": threshold,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
        })
    best = max(results, key=lambda row: (row["micro_f1"], row["confidence"]))
    if not best["micro_f1"]:
        parser.error("No candidate threshold yielded a true positive on validation.")
    revision, clean = git_state()
    write_json(args.output, {
        "config_source": args.config.as_posix(),
        "git_revision": revision,
        "clean_tree": clean,
        "weights_sha256": sha256(detector.weights),
        "split": "val",
        "split_sha256": split_hash,
        "images": len(paths),
        "nms_iou": config["nms_iou"],
        "matching_iou": config["matching_iou"],
        "selected": best,
        "thresholds": results,
    })
    print(f"Selected confidence {best['confidence']:.2f}, validation F1 {best['micro_f1']:.4f}")


if __name__ == "__main__":
    main()
