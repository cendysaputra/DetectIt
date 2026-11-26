import time
import platform
from itertools import batched
from pathlib import Path

import cv2
import torch
from pycocotools.coco import COCO

from detectit.data.dataset import custom_ground_truth, load_manifest
from detectit.eval.metrics import coco_metrics, coco_predictions, operating_annotations, operating_counts
from detectit.models.yolo import Detector
from detectit.utils.runs import git_state, is_reportable, sha256

VALIDATION_CONFIDENCE = 0.001
VALIDATION_NMS_IOU = 0.7
PUBLISHED = {
    "yolo11n.pt": (0.395, "https://docs.ultralytics.com/models/yolo11/"),
    "yolov8n.pt": (0.373, "https://docs.ultralytics.com/models/yolov8/"),
}


def validation_data(config, split):
    dataset = config["dataset"]
    if "manifest" in dataset:
        manifest = load_manifest(dataset["manifest"])
        ground_truth, paths = custom_ground_truth(manifest, split)
        return ground_truth, paths, sha256(dataset["manifest"]), manifest
    if split != "val":
        raise ValueError("Only val2017 has public annotations for this COCO config.")
    annotations = Path(dataset["annotations"])
    ground_truth = COCO(str(annotations))
    paths = {
        image_id: Path(dataset["images"]) / info["file_name"]
        for image_id, info in ground_truth.imgs.items()
    }
    if len(paths) != 5000 or set(path.name for path in paths.values()) != {
        path.name for path in Path(dataset["images"]).glob("*.jpg")
    }:
        raise ValueError("Use the official 5,000-image COCO val2017 folder and annotations.")
    return ground_truth, paths, sha256(annotations), dataset


def score(detector, ground_truth, paths, image_ids, config, device, batch):
    category_ids = {item["name"]: key for key, item in ground_truth.cats.items()}
    categories = {class_id: category_ids[name] for class_id, name in detector.names.items()}
    if set(categories.values()) != set(category_ids.values()):
        raise ValueError("Checkpoint classes must match the evaluation dataset classes.")

    predictions = []
    totals = [0, 0, 0]
    same_nms = config["nms_iou"] == VALIDATION_NMS_IOU
    reuse_predictions = same_nms and config["confidence"] >= VALIDATION_CONFIDENCE
    for batch_index, ids in enumerate(batched(image_ids, batch)):
        sources = [str(paths[image_id]) for image_id in ids]
        validation = list(detector.predict_many(
            sources, confidence=VALIDATION_CONFIDENCE, nms_iou=VALIDATION_NMS_IOU, device=device,
        ))
        operating = validation if reuse_predictions else list(detector.predict_many(
            sources, confidence=config["confidence"], nms_iou=config["nms_iou"], device=device,
        ))
        for image_id, detections, deployed in zip(ids, validation, operating, strict=True):
            predictions.extend(coco_predictions(image_id, detections, categories))
            annotations = operating_annotations(ground_truth, image_id, categories)
            counts = operating_counts(deployed, annotations, config["matching_iou"], config["confidence"])
            totals = [previous + count for previous, count in zip(totals, counts, strict=True)]
        if batch_index % 25 == 0 or batch_index * batch + batch >= len(image_ids):
            print(f"Evaluated {min((batch_index + 1) * batch, len(image_ids))}/{len(image_ids)} images", flush=True)

    tp, fp, fn = totals
    metrics = coco_metrics(ground_truth, predictions, image_ids)
    metrics.update({
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "operating_counts": {"tp": tp, "fp": fp, "fn": fn},
    })
    return metrics


def latency(config, paths, image_ids, benchmark_images, warmup):
    images = [cv2.imread(str(paths[image_id])) for image_id in image_ids[:benchmark_images]]
    measurements = {}
    devices = {"cpu": platform.processor() or platform.machine()}
    if torch.cuda.is_available():
        devices["cuda:0"] = torch.cuda.get_device_name(0)
    for device, name in devices.items():
        detector = Detector(config)
        for index in range(warmup):
            detector.predict(images[index % len(images)], device=device)
        if device != "cpu":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for image in images:
            detector.predict(image, device=device)
        if device != "cpu":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        measurements[device] = {
            "name": name, "torch_threads": torch.get_num_threads(), "batch_size": 1, "images": len(images), "warmup": warmup,
            "ms_per_image": 1000 * seconds / len(images),
            "frames_per_second": len(images) / seconds,
        }
    return measurements


def evaluate(config, args, git_start):
    ground_truth, paths, split_hash, dataset = validation_data(config, args.split)
    image_ids = sorted(paths)
    if not image_ids:
        raise ValueError("The requested split is empty.")
    if args.limit is not None:
        image_ids = image_ids[:args.limit]
    if any(not paths[image_id].is_file() for image_id in image_ids):
        raise ValueError("Images declared in the split are missing.")
    detector = Detector(config)
    weights_hash = sha256(detector.weights)
    fixed_test = args.split != "test"
    if args.split == "test":
        lock = config.get("fixed_test", {})
        fixed_test = lock.get("weights_sha256") == weights_hash and lock.get("manifest_sha256") == split_hash
        if not fixed_test:
            raise ValueError("Commit a final checkpoint config with the weight and manifest hashes before test evaluation.")
    metrics = score(detector, ground_truth, paths, image_ids, config, args.device, args.batch)
    metrics["latency"] = latency(config, paths, image_ids, args.benchmark_images, args.warmup)
    revision, clean = git_state()
    metrics.update({
        "model": config["model"],
        "weights": detector.weights.name,
        "weights_sha256": weights_hash,
        "dataset": {key: dataset[key] for key in ("name", "version", "license", "source") if key in dataset},
        "split": "val2017" if dataset["name"] == "COCO" else args.split,
        "split_sha256": split_hash,
        "evaluated_images": len(image_ids),
        "git_revision": git_start[0],
        "confidence": config["confidence"],
        "nms_iou": config["nms_iou"],
        "matching_iou": config["matching_iou"],
        "ap_confidence": VALIDATION_CONFIDENCE,
        "ap_nms_iou": VALIDATION_NMS_IOU,
        "ap_matching_iou": [round(0.5 + i * 0.05, 2) for i in range(10)],
        "reportable": is_reportable(
            full_split=args.limit is None, clean_start=git_start[1], clean_end=clean,
            same_revision=revision == git_start[0], fixed_test=fixed_test,
        ),
    })
    if dataset["name"] == "COCO" and detector.weights.name in PUBLISHED:
        value, source = PUBLISHED[detector.weights.name]
        metrics["published_coco_reference"] = {"map_50_95": value, "source": source}
    return metrics
