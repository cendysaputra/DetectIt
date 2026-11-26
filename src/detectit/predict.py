import argparse
import json
from importlib.metadata import version
from dataclasses import asdict
from pathlib import Path

import cv2

from detectit.config import load_config
from detectit.models.yolo import Detector
from detectit.utils.drawing import class_counts, draw_detections
from detectit.utils.runs import sha256


def main():
    parser = argparse.ArgumentParser(description="Detect and annotate objects in an image.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--device")
    args = parser.parse_args()

    image = cv2.imread(str(args.source))
    if image is None:
        parser.error(f"OpenCV could not read {args.source}")
    config = load_config(args.config)
    detector = Detector(config)
    detections = detector.predict(image, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = args.output / f"{args.source.stem}-detections.png"
    if not cv2.imwrite(str(rendered), draw_detections(image, detections)):
        raise OSError(f"Could not write {rendered}")
    raw = args.output / f"{args.source.stem}-detections.json"
    raw.write_text(json.dumps({
        "source": str(args.source),
        "model": config["model"],
        "weights": detector.weights.name,
        "weights_sha256": sha256(detector.weights),
        # Exported weights (ONNX, TorchScript) load without a checkpoint dict.
        "ultralytics": (detector.model.ckpt or {}).get("version"),
        "installed_ultralytics": version("ultralytics"),
        "config": config,
        "image_size": {"width": image.shape[1], "height": image.shape[0]},
        "counts": class_counts(detections),
        "detections": [asdict(detection) for detection in detections],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(detections)} detections saved to {rendered} and {raw}")


if __name__ == "__main__":
    main()
