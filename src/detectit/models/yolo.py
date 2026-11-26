from dataclasses import dataclass
from pathlib import Path

from detectit.data.boxes import clip_xyxy


@dataclass(frozen=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]


class Detector:
    def __init__(self, config):
        from ultralytics import YOLO

        self.config = config
        weights = Path(config["weights"])
        weights.parent.mkdir(parents=True, exist_ok=True)
        if not weights.is_file() and weights.name not in {"yolo11n.pt", "yolov8n.pt"}:
            raise FileNotFoundError(f"Checkpoint not found: {weights}")
        self.model = YOLO(str(weights), task="detect")
        self.weights = Path(self.model.ckpt_path).resolve()
        self.names = self.model.names

    def predict(self, image, *, confidence=None, nms_iou=None, device=None):
        return next(self.predict_many(image, confidence=confidence, nms_iou=nms_iou, device=device))

    def predict_many(
        self, source, *, confidence=None, nms_iou=None, device=None,
        batch=1, rect=True, max_det=300,
    ):
        results = self.model.predict(
            source=source,
            imgsz=self.config["imgsz"],
            conf=self.config["confidence"] if confidence is None else confidence,
            iou=self.config["nms_iou"] if nms_iou is None else nms_iou,
            agnostic_nms=False,
            device=device,
            batch=batch,
            rect=rect,
            max_det=max_det,
            stream=True,
            verbose=False,
        )
        for result in results:
            height, width = result.orig_shape
            boxes = result.boxes
            yield [
                Detection(int(class_id), result.names[int(class_id)], score, clip_xyxy(box, width, height))
                for box, score, class_id in zip(
                    boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
                )
            ]
