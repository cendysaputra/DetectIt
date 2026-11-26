from collections import Counter

import cv2

from detectit.data.boxes import clip_xyxy


def draw_detections(image, detections):
    drawn = image.copy()
    height, width = drawn.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = (
            round(value) for value in clip_xyxy(detection.xyxy, width - 1, height - 1)
        )
        color = ((53 * detection.class_id + 64) % 192, 220, 230)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(text_height + baseline + 2, y1)
        cv2.rectangle(drawn, (x1, top - text_height - baseline - 2), (x1 + text_width + 4, top), color, -1)
        cv2.putText(drawn, label, (x1 + 2, top - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return drawn


def class_counts(detections):
    return dict(sorted(Counter(detection.label for detection in detections).items()))
