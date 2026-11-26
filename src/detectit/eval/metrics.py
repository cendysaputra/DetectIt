from collections import defaultdict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detectit.data.boxes import box_iou, xywh_to_xyxy, xyxy_to_xywh


def operating_counts(detections, annotations, matching_iou, confidence):
    by_class = defaultdict(list)
    for category, box in annotations:
        by_class[category].append(box)
    tp = fp = 0
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if detection.confidence < confidence:
            break
        candidates = by_class[detection.class_id]
        overlaps = [box_iou(detection.xyxy, box) for box in candidates]
        if overlaps and max(overlaps) >= matching_iou:
            candidates.pop(overlaps.index(max(overlaps)))
            tp += 1
        else:
            fp += 1
    return tp, fp, sum(map(len, by_class.values()))


def coco_predictions(image_id, detections, categories):
    return [
        {
            "image_id": image_id,
            "category_id": categories[detection.class_id],
            "bbox": list(xyxy_to_xywh(detection.xyxy)),
            "score": detection.confidence,
        }
        for detection in detections
    ]


def coco_metrics(ground_truth, predictions, image_ids):
    if predictions:
        results = ground_truth.loadRes(predictions)
    else:
        results = COCO()
        results.dataset = {
            "images": ground_truth.loadImgs(image_ids),
            "categories": list(ground_truth.cats.values()),
            "annotations": [],
        }
        results.createIndex()
    evaluator = COCOeval(ground_truth, results, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    precision = evaluator.eval["precision"][:, :, :, 0, -1]
    per_class = {}
    for index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, index]
        ap = values[values >= 0]
        ap50 = values[0][values[0] >= 0]
        name = ground_truth.cats[category_id]["name"]
        per_class[name] = {
            "ap_50_95": float(ap.mean()) if ap.size else None,
            "ap_50": float(ap50.mean()) if ap50.size else None,
        }
    return {
        "map_50": float(evaluator.stats[1]) if evaluator.stats[1] >= 0 else None,
        "map_50_95": float(evaluator.stats[0]) if evaluator.stats[0] >= 0 else None,
        "per_class_ap": per_class,
    }


def operating_annotations(ground_truth, image_id, categories):
    class_ids = {category: class_id for class_id, category in categories.items()}
    return [
        (class_ids[item["category_id"]], xywh_to_xyxy(item["bbox"]))
        for item in ground_truth.imgToAnns[image_id]
        if not item.get("iscrowd", 0)
    ]
