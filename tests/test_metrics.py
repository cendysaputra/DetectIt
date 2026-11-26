import torch
from pycocotools.coco import COCO
from ultralytics.utils.nms import non_max_suppression

from detectit.eval.metrics import coco_metrics, operating_counts
from detectit.models.yolo import Detection
from detectit.utils.runs import is_reportable


def test_operating_point_matches_once_by_score_and_class():
    box = (0, 0, 5, 5)
    detections = [
        Detection(0, "helmet", 0.3, box),
        Detection(0, "helmet", 0.8, box),
        Detection(1, "no helmet", 0.9, box),
        Detection(1, "no helmet", 0.1, (5, 5, 10, 10)),
    ]
    annotations = [(0, box), (1, (5, 5, 10, 10))]
    assert operating_counts(detections, annotations, 0.5, 0.25) == (1, 2, 1)
    assert operating_counts(detections, annotations, 0.5, 0.5) == (1, 1, 1)


def test_nms_keeps_overlapping_boxes_of_different_classes():
    prediction = torch.tensor([[
        [20, 20, 20], [20, 20, 20], [10, 10, 10], [10, 10, 10],
        [0.95, 0.8, 0.0], [0.0, 0.0, 0.85],
    ]])
    kept = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.5, nc=2)[0]
    assert kept[:, 5].tolist() == [0, 1]


def test_perfect_and_empty_coco_scores():
    ground_truth = COCO()
    ground_truth.dataset = {
        "info": {},
        "images": [{"id": 1}],
        "categories": [{"id": 7, "name": "helmet"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 7, "bbox": [0, 0, 8, 8],
            "area": 64, "iscrowd": 0,
        }],
    }
    ground_truth.createIndex()
    prediction = {"image_id": 1, "category_id": 7, "bbox": [0, 0, 8, 8], "score": 0.99}
    assert coco_metrics(ground_truth, [prediction], [1])["map_50_95"] > 0.99
    assert coco_metrics(ground_truth, [], [1])["map_50_95"] == 0


def test_reportable_requires_full_clean_fixed_run():
    flags = dict(full_split=True, clean_start=True, clean_end=True, same_revision=True, fixed_test=True)
    assert is_reportable(**flags)
    for flag in flags:
        assert not is_reportable(**(flags | {flag: False}))
