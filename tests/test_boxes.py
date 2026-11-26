import pytest

from detectit.data.boxes import (
    box_iou,
    clip_xyxy,
    normalized_cxcywh_to_xyxy,
    xywh_to_xyxy,
    xyxy_to_normalized_cxcywh,
    xyxy_to_xywh,
)


def test_box_format_round_trip():
    xyxy = (10, 20, 110, 80)
    coco = (10, 20, 100, 60)
    normalized = (0.3, 0.5, 0.5, 0.6)
    assert xyxy_to_xywh(xyxy) == coco
    assert xywh_to_xyxy(coco) == xyxy
    assert xyxy_to_normalized_cxcywh(xyxy, 200, 100) == pytest.approx(normalized)
    assert normalized_cxcywh_to_xyxy(normalized, 200, 100) == pytest.approx(xyxy)
    assert clip_xyxy((-1, -2, 205, 80), 200, 100) == (0, 0, 200, 80)


def test_iou_identity_disjoint_and_partial():
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1
    assert box_iou((0, 0, 2, 2), (2, 0, 3, 2)) == 0
    assert box_iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
