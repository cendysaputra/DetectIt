def xywh_to_xyxy(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def xyxy_to_xywh(box):
    left, top, right, bottom = box
    return left, top, right - left, bottom - top


def normalized_cxcywh_to_xyxy(box, width, height):
    x, y, w, h = box
    return (x - w / 2) * width, (y - h / 2) * height, (x + w / 2) * width, (y + h / 2) * height


def xyxy_to_normalized_cxcywh(box, width, height):
    x, y, w, h = xyxy_to_xywh(box)
    return (x + w / 2) / width, (y + h / 2) / height, w / width, h / height


def clip_xyxy(box, width, height):
    left, top, right, bottom = box
    return (
        min(max(0, left), width), min(max(0, top), height),
        min(max(0, right), width), min(max(0, bottom), height),
    )


def box_iou(first, second):
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    _, _, first_width, first_height = xyxy_to_xywh(first)
    _, _, second_width, second_height = xyxy_to_xywh(second)
    union = first_width * first_height + second_width * second_height - overlap
    return overlap / union if union > 0 else 0.0
