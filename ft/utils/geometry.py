from math import hypot


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_foot(bbox):
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def bbox_width(bbox):
    return float(bbox[2] - bbox[0])


def bbox_height(bbox):
    return float(bbox[3] - bbox[1])


def distance(p1, p2):
    return hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1]))


def clip_bbox(bbox, frame):
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, int(bbox[0])))
    y1 = max(0, min(height - 1, int(bbox[1])))
    x2 = max(0, min(width, int(bbox[2])))
    y2 = max(0, min(height, int(bbox[3])))
    return [x1, y1, x2, y2]

