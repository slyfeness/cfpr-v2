"""Side-aware association between expected DWPose hand boxes and WiLoR regions."""
from __future__ import annotations

import numpy as np


def xyxy_iou(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return 0.0
    x0, y0 = np.maximum(a[:2], b[:2])
    x1, y1 = np.minimum(a[2:], b[2:])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _one_side(regions, expected_box, expected_is_right, min_iou):
    expected_box = np.asarray(expected_box, dtype=float)
    if expected_box.shape != (4,) or not np.isfinite(expected_box).all():
        return np.nan, np.nan
    candidates = [region for region in regions
                  if bool(region["canonical_is_right"]) == bool(expected_is_right)
                  and xyxy_iou(region["box"], expected_box) >= min_iou]
    if not candidates:
        return 0.0, 1.0
    best = max(candidates, key=lambda r: xyxy_iou(r["box"], expected_box))
    return float(np.mean(np.asarray(best["visibility"], dtype=float))), 0.0


def associate_expected_hands(regions, left_box, right_box, min_iou=0.3):
    vis_left, missing_left = _one_side(regions, left_box, False, min_iou)
    vis_right, missing_right = _one_side(regions, right_box, True, min_iou)
    return {"vis_left": vis_left, "vis_right": vis_right,
            "missing_left": missing_left, "missing_right": missing_right}
