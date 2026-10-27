"""Refine one mask component, preserving the other blobs (no disk writes)."""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class BrushConstants:
    merge_px: float = 3
    snap_px: float = 15
    hole_abs_px: float = 400
    hole_rel: float = 0.01
    box_pad_px: float = 8
    n_pos: int = 4
    sam_min_iou: float = 0.5
    sam_max_grow: float = 1.5
    trim_px: float = 2


CONNECT = np.ones((3, 3), bool)


def holes(mask):
    return ndi.binary_fill_holes(mask, structure=CONNECT) & ~mask


def inside_distance(mask):
    # Explicit zero border also handles masks touching the image boundary.
    return ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]


def select_blob(mask, point, constants=BrushConstants()):
    scale = np.sqrt(mask.size) / 512
    radius = max(0, round(constants.merge_px * scale))
    closed = mask.copy()
    if radius:
        padded = np.pad(mask.astype(np.uint8), radius)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[radius:-radius, radius:-radius] > 0
    labels, n = ndi.label(closed, CONNECT)
    x, y = point
    chosen = labels[y, x]
    if not chosen:
        # Prefer the innermost enclosing component when rings are nested.
        enclosing = [(int((labels == i).sum()), i) for i in range(1, n + 1)
                     if holes(labels == i)[y, x]]
        if enclosing:
            chosen = min(enclosing)[1]
    if not chosen and n:
        distance, nearest = ndi.distance_transform_edt(~closed, return_indices=True)
        if distance[y, x] <= constants.snap_px * scale:
            chosen = labels[nearest[0, y, x], nearest[1, y, x]]
    component = labels == chosen if chosen else np.zeros_like(mask)
    return component, mask & component


def interior_points(mask, click, n_pos, scale):
    candidates = np.argwhere(inside_distance(mask) > max(1, round(2 * scale)))[:, ::-1]
    points = [np.asarray(click, float)] if mask[click[1], click[0]] else []
    if not len(candidates):
        # Thin fragments may have no eroded interior; keep an in-mask prompt.
        if not points:
            y, x = np.unravel_index(np.argmax(inside_distance(mask)), mask.shape)
            points.append(np.array([x, y], float))
        return np.asarray(points, np.float32)
    scores = np.full(len(candidates), np.inf)
    if points:
        scores = ((candidates - points[0]) ** 2).sum(1)
    else:
        # Deterministic first sample: deepest interior point, then farthest points.
        distance = inside_distance(mask)
        first = np.argmax(distance[candidates[:, 1], candidates[:, 0]])
        scores[first] = np.inf
        scores[np.arange(len(scores)) != first] = 0
    for _ in range(min(n_pos, len(candidates))):
        best = int(np.argmax(scores))
        if scores[best] <= 0:
            break
        point = candidates[best]
        points.append(point.astype(float))
        d2 = ((candidates - point) ** 2).sum(1)
        scores = d2 if len(points) == 1 else np.minimum(scores, d2)
    return np.asarray(points, np.float32).reshape(-1, 2)


def refine_blob(image, mask, point, shift, refine, constants=BrushConstants(), excluded=False):
    """refine(image_rgb, points, labels, box, prior) invokes the existing SAM helper.

    Growth is allowed in the padded box, as clarified by the user. Other blobs
    and their enclosed holes are protected; only this blob can lose pixels.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or image.shape[:2] != mask.shape:
        raise ValueError("Mask dimensions do not match the grayscale frame")
    point = np.asarray(point, float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Expected a finite [x, y] click")
    x, y = np.floor(point).astype(int)
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError("Click is outside the frame")
    point = (x, y)
    result = dict(mask=mask.copy(), added=np.zeros_like(mask), removed=np.zeros_like(mask),
                  holes_filled=0, nothing_near_click=False, sam_rejected=False, excluded=bool(excluded or mask.all()))
    if result["excluded"]:
        return result
    component, blob = select_blob(mask, point, constants)
    if not blob.any():
        result["nothing_near_click"] = True
        return result
    others = mask & ~blob
    protected = ndi.binary_fill_holes(others, structure=CONNECT)
    # Closing may itself erase pinholes. Recover the original holes too, while
    # never inserting the closing's exterior bridges merely because we selected them.
    hole_pixels = (holes(blob) | holes(component)) & ~mask & ~protected
    labels, count = ndi.label(hole_pixels, CONNECT)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    scale = np.sqrt(mask.size) / 512
    cap = max(constants.hole_abs_px * scale ** 2, constants.hole_rel * int(blob.sum()))
    small_ids = np.flatnonzero((sizes <= cap) & (np.arange(len(sizes)) > 0))
    small = np.isin(labels, small_ids)
    filled = blob | small
    ys, xs = np.where(filled)
    pad = max(0, round(constants.box_pad_px * scale))
    box = [max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
           min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)]
    points = interior_points(filled, point, constants.n_pos, scale)
    rgb = np.repeat(image[:, :, None], 3, axis=2) if image.ndim == 2 else image
    sam = np.asarray(refine(rgb, points, np.ones(len(points), np.int32), box, filled), bool)
    if sam.shape != mask.shape:
        raise ValueError("SAM returned a mask with the wrong dimensions")
    iou = int((sam & filled).sum()) / max(1, int((sam | filled).sum()))
    rejected = iou < constants.sam_min_iou or sam.sum() > constants.sam_max_grow * filled.sum()
    result.update(holes_filled=len(small_ids), sam_rejected=bool(rejected))
    if rejected:
        refined = filled
    else:
        allowed = np.zeros_like(mask)
        allowed[box[1]:box[3], box[0]:box[2]] = True
        refined = filled | (sam & allowed & ~protected)
        if not shift:
            trim = filled & ~sam & (inside_distance(filled) <= constants.trim_px * scale) & ~small
            refined &= ~trim
    effective = others | refined
    result.update(mask=effective, added=effective & ~mask, removed=mask & ~effective)
    return result
