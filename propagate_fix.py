"""Anchored mask propagation ("propagate this fix", Phase 2).

Carry a human-corrected mask from ONE anchor frame to its temporal neighbours with
the SAM3 video tracker, gated per-frame by the object's geometric seeds so drift is
detected and bounded instead of silently written:

  - the scene is static within a clip (only the camera moves), so adjacent frames
    are near-duplicates -> the tracker's best-case regime;
  - every frame has a reprojected 3D footprint (seeds.json) -> if the propagated
    mask departs from it (bbox-IoU below `iou_floor`), propagation STOPS in that
    direction; frames with no seed (object occluded / out of view) also stop it;
  - results are returned for PREVIEW; nothing is written here. The caller (GUI)
    writes accepted frames with src="prop" after human confirmation.

The tracker is a separate model from the GUI's image model (~GB-scale); it is
lazy-loaded once per process and cached.
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

_TRACKER = None


def load_tracker():
    """Build (once) the SAM3 video tracker, SAM2-style API (see
    test_sam3_video_tracker.py, which validated this path)."""
    global _TRACKER
    if _TRACKER is None:
        from sam3.model_builder import build_sam3_video_model
        m = build_sam3_video_model()
        pred = m.tracker
        pred.backbone = m.detector.backbone
        _TRACKER = pred
    return _TRACKER


def _bbox_iou_mask_box(mask, box):
    """IoU between a mask's bbox and a seed box [x0,y0,x1,y1]."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0
    a = [xs.min(), ys.min(), xs.max(), ys.max()]
    ix0, iy0 = max(a[0], box[0]), max(a[1], box[1])
    ix1, iy1 = min(a[2], box[2]), min(a[3], box[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (box[2] - box[0]) * (box[3] - box[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def propagate(capture, obj_dir, anchor_name, window=25, iou_floor=0.2,
              progress=None, tracker=None):
    """Propagate the anchor frame's SAVED mask to up to `window` frames each way.

    Returns (results, info): results = {frame_name: bool mask} for frames that
    passed the geometry gate (anchor excluded); info = per-frame {"iou": float,
    "stopped": reason} diagnostics, in propagation order. Writes nothing.
    """
    import json
    import torch

    say = progress or (lambda s: None)
    od = Path(obj_dir)
    mi = json.load(open(od / "masks_index.json"))
    seeds = json.load(open(od / "seeds.json")) if (od / "seeds.json").exists() else {}
    if anchor_name not in mi:
        raise ValueError(f"anchor frame has no saved mask: {anchor_name}")
    meta = mi[anchor_name]
    sid = meta["session"]
    anchor_mask = cv2.imread(str(od / meta["mask_file"]), 0)
    if anchor_mask is None:
        raise ValueError(f"anchor mask file unreadable: {meta['mask_file']}")
    anchor_mask = anchor_mask > 127

    fdir = Path(capture) / "sessions" / sid / "raw_data" / "images" / "cam0"
    names = sorted((f"images/cam0/{p.name}" for p in fdir.glob("*.jpg")),
                   key=lambda n: int(Path(n).stem))
    a = names.index(anchor_name)
    lo, hi = max(0, a - window), min(len(names) - 1, a + window)
    win = names[lo:hi + 1]
    a_local = a - lo

    tr = tracker or load_tracker()
    results, info = {}, []
    with tempfile.TemporaryDirectory() as td:
        for i, n in enumerate(win):                     # zero-padded order for the loader
            os.symlink(fdir / Path(n).name, Path(td) / f"{i:05d}.jpg")
        say(f"tracker: loading {len(win)} frames …")
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if torch.cuda.is_available() else torch.no_grad())
        with torch.inference_mode(), ctx:
            st = tr.init_state(video_path=td)
            tr.add_new_mask(st, frame_idx=a_local, obj_id=1,
                            mask=torch.from_numpy(anchor_mask.astype(np.float32)))
            for reverse in (False, True):
                say(f"propagating {'backward' if reverse else 'forward'} …")
                for fi, _oids, _lr, vrm, _sc in tr.propagate_in_video(
                        st, start_frame_idx=a_local,
                        max_frame_num_to_track=window,
                        propagate_preflight=True, reverse=reverse):
                    if fi == a_local:
                        continue
                    name = win[fi]
                    m = (vrm[0] > 0).squeeze().cpu().numpy().astype(bool)
                    seed = seeds.get(name)
                    if seed is None:                    # occlusion gap = span boundary
                        info.append({"name": name, "iou": None, "stopped": "no-seed"})
                        break
                    iou = _bbox_iou_mask_box(m, seed["box"])
                    if iou < iou_floor:                 # drifted off the 3D footprint
                        info.append({"name": name, "iou": iou, "stopped": "drift"})
                        break
                    results[name] = m
                    info.append({"name": name, "iou": iou, "stopped": None})
    return results, info
