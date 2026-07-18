"""Gap-filling mask propagation between human-verified anchors ("propagate this fix").

Hand-saved masks (src="hand") are CO-GROUND-TRUTH, alongside the geometric seeds:
they are never overwritten, and they are what propagation interpolates BETWEEN.
One run covers the object's whole visible span:

  - every src="hand" frame in the span is fed to the SAM3 video tracker as an
    anchor (add_new_mask); gaps are filled from the NEARER anchor side (forward
    pass from the first anchor, backward pass from the last, merged per frame);
  - frames BETWEEN two anchors are always proposed (pinned both sides); ones that
    fail the confidence gate are flagged "lowconf" for the preview, not dropped —
    GT needs every visible frame masked, the human judges the flagged ones;
  - frames OUTSIDE the outermost anchors are extrapolation: propagation stops at
    the first gate failure on each side (conservative at the unpinned edges);
  - the gate: geometric agreement with the reprojected seed footprint (IoU or
    seed-box coverage) OR temporal consistency with the previous accepted mask.
    Geometry alone is untrustworthy at span starts (misplaced seeds were observed
    against verified masks on real data), hence the disjunction;
  - masks that exist but predate provenance (no "src") are proposed for overwrite
    but flagged "legacy" so an old hand-fix can be spotted in the preview.

Returns proposals for PREVIEW; writes nothing. The caller writes confirmed frames
with src="prop" and must skip src="hand" regardless (belt and braces).
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

_TRACKER = None


def load_tracker():
    """Build (once) the SAM3 video tracker, SAM2-style API (validated by
    test_sam3_video_tracker.py)."""
    global _TRACKER
    if _TRACKER is None:
        from sam3.model_builder import build_sam3_video_model
        m = build_sam3_video_model()
        pred = m.tracker
        pred.backbone = m.detector.backbone
        _TRACKER = pred
    return _TRACKER


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    return None if len(xs) == 0 else [xs.min(), ys.min(), xs.max(), ys.max()]


def _box_iou_cover(a, b):
    """(IoU, inter/area_b) between boxes a and b. `cover` is robust when b (a
    sparse-lidar seed box) is much smaller than the true object bbox."""
    if a is None or b is None:
        return 0.0, 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    A = (a[2] - a[0]) * (a[3] - a[1])
    B = (b[2] - b[0]) * (b[3] - b[1])
    iou = inter / (A + B - inter) if A + B - inter > 0 else 0.0
    return iou, (inter / B if B > 0 else 0.0)


def _gate(m, prev, seed, iou_floor):
    """(geo_ok, temporal_ok) for a proposed mask vs the seed footprint and the
    previous accepted mask (which chains back to a human anchor)."""
    if m.sum() < 200:
        return False, False
    geo_ok = False
    if seed is not None:
        g_iou, g_cover = _box_iou_cover(_mask_bbox(m), seed["box"])
        geo_ok = max(g_iou, g_cover) >= iou_floor
    t_iou, _ = _box_iou_cover(_mask_bbox(m), _mask_bbox(prev))
    ratio = int(m.sum()) / max(1, int(prev.sum()))
    temporal_ok = t_iou >= 0.4 and 0.4 <= ratio <= 2.5
    return geo_ok, temporal_ok


def propagate(capture, obj_dir, anchor_name, iou_floor=0.2, max_span=600,
              mode="forward", frontier_name=None,
              progress=None, tracker=None, window=None):
    """Fill the gaps between hand anchors across the object's visible span.

    mode="forward" (default): propose ONLY frames AFTER the clicked frame — the
    annotator works forward in time; frames behind them are green-lit and a
    future fix must never flow back. mode="span" fills the whole span (opt-in
    bootstrap). `frontier_name`, if given, marks the object's "confirmed up to
    here" frame: frames at-or-before it are NEVER proposed in either mode, and
    the frontier frame's own mask (whatever its src) is fed to the tracker as an
    approved anchor.

    Returns (results, flags, meta):
      results = {frame_name: bool mask}  — proposals only; NEVER includes a
                src="hand" frame or a frame at/behind the frontier;
      flags   = {frame_name: {"lowconf": bool, "legacy": bool}};
      meta    = {"anchors": int, "span": int, "stops": [...], "invisible": int}.
    `window` kept for API compat.
    """
    import json
    import torch

    say = progress or (lambda s: None)
    od = Path(obj_dir)
    mi = json.load(open(od / "masks_index.json"))
    seeds = json.load(open(od / "seeds.json")) if (od / "seeds.json").exists() else {}
    if anchor_name not in mi:
        raise ValueError(f"anchor frame has no saved mask: {anchor_name}")
    sid = mi[anchor_name]["session"]

    fdir = Path(capture) / "sessions" / sid / "raw_data" / "images" / "cam0"
    names = sorted((f"images/cam0/{p.name}" for p in fdir.glob("*.jpg")),
                   key=lambda n: int(Path(n).stem))
    idx = {n: i for i, n in enumerate(names)}
    frontier_glob = idx.get(frontier_name, -1)       # global index; -1 = no frontier

    # visible span = everything the object touches (seeds or masks)
    cov = sorted(idx[n] for n in (set(seeds) | set(mi)) if n in idx)
    lo, hi = cov[0], cov[-1]
    a_glob = idx[anchor_name]
    if hi - lo + 1 > max_span:                       # cap huge spans around the click
        lo = max(lo, a_glob - max_span // 2)
        hi = min(hi, a_glob + max_span // 2)
    win = names[lo:hi + 1]
    click_local = win.index(anchor_name)

    hand = {n for n in win if mi.get(n, {}).get("src") == "hand"}
    anchor_names = set(hand) | {anchor_name}
    if frontier_name in mi and frontier_name in win:  # blessed frontier mask = anchor
        anchor_names.add(frontier_name)
    anchors = sorted(win.index(n) for n in anchor_names)

    def _writable(i):
        """May frame i be proposed? Not hand, not at/behind the frontier, and in
        forward mode only frames after the clicked fix."""
        if win[i] in hand or (lo + i) <= frontier_glob:
            return False
        return mode == "span" or i > click_local

    say(f"{len(anchors)} anchor(s), span {len(win)} frames, mode={mode}"
        + (f", frontier@{frontier_glob - lo}" if frontier_glob >= lo else ""))

    def _anchor_mask(local):
        m = cv2.imread(str(od / mi[win[local]]["mask_file"]), 0)
        if m is None:
            raise ValueError(f"anchor mask unreadable: {win[local]}")
        return m > 127

    tr = tracker or load_tracker()
    fwd, bwd = {}, {}
    anchor_set = set(anchors)
    with tempfile.TemporaryDirectory() as td:
        for i, n in enumerate(win):
            os.symlink(fdir / Path(n).name, Path(td) / f"{i:05d}.jpg")
        say(f"tracker: loading {len(win)} frames …")
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
               if torch.cuda.is_available() else torch.no_grad())
        with torch.inference_mode(), ctx:
            st = tr.init_state(video_path=td, offload_video_to_cpu=len(win) > 120)
            for al in anchors:
                tr.add_new_mask(st, frame_idx=al, obj_id=1,
                                mask=torch.from_numpy(_anchor_mask(al).astype(np.float32)))
            passes = ([(False, fwd, click_local)] if mode == "forward" else
                      [(False, fwd, anchors[0]), (True, bwd, anchors[-1])])
            for reverse, store, start in passes:
                say(f"propagating {'backward' if reverse else 'forward'} "
                    f"({len(win)} frames) …")
                for fi, _oids, _lr, vrm, _sc in tr.propagate_in_video(
                        st, start_frame_idx=start,
                        max_frame_num_to_track=len(win),
                        propagate_preflight=True, reverse=reverse):
                    if fi in anchor_set or win[fi] in hand:
                        continue
                    store[fi] = (vrm[0] > 0).squeeze().cpu().numpy().astype(bool)

    # merge the two passes: each frame takes the pass whose anchor side is nearer
    def _pick(i):
        if i < anchors[0]:
            return bwd.get(i, fwd.get(i))
        if i > anchors[-1]:
            return fwd.get(i, bwd.get(i))
        d_l = min(i - a for a in anchors if a <= i)
        d_r = min(a - i for a in anchors if a >= i)
        first, second = (fwd, bwd) if d_l <= d_r else (bwd, fwd)
        return first.get(i, second.get(i))

    results, flags, stops = {}, {}, []
    invisible = 0
    say("gating & merging …")

    def _consider(i, prev):
        """Gate frame i against `prev`. Returns (mask, geo_ok, temporal_ok), or
        None (hand/frontier-protected/absent/out-of-mode), or "invisible" for a
        near-empty track — object out of view, NO mask is the correct GT."""
        m = _pick(i)
        if m is None or not _writable(i):
            return None
        if m.sum() < 200:
            return "invisible"
        geo_ok, temporal_ok = _gate(m, prev, seeds.get(win[i]), iou_floor)
        return m, geo_ok, temporal_ok

    def _accept(i, m, ok):
        name = win[i]
        results[name] = m
        flags[name] = {"lowconf": not ok,
                       "legacy": name in mi and "src" not in mi[name]}

    # between anchors: ALWAYS fill (pinned both sides); gate failures only flag.
    # `prev` advances every frame (frame-to-frame smoothness): one bad frame flags
    # once instead of cascading flags down the whole gap; a smooth wrong track
    # still gets flagged where it snaps back at the next anchor.
    for a0, a1 in zip(anchors, anchors[1:]):
        prev = _anchor_mask(a0)
        for i in range(a0 + 1, a1):
            got = _consider(i, prev)
            if got is None:
                continue
            if got == "invisible":
                invisible += 1
                continue
            m, geo_ok, temporal_ok = got
            _accept(i, m, geo_ok or temporal_ok)
            prev = m
    # outside the outermost anchors: extrapolation — stop at first gate failure
    # (an invisible track at the edge also means the object is gone: stop)
    for rng, a_edge in ((range(anchors[-1] + 1, len(win)), anchors[-1]),
                        (range(anchors[0] - 1, -1, -1), anchors[0])):
        prev = _anchor_mask(a_edge)
        for i in rng:
            got = _consider(i, prev)
            if got is None:
                continue
            if got == "invisible":
                stops.append({"name": win[i], "stopped": "edge-invisible"})
                break
            m, geo_ok, temporal_ok = got
            if not (geo_ok or temporal_ok):
                stops.append({"name": win[i], "stopped": "edge-gate"})
                break
            _accept(i, m, True)
            prev = m

    meta = {"anchors": len(anchors), "span": len(win), "stops": stops,
            "invisible": invisible}
    return results, flags, meta
