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
  - frames OUTSIDE the outermost anchors are extrapolation. In forward mode the
    window extends past the last covered frame to the end of the walk (max_span-
    capped): one click carries the object's whole remaining presence, INCLUDING
    across off-screen gaps between visits (invisible frames propose nothing and
    never stop the run). The run tolerates brief wobble — up to `patience`
    consecutive gate failures before stopping; wobble frames that recover are
    proposed flagged "lowconf", a trailing run that never recovers is dropped;
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
# extra anchors sampled from the human-vouched region at/behind the frontier
# ("good up to here" teaches the tracker, not just protects). 0 disables.
BLESSED_EXTRA_CAP = 40
# a multi-SAM object (many disjoint blobs — e.g. 14 windows) is tracked as ONE
# SAM3 object today, and all but one blob fades. Instead track each blob as its
# OWN object in the same session (SAM3 is natively multi-object, so it is one
# parallel pass, not N runs); the per-frame output is their UNION, so nothing
# downstream sees parts. Capped so a pathological split can't explode memory.
MULTIPART_CAP = 96


def _split_components(mask, min_area=120):
    """Connected components of a bool mask, each >= min_area px (else the whole
    mask as one). cv2 import is local to keep this module's top clean."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    out = [lab == i for i in range(1, n) if int((lab == i).sum()) >= min_area]
    return out or [mask.astype(bool)]


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
    previous accepted mask (which chains back to a human anchor). temporal_ok
    may be the string "exit" (truthy): the mask shrinks IN PLACE while touching
    the image border — the object is leaving the frame, not the track drifting;
    partial masks during the exit are valid GT (proposed flagged lowconf), and
    stopping there was measured to strand every visit after the first
    (cardboard_boxes: edge-gate death at t=12.0s, five visits unreached)."""
    if m.sum() < 200:
        return False, False
    geo_ok = False
    if seed is not None:
        bx = seed["box"]
        if (bx[2] - bx[0]) * (bx[3] - bx[1]) > 0.4 * m.size:
            seed = None       # frame-sized box (legacy degenerate seeds) is no
    if seed is not None:      # evidence — fall back to temporal-only
        g_iou, g_cover = _box_iou_cover(_mask_bbox(m), seed["box"])
        geo_ok = max(g_iou, g_cover) >= iou_floor
    bm, bp = _mask_bbox(m), _mask_bbox(prev)
    t_iou, _ = _box_iou_cover(bm, bp)
    ratio = int(m.sum()) / max(1, int(prev.sum()))
    temporal_ok = t_iou >= 0.4 and 0.4 <= ratio <= 2.5
    if not temporal_ok and bm and bp and ratio <= 1.0:
        _, cover_m = _box_iou_cover(bp, bm)          # how much of m sits in prev's box
        border = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
        if cover_m >= 0.8 and border:
            temporal_ok = "exit"
    return geo_ok, temporal_ok


def propagate(capture, obj_dir, anchor_name, iou_floor=0.2, max_span=600,
              mode="forward", frontier_name=None,
              progress=None, tracker=None, window=None):
    """Fill the gaps between hand anchors across the object's visible span.

    mode="forward" (default): propose ONLY frames AFTER the clicked frame — the
    annotator works forward in time; frames behind them are green-lit and a
    future fix must never flow back. Forward reach extends to the end of the
    walk (max_span-capped), even into frames with no mask/seed coverage yet —
    the patience stop rule ends the run when the object leaves view or the
    track degrades for good. mode="span" fills the whole span (opt-in
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
    if mode == "forward":
        # carry past the last covered frame to the end of the walk — for a
        # hand-only object cov collapses to the already-masked frames, which
        # left nothing to propose INTO. The patience stop rule below decides
        # where the object actually ends; max_span still caps the window.
        hi = len(names) - 1
    if hi - lo + 1 > max_span:                       # cap huge spans around the click
        lo = max(lo, a_glob - max_span // 2)
        hi = min(hi, a_glob + max_span // 2)
    win = names[lo:hi + 1]
    click_local = win.index(anchor_name)

    hand = {n for n in win if mi.get(n, {}).get("src") == "hand"}
    anchor_names = set(hand) | {anchor_name}
    if frontier_name in mi and frontier_name in win:  # blessed frontier mask = anchor
        anchor_names.add(frontier_name)
    # "good up to here" TEACHES, not just protects: every masked frame at/behind
    # the frontier is human-vouched (g), whatever its src — feed it to the
    # tracker as an anchor so fills inherit approved shape/scale, not just the
    # hand frames'. Sampled to bound tracker memory: the frames nearest the
    # click teach current appearance, a uniform sweep of the rest teaches the
    # object from the walk's other viewpoints.
    blessed = sorted((n for n in win if n in mi and (lo + win.index(n)) <= frontier_glob
                      and n not in anchor_names),
                     key=lambda n: abs(win.index(n) - click_local))
    if blessed and BLESSED_EXTRA_CAP > 0:
        near = blessed[:10]
        rest = blessed[10:]
        step = max(1, len(rest) // max(1, BLESSED_EXTRA_CAP - len(near)))
        anchor_names |= set(near) | set(rest[::step][:BLESSED_EXTRA_CAP - len(near)])
    # prevalidate: a broken mask file must drop that anchor, not abort the run
    # (the clicked anchor stays load-bearing and still errors loudly below)
    for n in sorted(anchor_names - {anchor_name}):
        p = od / mi[n]["mask_file"]
        if cv2.imread(str(p), 0) is None:
            anchor_names.discard(n)
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
            # Multi-part fires ONLY on several COMPARABLE, separated blobs
            # (many windows) — NOT on a solid object with a small satellite
            # (that keeps the strong single-object multi-anchor path). A blob
            # counts if it is >= 25% of the largest and >= 800 px; >=2 such =
            # multi-part.
            click_comps = _split_components(_anchor_mask(click_local))
            big = max((int(c.sum()) for c in click_comps), default=0)
            parts = [c for c in click_comps
                     if int(c.sum()) >= max(800, 0.25 * big)]
            if len(parts) < 2:
                for al in anchors:                   # single-object, multi-anchor
                    tr.add_new_mask(st, frame_idx=al, obj_id=1,
                                    mask=torch.from_numpy(_anchor_mask(al).astype(np.float32)))
                n_obj = 1
            else:
                # seed EVERY drawn blob from EVERY anchor frame — a window seen
                # only in another seed frame (e.g. the lower windows) would
                # otherwise never be tracked. Redundant tracks of the same window
                # across frames just re-vote into the union (harmless); capped
                # biggest-first so memory stays bounded.
                plan = []                            # (area, frame_idx, comp)
                for al in anchors:
                    am = _anchor_mask(al)
                    cc = _split_components(am)
                    b = max((int(c.sum()) for c in cc), default=0)
                    for c in cc:
                        if int(c.sum()) >= max(600, 0.12 * b):
                            plan.append((int(c.sum()), al, c))
                plan.sort(reverse=True, key=lambda t: t[0])
                for oid, (_a, al, c) in enumerate(plan[:MULTIPART_CAP], start=1):
                    tr.add_new_mask(st, frame_idx=al, obj_id=oid,
                                    mask=torch.from_numpy(c.astype(np.float32)))
                n_obj = min(len(plan), MULTIPART_CAP)
                say(f"multi-part: {len(plan)} blobs across {len(anchors)} anchor "
                    f"frame(s) -> {n_obj} component tracks (capped {MULTIPART_CAP})")
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
                    a = (vrm > 0).cpu().numpy()      # [n_obj, 1, H, W] (or [1,..])
                    store[fi] = a.reshape(-1, a.shape[-2], a.shape[-1]).any(0)

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
            _accept(i, m, geo_ok or temporal_ok is True)   # "exit" passes, flagged
            prev = m
    # outside the outermost anchors: extrapolation into virgin frames. Invisible
    # frames (object out of view, <200px) propose nothing — no mask IS the GT —
    # and NEVER stop the run: a walk revisits an object many times, and the fill
    # must bridge the off-screen gaps between visits just like the between-
    # anchor path does (stopping at the first gap was measured to kill the run
    # at t=11.9s on cardboard_boxes with five more visits ahead). Gate failures
    # are the only stop signal: up to `patience` consecutive ones are buffered
    # and proposed flagged "lowconf" if the track recovers. `prev` (the temporal
    # reference) advances only on gate-passing frames, so recovery is judged
    # against the last GOOD mask — a smoothly drifting wrong track can't
    # re-green-light itself. Patience exhausted -> stop, DROP the trailing
    # failed run (a track that never recovers is garbage).
    # patience exhausted no longer ABORTS the walk: the run goes "lost" —
    # proposes nothing — and RESUMES when the seed geometry re-acquires the
    # object on 2 consecutive frames (a single fluke match must not restart
    # it). This is what lets one click span a multi-visit walk even when the
    # tracker jumps to a wrong object at a visit boundary (measured: red
    # printer at t=11.5s on cardboard_boxes): the wrong stretch is dropped,
    # the next visit still gets filled. No trustworthy seeds ahead -> lost
    # stays lost, which equals the old stop.
    patience = 5
    resumes = []
    for rng, a_edge in ((range(anchors[-1] + 1, len(win)), anchors[-1]),
                        (range(anchors[0] - 1, -1, -1), anchors[0])):
        prev = _anchor_mask(a_edge)
        misses, buffered = 0, []          # buffered: gate-failed (i, m) awaiting recovery
        lost, pend = False, None          # pend: first geo-confirmed frame while lost
        for i in rng:
            got = _consider(i, prev)
            if got is None:
                continue
            if got == "invisible":
                invisible += 1
                pend = None               # an off-screen gap breaks a confirmation pair
                continue
            m, geo_ok, temporal_ok = got
            if lost:
                if geo_ok:
                    if pend is None:
                        pend = (i, m)
                    else:                 # 2nd consecutive geo pass -> resume
                        _accept(pend[0], pend[1], True)
                        _accept(i, m, True)
                        prev = m
                        lost, pend = False, None
                        resumes.append(win[i])
                else:
                    pend = None
                continue
            if geo_ok or temporal_ok:
                for bi, bm in buffered:
                    _accept(bi, bm, False)
                buffered, misses = [], 0
                _accept(i, m, geo_ok or temporal_ok is True)  # "exit" -> lowconf
                prev = m
            else:
                misses += 1
                if misses > patience:
                    stops.append({"name": win[i], "stopped": "edge-gate"})
                    lost, buffered, misses = True, [], 0
                    continue
                buffered.append((i, m))

    meta = {"anchors": len(anchors), "span": len(win), "stops": stops,
            "invisible": invisible, "resumes": resumes}
    return results, flags, meta
