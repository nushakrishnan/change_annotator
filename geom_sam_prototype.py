"""Geometry-assisted annotation prototype (1 object, end-to-end).

Two-stage pipeline across two envs (file handoff):

  src-mask  [sam3_env]  : click on source frame -> SAM3 image mask  -> src_mask.png
  seeds     [lamar_env] : lift mask onto mesh (raycast), reproject the 3D object
                          into every frame with a mesh-depth occlusion test
                          -> per-frame seed point(s)+box -> seeds.json
  perframe  [sam3_env]  : SAM3 image predict(points+box) per frame  -> per-frame masks
  track     [sam3_env]  : SAM3 *video* tracker, per visible span, seeded by the geom
                          mask -> temporally-consistent masks (refines perframe)

Run (within aria_a_rgb):
  ~/sam3_env/bin/python geom_sam_prototype.py src-mask  --capture C --session aria_a_rgb \
        --src-name images/cam0/16700526188301.jpg --click 300 180
  PYTHONPATH=~/repos/lamaria-indoor ~/lamar_env/bin/python geom_sam_prototype.py seeds \
        --capture C --session aria_a_rgb --ref navvis_a --src-name images/cam0/16700526188301.jpg --n 40
  ~/sam3_env/bin/python geom_sam_prototype.py perframe --capture C --session aria_a_rgb
"""
import argparse
import json
from pathlib import Path

import numpy as np
import cv2

OUT = "changes/geom_sam_out"


def out_dir(capture, obj=None):
    d = Path(capture) / OUT
    if obj:
        d = d / obj
    d.mkdir(parents=True, exist_ok=True)
    return d


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _load_sam_image():
    from sam3 import build_sam3_image_model
    try:
        from sam3 import Sam3Processor
    except Exception:
        from sam3.model.sam3_image_processor import Sam3Processor
    try:
        model = build_sam3_image_model(enable_inst_interactivity=True)
    except TypeError:
        model = build_sam3_image_model()
    return model, Sam3Processor(model)


def _segment(model, processor, img_rgb, points=None, labels=None, box=None, multimask=True):
    """SAM3 image segmentation via processor.set_image + model.predict_inst."""
    import torch
    from PIL import Image as PILImage
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _null()
    with torch.inference_mode(), ctx:
        state = processor.set_image(PILImage.fromarray(img_rgb))  # processor expects PIL (W,H)
        kw = dict(multimask_output=multimask)
        if points is not None:
            kw["point_coords"] = np.asarray(points, np.float32)
            kw["point_labels"] = np.asarray(labels, np.int32)
        if box is not None:
            kw["box"] = np.asarray(box, np.float32)
        masks, scores, _ = model.predict_inst(state, **kw)
    return np.asarray(masks), np.asarray(scores)


# ───────────────────────── stage: src-mask (sam3_env) ─────────────────────────
def cmd_src_mask(args):
    cap = Path(args.capture)
    rgb_path = cap / "sessions" / args.session / "raw_data" / args.src_name
    img = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

    pos = np.array(args.points, np.float32).reshape(-1, 2)
    neg = np.array(args.neg, np.float32).reshape(-1, 2) if args.neg else np.zeros((0, 2), np.float32)
    pts = np.concatenate([pos, neg], 0)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)

    model, processor = _load_sam_image()
    masks, scores = _segment(model, processor, img, points=pts, labels=labels, multimask=True)
    best = masks[int(np.argmax(scores))].astype(bool)
    d = out_dir(cap, args.obj)
    cv2.imwrite(str(d / "src_mask.png"), (best * 255).astype(np.uint8))
    ov = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ov[best] = (0.5 * ov[best] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for px, py in pos:
        cv2.circle(ov, (int(px), int(py)), 6, (0, 255, 0), -1)   # green = positive
    for px, py in neg:
        cv2.circle(ov, (int(px), int(py)), 6, (255, 0, 0), -1)   # blue = negative
    cv2.imwrite(str(d / "src_mask_overlay.png"), ov)
    json.dump({"src_name": args.src_name, "points": pos.tolist(), "neg": neg.tolist(),
               "mask_px": int(best.sum()), "score": float(scores.max())},
              open(d / "src_meta.json", "w"))
    print(f"src mask: {int(best.sum())} px, score {scores.max():.3f} "
          f"({len(pos)} pos, {len(neg)} neg) -> {d/'src_mask_overlay.png'}")


# ───────────────────────── stage: seeds (lamar_env) ─────────────────────────
def _session(cap, sid, ref):
    from scantools.capture import Capture
    from scantools.capture.timed_reconstruction import TimedReconstruction
    from scantools.capture.colmap_session import ColmapQuerySession
    capo = Capture.load(Path(cap))
    cmp = capo.session_path(sid) / "proc" / ref / "colmap_model_aligned"
    tr = TimedReconstruction.read(cmp.parent, colmap_model=cmp.name)
    sess = ColmapQuerySession(reconstruction=tr.reconstruction, data_path=capo.data_path(sid),
                              id=sid, _explicit_timestamps=tr.timestamps)
    return capo, sess


def _load_T(path):
    """Read a 7-tuple (qw qx qy qz tx ty tz) NavVis->NavVis bridge as a Pose3d."""
    from scantools.capture.pose import Pose3d
    line = next(l for l in Path(path).read_text().splitlines()
                if l.strip() and not l.startswith("#"))
    qw, qx, qy, qz, tx, ty, tz = (float(v) for v in line.split()[:7])
    return Pose3d.from_list([qw, qx, qy, qz, tx, ty, tz])


def _apply_T(T, pts):
    """Apply a Pose3d (R, t) to an (N, 3) point array."""
    R = np.asarray(T.R, dtype=np.float64)
    t = np.asarray(T.t, dtype=np.float64).reshape(3)
    return (R @ np.asarray(pts, dtype=np.float64).T).T + t


def _scaled_camera(cam, s):
    """A downscaled pinhole copy of `cam` for cheap occlusion-depth rendering.

    The occlusion test only samples depth at reprojected points within a coarse
    tolerance, so rendering at a fraction of full resolution is ~10x faster with
    sub-mm depth error. Returns (camera, sx, sy); falls back to the original
    camera (sx=sy=1) for s>=1 or non-pinhole models.
    """
    from scantools.capture.sensors import Camera
    if s >= 0.999 or cam.model_name not in ("PINHOLE", "SIMPLE_PINHOLE"):
        return cam, 1.0, 1.0
    W, H = int(round(cam.width * s)), int(round(cam.height * s))
    sx, sy = W / cam.width, H / cam.height
    fx, fy, cx, cy = cam.projection_params
    return Camera("PINHOLE", [W, H, fx * sx, fy * sy, cx * sx, cy * sy]), sx, sy


def _seeds_for_session(pts3d, sess, renderer, capo, sid, n, min_vis, dbg_root,
                       skip_name=None, occ_tol=0.10, occ_scale=0.35):
    """Reproject `pts3d` (already expressed in `sess`'s world frame) into cam0
    frames of `sess`, occlusion-tested against `renderer` (which must hold
    `sess`'s OWN state mesh). Returns {image_name: seed} and writes debug
    overlays under dbg_root/sid/.

    n <= 0 (or n >= #frames) seeds EVERY frame — the thorough default — so the
    object is masked in every frame where it is actually visible, not just a
    sparse sample. n > 0 evenly subsamples n frames (a quick coarse pass).
    Occlusion depth is rendered at `occ_scale` of full resolution (10x faster,
    sub-mm error vs the cm-scale tolerance).

    The occlusion test uses the target state's geometry on purpose: for an
    object that exists only in the source state, its reprojected footprint
    lands on the vacated/arrival region in the other state (the change signal),
    and is only dropped where the target-state geometry genuinely occludes it.
    """
    from scantools.utils.geometry import project, sample_depth

    keys = [k for k in sess.images.key_pairs() if "cam0" in str(k[1])]
    keys.sort(key=lambda k: k[0])
    if n and 0 < n < len(keys):
        idx = np.linspace(0, len(keys) - 1, n).round().astype(int)
        targets = [keys[i] for i in sorted(set(idx))]
    else:
        targets = keys  # thorough: every frame

    dbg = dbg_root / sid
    dbg.mkdir(parents=True, exist_ok=True)
    occ_cams = {}  # cam_t -> (downscaled camera, sx, sy), built once per sensor
    seeds = {}
    for j, (ts_t, cam_t) in enumerate(targets):
        if j and j % 100 == 0:
            print(f"  [{sid}] occlusion-testing frame {j}/{len(targets)} "
                  f"({len(seeds)} seeds so far)", flush=True)
        name = str(sess.images[ts_t, cam_t])
        if skip_name is not None and name == skip_name:
            continue
        cam_o = sess.sensors[cam_t]
        if cam_t not in occ_cams:
            occ_cams[cam_t] = _scaled_camera(cam_o, occ_scale)
        cam_s, sx, sy = occ_cams[cam_t]
        T_t = sess.get_pose(ts_t, cam_t)
        p2d_t, z_t, vis = project(pts3d, cam_o, pose=T_t.inverse())
        if not vis.any():
            continue
        _, mesh_depth = renderer.render_from_capture(T_t, cam_s)
        occ_z, occ_valid = sample_depth(p2d_t[vis] * np.array([sx, sy]), mesh_depth)
        visible = occ_valid & (z_t[vis] <= occ_z + occ_tol)
        pv = p2d_t[vis][visible]
        if len(pv) < min_vis:
            continue
        cen = pv.mean(0)
        x0, y0 = pv.min(0)
        x1, y1 = pv.max(0)
        seeds[name] = {"session": sid,
                       "points": [[float(cen[0]), float(cen[1])]], "labels": [1],
                       "box": [float(x0), float(y0), float(x1), float(y1)],
                       "n_visible": int(len(pv))}
        im = cv2.imread(str(capo.data_path(sid) / name))
        if im is not None:
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
            cv2.circle(im, (int(cen[0]), int(cen[1])), 5, (0, 0, 255), -1)
            cv2.imwrite(str(dbg / Path(name).name), im)
    return seeds


def cmd_seeds(args):
    from scantools.proc.rendering import Renderer, compute_rays
    from scantools.utils.io import read_mesh

    cap = Path(args.capture)
    od = out_dir(cap, args.obj)
    capo, sess = _session(cap, args.session, args.ref)
    mesh_path = capo.proc_path(args.ref) / capo.sessions[args.ref].proc.meshes["mesh"]
    renderer = Renderer(read_mesh(mesh_path))

    # locate source key + pose (the frame the mask was drawn on)
    src_key = next(k for k in sess.images.key_pairs()
                   if str(sess.images[k[0], k[1]]) == args.src_name)
    ts_s, cam_s = src_key
    cam_src = sess.sensors[cam_s]
    T_src = sess.get_pose(ts_s, cam_s)

    # lift mask -> 3D points on the SOURCE-state mesh
    mask = cv2.imread(str(od / "src_mask.png"), 0) > 127
    rows, cols = np.where(mask)
    if len(cols) > 4000:
        sel = np.random.RandomState(0).choice(len(cols), 4000, replace=False)
        rows, cols = rows[sel], cols[sel]
    p2d = np.stack([cols, rows], 1).astype(np.float32)
    o, d = compute_rays(T_src, cam_src, p2d=p2d)
    pts3d, valid = renderer.compute_intersections((o, d))
    pts3d = np.asarray(pts3d)
    # robust outlier removal (drop mask-edge background bleed)
    c = np.median(pts3d, 0)
    dist = np.linalg.norm(pts3d - c, axis=1)
    pts3d = pts3d[dist < np.percentile(dist, 90)]
    print(f"object 3D points: {len(pts3d)} in {args.ref} frame (median center {c.round(2)})")

    dbg_root = od / "seed_dbg"

    # ── propagate WITHIN the source state (occlusion vs source mesh) ──
    seeds = _seeds_for_session(pts3d, sess, renderer, capo, args.session,
                               args.n, args.min_vis, dbg_root, skip_name=args.src_name,
                               occ_scale=args.occ_scale)
    print(f"[{args.session}] within-state seeds: {len(seeds)} frames "
          f"(occlusion vs {args.ref})")
    renderer = None  # release before constructing the second renderer

    # ── propagate ACROSS into the other state via the NavVis->NavVis bridge ──
    if not args.no_cross:
        bridge = Path(args.bridge) if args.bridge else (
            cap / "changes" / f"{args.other_ref}_to_{args.ref}"
            / f"T_{args.ref}_from_{args.other_ref}.txt")
        T_ref_from_other = _load_T(bridge)
        # bridge maps other-frame -> source(ref)-frame; we need the inverse to
        # carry the source object's 3D points into the other session's frame.
        pts3d_other = _apply_T(T_ref_from_other.inverse(), pts3d)
        capo_o, sess_o = _session(cap, args.other_session, args.other_ref)
        mesh_o = (capo_o.proc_path(args.other_ref)
                  / capo_o.sessions[args.other_ref].proc.meshes["mesh"])
        renderer_o = Renderer(read_mesh(mesh_o))
        seeds_o = _seeds_for_session(pts3d_other, sess_o, renderer_o, capo_o,
                                     args.other_session, args.n, args.min_vis, dbg_root,
                                     occ_scale=args.occ_scale)
        print(f"[{args.other_session}] cross-state seeds: {len(seeds_o)} frames "
              f"(via {bridge.name}, occlusion vs {args.other_ref})")
        renderer_o = None
        seeds.update(seeds_o)

    json.dump(seeds, open(od / "seeds.json", "w"), indent=1)
    n_src = sum(1 for s in seeds.values() if s["session"] == args.session)
    print(f"total seeds: {len(seeds)} "
          f"({args.session}={n_src}, other={len(seeds) - n_src}) "
          f"-> {od/'seeds.json'}")


# ───────────────────────── stage: perframe (sam3_env) ─────────────────────────
def cmd_perframe(args):
    cap = Path(args.capture)
    od = out_dir(cap, args.obj)
    seeds = json.load(open(od / "seeds.json"))
    model, processor = _load_sam_image()

    masks_dir = od / "masks"
    if args.save_masks:
        masks_dir.mkdir(exist_ok=True)
    mask_index = {}

    names = list(seeds.keys())
    h = w = None
    frames = []
    for name in names:
        s = seeds[name]
        sid = s.get("session", args.session)  # seeds may span both sessions
        rgb_path = cap / "sessions" / sid / "raw_data" / name
        bgr = cv2.imread(str(rgb_path))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks, _ = _segment(model, processor, img, points=s["points"], labels=s["labels"],
                            box=s["box"], multimask=False)
        mask = masks[0].astype(bool)
        if args.save_masks:
            flat = name.replace("/", "_")
            cv2.imwrite(str(masks_dir / flat), (mask * 255).astype(np.uint8))
            mask_index[name] = {"session": sid, "mask_file": f"masks/{flat}",
                                "px": int(mask.sum()), "box": s["box"],
                                "n_visible": s.get("n_visible")}
        ov = bgr.copy()
        ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = [int(v) for v in s["box"]]
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 0), 1)
        for px, py in s["points"]:
            cv2.circle(ov, (int(px), int(py)), 4, (0, 255, 255), -1)
        cv2.putText(ov, f"{Path(name).stem} vis={s['n_visible']} m={int(mask.sum())}",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        h, w = ov.shape[:2]
        frames.append(ov)

    if frames:
        vw = cv2.VideoWriter(str(od / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
        # contact sheet of up to 12
        sel = frames[:: max(1, len(frames) // 12)][:12]
        tiles = [cv2.resize(f, (242, 242)) for f in sel]
        while len(tiles) % 4:
            tiles.append(np.zeros((242, 242, 3), np.uint8))
        rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(od / "result_contact.png"), np.concatenate(rows, 0))
    if args.save_masks:
        json.dump(mask_index, open(od / "masks_index.json", "w"), indent=1)
    print(f"per-frame: {len(frames)} frames"
          + (f", {len(mask_index)} masks saved" if args.save_masks else "")
          + f" -> {od}")


# ─────────────── stage: track (sam3_env, SAM3 *video* per-span refine) ───────────────
def _spans(seeded, split_gap):
    """Split index-sorted [(global_idx, name), ...] into contiguous visible spans
    (a new span starts when the frame-index gap exceeds split_gap = out of view)."""
    spans, cur = [], [seeded[0]]
    for gi, n in seeded[1:]:
        if gi - cur[-1][0] <= split_gap:
            cur.append((gi, n))
        else:
            spans.append(cur); cur = [(gi, n)]
    spans.append(cur)
    return spans


def cmd_track(args):
    """Refine the per-frame geom masks with the SAM3 video tracker.

    The object is visible in disjoint spans (it leaves and re-enters view), so a
    single video track across the whole walk would drift through the gaps. Instead
    each contiguous visible span is tracked as its own mini-video: seed the tracker
    with the geom mask at the span's strongest frame, propagate fwd+bwd within the
    span. Gives temporally-consistent masks without painting the out-of-view gaps.
    Frames a span couldn't cover keep their existing geom mask.
    """
    import os
    import shutil
    import tempfile
    from collections import defaultdict
    import torch

    cap = Path(args.capture)
    od = out_dir(cap, args.obj)
    seeds = json.load(open(od / "seeds.json"))
    mi_path = od / "masks_index.json"
    mi = json.load(open(mi_path)) if mi_path.exists() else {}
    masks_dir = od / "masks"
    masks_dir.mkdir(exist_ok=True)

    from sam3.model_builder import build_sam3_video_model
    print("loading SAM3 video model ...", flush=True)
    model = build_sam3_video_model()
    pred = model.tracker
    pred.backbone = model.detector.backbone

    by_session = defaultdict(list)
    for name, s in seeds.items():
        by_session[s.get("session", args.session)].append(name)

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _null()
    new_mi = dict(mi)  # video masks where we get them, geom masks elsewhere
    n_video = 0
    with torch.inference_mode(), ctx:
        for sid, names in by_session.items():
            fdir = cap / "sessions" / sid / "raw_data" / "images" / "cam0"
            allfiles = sorted(fdir.glob("*.jpg"), key=lambda p: int(p.stem))
            gidx = {f"images/cam0/{p.name}": i for i, p in enumerate(allfiles)}
            seeded = sorted((gidx[n], n) for n in names if n in gidx)
            spans = _spans(seeded, args.split_gap)
            print(f"[{sid}] {len(seeded)} visible frames -> {len(spans)} span(s)", flush=True)

            for si, span in enumerate(spans):
                # strongest frame in the span = best geom anchor
                best = max(span, key=lambda gn: seeds[gn[1]].get("n_visible", 0))[1]
                seed_path = masks_dir / ("images_cam0_" + Path(best).name)
                seed_mask = (cv2.imread(str(seed_path), 0) > 127) if seed_path.exists() else None
                if seed_mask is None or seed_mask.sum() < 50:
                    print(f"  span {si}: no usable seed mask at {Path(best).name}; keeping geom", flush=True)
                    continue

                tmp = Path(tempfile.mkdtemp(prefix="sam3span_"))
                try:
                    for _, n in span:
                        os.symlink(allfiles[gidx[n]], tmp / Path(n).name)
                    localsorted = sorted(tmp.glob("*.jpg"), key=lambda p: int(p.stem))
                    lidx = {f"images/cam0/{p.name}": j for j, p in enumerate(localsorted)}
                    lf = lidx[best]
                    st = pred.init_state(video_path=str(tmp))
                    pred.add_new_mask(st, frame_idx=lf, obj_id=1,
                                      mask=torch.from_numpy(seed_mask))
                    pred.propagate_in_video_preflight(st)
                    local_masks = {}
                    for reverse in (False, True):
                        for fi, _, _, vrm, _ in pred.propagate_in_video(
                                st, start_frame_idx=lf, max_frame_num_to_track=len(localsorted),
                                reverse=reverse, tqdm_disable=True):
                            local_masks[int(fi)] = np.asarray(vrm[0].squeeze().cpu()) > 0
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

                kept = 0
                for _, n in span:
                    mk = local_masks.get(lidx[n])
                    if mk is None or not mk.any():
                        continue
                    flat = n.replace("/", "_")
                    cv2.imwrite(str(masks_dir / flat), (mk * 255).astype(np.uint8))
                    new_mi[n] = {"session": sid, "mask_file": f"masks/{flat}",
                                 "px": int(mk.sum()), "n_visible": seeds[n].get("n_visible")}
                    kept += 1
                n_video += kept
                print(f"  span {si}: frames {span[0][0]}-{span[-1][0]} "
                      f"seed={Path(best).stem} -> {kept} video masks", flush=True)

    json.dump(new_mi, open(mi_path, "w"), indent=1)
    print(f"track: {n_video} frames refined by SAM3 video "
          f"({len(new_mi)} total masks) -> {mi_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("src-mask", "seeds", "perframe", "track"):
        p = sub.add_parser(name)
        p.add_argument("--capture", required=True)
        p.add_argument("--session", default="aria_a_rgb")
        p.add_argument("--obj", default=None,
                       help="object id; namespaces outputs under geom_sam_out/<obj>/")
        if name == "src-mask":
            p.add_argument("--src-name", required=True)
            p.add_argument("--points", type=float, nargs="+", required=True,
                           help="positive clicks: x1 y1 x2 y2 ...")
            p.add_argument("--neg", type=float, nargs="*", default=[],
                           help="negative clicks: x1 y1 x2 y2 ...")
        if name == "seeds":
            p.add_argument("--ref", default="navvis_a")
            p.add_argument("--src-name", required=True)
            p.add_argument("--n", type=int, default=0,
                           help="frames to seed: 0 = every frame (thorough, default); "
                                "N>0 evenly subsamples N frames (quick coarse pass)")
            p.add_argument("--min-vis", type=int, default=10,
                           help="min reprojected object points visible to keep a frame")
            p.add_argument("--occ-scale", type=float, default=0.35,
                           help="render occlusion depth at this fraction of full res "
                                "(speed; the cm-scale tolerance absorbs the error)")
            p.add_argument("--other-session", default="aria_b_rgb",
                           help="the other state's Aria session to cross-propagate into")
            p.add_argument("--other-ref", default="navvis_b",
                           help="NavVis reference for the other state")
            p.add_argument("--bridge", default=None,
                           help="path to T_<ref>_from_<other_ref>.txt; default derived from capture")
            p.add_argument("--no-cross", action="store_true",
                           help="only propagate within the source session")
        if name == "perframe":
            p.add_argument("--save-masks", action="store_true",
                           help="persist per-frame binary masks + masks_index.json")
        if name == "track":
            p.add_argument("--split-gap", type=int, default=4,
                           help="frame-index gap above which the object is treated as "
                                "having left view -> a new span (separate video track)")
    args = ap.parse_args()
    {"src-mask": cmd_src_mask, "seeds": cmd_seeds, "perframe": cmd_perframe,
     "track": cmd_track}[args.cmd](args)


if __name__ == "__main__":
    main()
