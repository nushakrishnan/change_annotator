"""Geometry-assisted annotation prototype (1 object, end-to-end).

Three-stage pipeline, all in ONE env (file handoff between stages):

  src-mask  : click on source frame -> SAM3 image mask  -> src_mask.png
  seeds     : lift mask onto mesh (raycast), reproject the 3D object
              into every frame with a mesh-depth occlusion test
              -> per-frame seed point(s)+box -> seeds.json   (needs scantools)
  perframe  : SAM3 image predict(points+box) per frame  -> per-frame masks

Imports are lazy per stage (sam3 in src-mask/perframe; scantools in seeds) so a
stage only pays for what it uses. The `seeds` stage needs scantools on
PYTHONPATH (lamaria-indoor). Run with the single env from setup.sh:
  PY=~/annotator_env/bin/python; PP=~/repos/lamaria-indoor
  $PY geom_sam_prototype.py src-mask  --capture C --session aria_a_rgb \
        --src-name images/cam0/16700526188301.jpg --points 300 180
  PYTHONPATH=$PP $PY geom_sam_prototype.py seeds \
        --capture C --session aria_a_rgb --ref navvis_a --src-name images/cam0/16700526188301.jpg --n 40
  $PY geom_sam_prototype.py perframe --capture C --session aria_a_rgb
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import cv2

# Output workspace root (under the capture). Override with GEOM_OUT to run an
# isolated workspace (e.g. GEOM_OUT=changes/geom_sam_out_test) without touching
# existing masks. Inherited by the seeds subprocess via the environment.
OUT = os.environ.get("GEOM_OUT", "changes/geom_sam_out")


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


def _segment_concept(model, processor, img_rgb, phrase):
    """SAM3 CONCEPT (text) segmentation: return every instance matching `phrase`
    as (masks bool[N,H,W], boxes xyxy-pixel[N,4], scores[N]). Unlike the point+box
    `_segment`, this segments by learned object identity, so it pulls in thin
    structure (chair legs/base, table legs) a single interior click drops. Empty
    arrays if nothing matches. Caller disambiguates instances by box overlap."""
    import torch
    from PIL import Image as PILImage
    H, W = img_rgb.shape[:2]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _null()
    with torch.inference_mode(), ctx:
        state = processor.set_image(PILImage.fromarray(img_rgb))
        state = processor.set_text_prompt(prompt=phrase, state=state)
    boxes = state.get("boxes")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, H, W), bool), np.zeros((0, 4)), np.zeros((0,))
    # cast to float32 first: under autocast these are bfloat16, which numpy can't convert
    boxes = boxes.detach().float().cpu().numpy()
    m = state["masks"].detach().float().cpu().numpy()
    if m.ndim == 4:                      # (N,1,H,W) -> (N,H,W)
        m = m[:, 0]
    masks = m > 0
    scores = (state["scores"].detach().float().cpu().numpy()
              if state.get("scores") is not None else np.ones(len(boxes)))
    return masks, boxes, scores


def _segment_refine(model, processor, img_rgb, points, labels, box, prior_mask, mask_res=288):
    """Interactive refine (the GUI 'smart brush'): seed SAM from the CURRENT mask
    (`prior_mask` as a low-res logit mask_input) plus pos/neg scribble points, so a
    rough dab pulls a missed part in / a rough stroke removes bleed while SAM snaps
    to the object edge. `mask_res` = 4x the model's image-embed grid (72 -> 288 at
    the 1008 processing resolution); change it if the backbone/resolution changes."""
    import torch
    from PIL import Image as PILImage
    mlow = cv2.resize(prior_mask.astype(np.uint8), (mask_res, mask_res),
                      interpolation=cv2.INTER_NEAREST)
    mask_input = np.where(mlow > 0, 12.0, -12.0).astype(np.float32)[None]
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else _null()
    with torch.inference_mode(), ctx:
        state = processor.set_image(PILImage.fromarray(img_rgb))
        masks, _scores, _ = model.predict_inst(
            state, point_coords=np.asarray(points, np.float32),
            point_labels=np.asarray(labels, np.int32), box=np.asarray(box, np.float32),
            mask_input=mask_input, multimask_output=False)
    return np.asarray(masks)[0].astype(bool)


def index_keep_protected(old_index, seed_names, frontier_ts):
    """Split an existing masks_index before a per-frame regeneration: returns the
    entries that MUST survive. Kept: src='hand' frames and frames at/behind the
    verified frontier (co-ground-truth, never regenerated), plus non-'geom'
    entries (prop/concept/legacy) for frames no longer seeded — dropping those
    would delete propagation/remask/hand-era work. Stale 'geom' entries for
    frames that dropped out of the seeds ARE dropped (the geometry now says the
    object isn't there; exporting them would corrupt GT)."""
    keep = {}
    for name, e in old_index.items():
        prot = e.get("src") == "hand" or int(Path(name).stem) <= frontier_ts
        if prot or (name not in seed_names and e.get("src") != "geom"):
            keep[name] = e
    return keep


def frontier_ts(verified_until):
    """Frontier frame name -> comparable timestamp (-1 = no frontier)."""
    return int(Path(verified_until).stem) if verified_until else -1


def _box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def pick_concept_instance(masks, boxes, scores, geom_box,
                          iou_thr=0.3, score_thr=0.5, contain_thr=0.7, pad_frac=0.3):
    """From concept-path detections, return the mask matching `geom_box` (=[x0,y0,x1,y1]
    from the object's geometry seed), or None to fall back. Three gates, all general:
      - box IoU with geom_box >= iou_thr           (right instance)
      - detection score >= score_thr               (confident)
      - >= contain_thr of the mask lies within geom_box dilated by pad_frac
        (rejects motion-blur floor-spills and whole-frame blobs, while allowing
         legs to poke a bit past the lidar-derived box)."""
    if len(boxes) == 0:
        return None
    ious = [_box_iou(boxes[k], geom_box) for k in range(len(boxes))]
    b = int(np.argmax(ious))
    if ious[b] < iou_thr or float(scores[b]) < score_thr or not masks[b].any():
        return None
    x0, y0, x1, y1 = geom_box
    pad = pad_frac * max(x1 - x0, y1 - y0)
    ys, xs = np.where(masks[b])
    inside = ((xs >= x0 - pad) & (xs <= x1 + pad) &
              (ys >= y0 - pad) & (ys <= y1 + pad)).mean()
    return masks[b] if inside >= contain_thr else None


# ───────────────────────────── stage: src-mask ──────────────────────────────
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


# ─────────────────── stage: seeds (needs scantools on PYTHONPATH) ───────────────────
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
                       skip_name=None, occ_tol=0.10, occ_scale=0.35, min_frac=0.0,
                       write_dbg=True):
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
    if write_dbg:
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
        # strict occlusion: keep the frame only if a MIN COUNT and a MIN FRACTION of
        # the object's in-frustum points are unoccluded. The fraction rejects frames
        # where the object is mostly hidden (e.g. behind a wall) but a few points leak
        # past the mesh -- the "change behind a wall" case that must not be logged.
        if len(pv) < min_vis or (min_frac > 0 and len(pv) < min_frac * int(vis.sum())):
            continue
        cen = pv.mean(0)
        x0, y0 = pv.min(0)
        x1, y1 = pv.max(0)
        seeds[name] = {"session": sid,
                       "points": [[float(cen[0]), float(cen[1])]], "labels": [1],
                       "box": [float(x0), float(y0), float(x1), float(y1)],
                       "n_visible": int(len(pv))}
        if write_dbg:
            im = cv2.imread(str(capo.data_path(sid) / name))
            if im is not None:
                cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
                cv2.circle(im, (int(cen[0]), int(cen[1])), 5, (0, 0, 255), -1)
                cv2.imwrite(str(dbg / Path(name).name), im)
    return seeds


def _lift_mask(mask, src_name, sess, renderer, compute_rays):
    """Raycast a 2D `mask` drawn on frame `src_name` onto the mesh held by
    `renderer` -> (M,3) surface points, with a per-mask outlier trim that drops
    mask-edge background bleed. Empty array if nothing hits."""
    src_key = next(k for k in sess.images.key_pairs()
                   if str(sess.images[k[0], k[1]]) == src_name)
    ts_s, cam_s = src_key
    T_src = sess.get_pose(ts_s, cam_s)
    rows, cols = np.where(mask)
    if len(cols) == 0:
        return np.empty((0, 3))
    if len(cols) > 4000:
        sel = np.random.RandomState(0).choice(len(cols), 4000, replace=False)
        rows, cols = rows[sel], cols[sel]
    p2d = np.stack([cols, rows], 1).astype(np.float32)
    o, d = compute_rays(T_src, sess.sensors[cam_s], p2d=p2d)
    pts, _ = renderer.compute_intersections((o, d))
    pts = np.asarray(pts)
    if len(pts) == 0:
        return pts
    c = np.median(pts, 0)
    dist = np.linalg.norm(pts - c, axis=1)
    return pts[dist < np.percentile(dist, 90)]


def _load_src_masks(od, fallback_src_name):
    """Seed masks for the object as a list of (src_name, bool_mask). Prefers the
    multi-seed index (src_index.json: spread-out frames unioned for a complete 3D
    object); falls back to the single src_mask.png (one frame)."""
    idx = od / "src_index.json"
    if idx.exists():
        out = []
        for it in json.load(open(idx)):
            m = cv2.imread(str(od / it["mask_file"]), 0)
            if m is not None:
                out.append((it["src_name"], m > 127))
        return out
    m = cv2.imread(str(od / "src_mask.png"), 0)
    return [(fallback_src_name, m > 127)] if m is not None else []


def cmd_seeds(args):
    from scantools.proc.rendering import Renderer, compute_rays
    from scantools.utils.io import read_mesh

    cap = Path(args.capture)
    od = out_dir(cap, args.obj)
    capo, sess = _session(cap, args.session, args.ref)
    mesh_path = capo.proc_path(args.ref) / capo.sessions[args.ref].proc.meshes["mesh"]
    renderer = Renderer(read_mesh(mesh_path))

    # lift each seed mask (one or several spread-out frames) onto the SOURCE-state
    # mesh and UNION the 3D points -> a far more complete object than a single view,
    # so every downstream frame gets a full, accurate seed.
    src_masks = _load_src_masks(od, args.src_name)
    if not src_masks:
        raise SystemExit(f"no source masks under {od} (src_index.json or src_mask.png)")
    chunks = [c for c in (_lift_mask(m, sn, sess, renderer, compute_rays)
                          for sn, m in src_masks) if len(c)]
    if not chunks:
        raise SystemExit("no seed points hit the mesh")
    pts3d = np.concatenate(chunks, 0)
    print(f"object 3D points: {len(pts3d)} from {len(chunks)}/{len(src_masks)} seed "
          f"frame(s) in {args.ref} frame (center {np.median(pts3d, 0).round(2)})")

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


# ───────────────────────────── stage: perframe ──────────────────────────────
def cmd_perframe(args):
    cap = Path(args.capture)
    od = out_dir(cap, args.obj)
    seeds = json.load(open(od / "seeds.json"))
    model, processor = _load_sam_image()

    masks_dir = od / "masks"
    if args.save_masks:
        masks_dir.mkdir(exist_ok=True)
    # co-GT protection (same rules as the GUI): keep hand frames + frames at/behind
    # the object's verified frontier + non-geom work on unseeded frames; never
    # rebuild over them. Frontier comes from the workspace's gui_objects.json.
    mi_path = od / "masks_index.json"
    old_index = json.load(open(mi_path)) if mi_path.exists() else {}
    gobj_path = out_dir(cap) / "gui_objects.json"
    gobj = json.load(open(gobj_path)) if gobj_path.exists() else {}
    f_ts = frontier_ts(gobj.get(args.obj, {}).get("verified_until"))
    mask_index = index_keep_protected(old_index, set(seeds), f_ts)
    n_prot = len(mask_index)
    if n_prot:
        print(f"protected (hand/frontier/non-geom): {n_prot} frames kept as-is")

    names = list(seeds.keys())
    h = w = None
    frames = []
    for name in names:
        if name in mask_index:                    # protected — do not regenerate
            continue
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
            mask_index[name] = {"session": sid, "mask_file": f"masks/{flat}", "src": "geom",
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


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("src-mask", "seeds", "perframe"):
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
    args = ap.parse_args()
    {"src-mask": cmd_src_mask, "seeds": cmd_seeds,
     "perframe": cmd_perframe}[args.cmd](args)


if __name__ == "__main__":
    main()
