"""Per-frame change masks for evaluation, with an MV3DCD-suited symmetric variant.

Two products, both derived from the human-verified per-sequence object masks
(geom_sam_prototype + gui.py, under `<geom_out>/<id>__<state>/masks_index.json`):

1. NATIVE change mask (default) -- per frame, the UNION over objects of the
   verified masks (annotation_spec.md §6/§7). Directional: a post frame carries
   only post-native changes (objects at their post location). Pure cv2+numpy.

2. SYMMETRIC / MV3DCD change mask (`--symmetric`) -- a post step that ALSO marks,
   in each post frame, the footprint of the *pre* changes (and vice versa), so the
   GT matches what a multi-view 3D detector like MV3DCD emits: every changed region,
   both directions, in every image (see the long discussion in annotation_spec.md).

   The cross-projection is mesh-anchored and multi-view consistent, NOT a per-frame
   2D warp:
     - vote each object's verified masks onto the source-state NavVis mesh FACES
       (a face is "footprint" if it projects inside the mask in >= `ratio` of the
       views that can see it) -> one consistent 3D footprint on the scanned surface;
     - carry those faces across the rigid NavVis->NavVis bridge (exact);
     - render the labeled sub-mesh into every target frame, occlusion-tested against
       the target-state mesh -> a solid, connected silhouette (no splat/close, no SAM).
   This second product needs the lamar stack (scantools/raybender/open3d/open3d) and
   geom_sam_prototype; those imports are local so product 1 stays dependency-light.

Standalone:
  # native only:
  python change_mask.py --capture /path/to/captures/changes/cnb_e100
  # + symmetric MV3DCD GT (lamar_env):
  PYTHONPATH=~/repos/lamaria-indoor ~/lamar_env/bin/python change_mask.py \
      --capture /path/to/captures/changes/cnb_e100 --symmetric
"""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

GEOM_OUT = "changes/geom_sam_out"   # geom-out root relative to the capture


def _flat_png(frame):
    """'images/cam0/1668..jpg' -> 'images_cam0_1668...png'."""
    return Path(frame.replace("/", "_")).with_suffix(".png").name


# ───────────────────────────── viz panels ─────────────────────────────
def _rgb_path(capture, session, frame):
    """Source RGB frame on disk: sessions/<session>/raw_data/<frame>."""
    return Path(capture) / "sessions" / session / "raw_data" / frame


def _viz_panel(rgb, mask, native=None, ghost=None, alpha=0.5):
    """A side-by-side [ RGB | overlay | mask ] BGR panel for eyeballing a mask
    against its source image. When `native`/`ghost` are given the overlay colors
    native GREEN and cross-projected ghost RED (green wins on overlap); otherwise
    the whole mask is drawn green. Missing/other-shaped inputs are handled."""
    H, W = mask.shape
    if rgb is None:
        rgb = np.zeros((H, W, 3), np.uint8)
    elif rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))

    def _fit(b):
        if b is None:
            return None
        if b.shape != (H, W):
            b = cv2.resize(b.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        return b

    native, ghost = _fit(native), _fit(ghost)
    ov = rgb.copy()
    if native is None and ghost is None:
        ov[mask] = (0, 255, 0)
    else:
        if ghost is not None:
            ov[ghost] = (0, 0, 255)
        if native is not None:
            ov[native] = (0, 255, 0)
    ov = cv2.addWeighted(rgb, 1 - alpha, ov, alpha, 0)
    mask_bgr = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    sep = np.full((H, 6, 3), 255, np.uint8)
    return np.hstack([rgb, sep, ov, sep, mask_bgr])


def _write_viz(capture, subdir, unions, prov, split=None, verbose=True):
    """Write [ RGB | overlay | mask ] panels to changes/<subdir>/<state>_viz/*.jpg.
    `split` (optional) {state: {frame: (native_bool, ghost_bool)}} colors native
    vs. cross-projected in the overlay; without it the mask is drawn as one color."""
    capture = Path(capture)
    for state, frames in unions.items():
        vdir = capture / "changes" / subdir / f"{state}_viz"
        vdir.mkdir(parents=True, exist_ok=True)
        n_missing = 0
        for frame, mask in frames.items():
            session = prov.get(state, {}).get(frame, {}).get("session")
            rgb = cv2.imread(str(_rgb_path(capture, session, frame))) if session else None
            if rgb is None:
                n_missing += 1
            nat, gho = (split.get(state, {}).get(frame, (None, None)) if split
                        else (None, None))
            panel = _viz_panel(rgb, mask, nat, gho)
            fname = Path(_flat_png(frame)).with_suffix(".jpg").name
            cv2.imwrite(str(vdir / fname), panel)
        if verbose:
            miss = f" ({n_missing} without RGB)" if n_missing else ""
            print(f"  {state}_viz: {len(frames)} panels{miss} -> {vdir}")


# ───────────────────────── native union (cv2-only) ─────────────────────────
def _native_unions(capture, geom_out, object_keys, verbose):
    """{state: {frame: bool union}} and {state: {frame: {session}}} from the
    verified per-object masks_index.json files. State comes from the '__<state>'
    dir suffix (each object dir is single-state)."""
    capture = Path(capture)
    root = capture / geom_out
    if object_keys is None:
        object_keys = sorted(p.name for p in root.iterdir()
                             if p.is_dir() and "__" in p.name)
    unions, prov = {}, {}
    for key in object_keys:
        if "__" not in key:
            continue
        _, state = key.rsplit("__", 1)
        mi_path = root / key / "masks_index.json"
        if not mi_path.exists():
            if verbose:
                print(f"  skip {key}: no masks_index.json")
            continue
        for frame, meta in json.load(open(mi_path)).items():
            m = cv2.imread(str(root / key / meta["mask_file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                if verbose:
                    print(f"  warn {key}/{meta['mask_file']}: unreadable, skipped")
                continue
            mb = m > 127
            acc = unions.setdefault(state, {}).get(frame)
            if acc is None:
                unions[state][frame] = mb
                prov.setdefault(state, {})[frame] = {"session": meta.get("session")}
            elif mb.shape == acc.shape:
                unions[state][frame] = acc | mb
            else:                                   # frames share a camera; shouldn't happen
                mb = cv2.resize(mb.astype(np.uint8), (acc.shape[1], acc.shape[0]),
                                interpolation=cv2.INTER_NEAREST) > 0
                unions[state][frame] = acc | mb
    return unions, prov


def _write_masks(capture, subdir, unions, prov):
    """Write {state:{frame:bool}} to <capture>/changes/<subdir>/<state>/*.png and an
    index; return the spec §6 block {state: {frame: '<relpath under changes/>'}}."""
    capture = Path(capture)
    out_root = capture / "changes" / subdir
    block, index = {}, {}
    for state, frames in unions.items():
        (out_root / state).mkdir(parents=True, exist_ok=True)
        for frame, mask in frames.items():
            fname = _flat_png(frame)
            cv2.imwrite(str(out_root / state / fname), (mask.astype(np.uint8) * 255))
            rel = f"{subdir}/{state}/{fname}"
            block.setdefault(state, {})[frame] = rel
            index.setdefault(state, {})[frame] = {
                "mask_file": rel, "session": prov.get(state, {}).get(frame, {}).get("session"),
                "px": int(mask.sum())}
    json.dump(index, open(capture / "changes" / f"{subdir}_index.json", "w"), indent=1)
    return block


def build_change_masks(capture, geom_out=GEOM_OUT, object_keys=None, verbose=True, viz=False):
    """NATIVE per-frame change mask = union of object masks. Returns the spec §6
    block {state: {frame: relpath}}; writes change_mask/<state>/*.png."""
    unions, prov = _native_unions(capture, geom_out, object_keys, verbose)
    block = _write_masks(capture, "change_mask", unions, prov)
    if viz:
        _write_viz(capture, "change_mask", unions, prov, verbose=verbose)
    if verbose:
        for state, frames in block.items():
            print(f"  {state}: {len(frames)} change frames")
    return block


# ─────────────────── mesh-anchored cross-projection (lamar) ───────────────────
def _triid_map(renderer, T_cam2w, camera):
    """Per-pixel triangle-id map (-1 where no hit) for `renderer`'s mesh at this
    pose, plus a validity mask. tri_ids index directly into the mesh triangles."""
    import raybender as rb
    import raybender.utils as rbutils
    from scantools.proc.rendering import compute_rays
    o, d = compute_rays(T_cam2w, camera)                 # full image, row-major
    geom_ids, bcoords = rb.ray_scene_intersection(renderer.scene, o, d)
    *_, tri_ids, bcoords, valid = rbutils.filter_intersections(geom_ids, bcoords)
    triid = np.full(valid.shape[0], -1, np.int64)
    triid[valid] = tri_ids
    H, W = camera.height, camera.width
    return triid.reshape(H, W), valid.reshape(H, W)


def _vote_faces(sess, renderer, n_tris, obj_frames, scale, min_views, ratio, verbose, tag):
    """Vote the verified masks onto mesh faces. A face is kept if it is visible in
    >= min_views source frames AND projects inside the mask in >= `ratio` of the
    frames that see it. Returns the kept face-index array. Multi-view consistency is
    enforced here: faces survive by agreement across views, not from any one frame."""
    import geom_sam_prototype as G
    seen = np.zeros(n_tris, np.int32)
    inside = np.zeros(n_tris, np.int32)
    for j, (frame, mask) in enumerate(obj_frames.items()):
        key = next(k for k in sess.images.key_pairs() if str(sess.images[k[0], k[1]]) == frame)
        ts, cam = key
        cam_v, _, _ = G._scaled_camera(sess.sensors[cam], scale)
        triid, valid = _triid_map(renderer, sess.get_pose(ts, cam), cam_v)
        mk = cv2.resize(mask.astype(np.uint8), (cam_v.width, cam_v.height),
                        interpolation=cv2.INTER_NEAREST) > 0
        seen[np.unique(triid[valid])] += 1
        inside[np.unique(triid[valid & mk])] += 1
        if verbose and j and j % 50 == 0:
            print(f"    [{tag}] voted {j}/{len(obj_frames)} frames", flush=True)
    keep = (seen >= min_views) & (inside >= ratio * np.maximum(seen, 1))
    L = np.where(keep)[0]
    if verbose:
        print(f"    [{tag}] kept {len(L)}/{n_tris} faces from {len(obj_frames)} views")
    return L, seen, inside


def _save_face_colored_ply(path, V, tris, vals, lo, hi):
    """Write the (V, tris) sub-mesh as a .ply, per-face colored by `vals`
    (red = lo, green = hi). Vertices are duplicated per face so each triangle
    gets a crisp flat color (cheap for the few-k footprint faces)."""
    import open3d as o3d
    verts = V[tris].reshape(-1, 3)
    faces = np.arange(len(verts)).reshape(-1, 3)
    r = np.clip((np.asarray(vals, np.float64) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    face_col = np.stack([1.0 - r, r, np.zeros_like(r)], 1)      # low=red -> high=green
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                  o3d.utility.Vector3iVector(faces))
    m.vertex_colors = o3d.utility.Vector3dVector(np.repeat(face_col, 3, axis=0))
    o3d.io.write_triangle_mesh(str(path), m)


def _ghost_for_direction(capture, src, dst, bridge, geom_out, object_keys, *,
                         vote_scale, render_scale, min_views, ratio, occ_tol, verbose,
                         dbg_dir=None):
    """Footprint of src-state changes rendered into dst-state frames.
    `bridge` is the Pose3d mapping src-ref coords -> dst-ref coords.
    Returns {dst_frame_name: bool mask in dst image space}."""
    import geom_sam_prototype as G
    import open3d as o3d
    from scantools.proc.rendering import Renderer
    from scantools.utils.io import read_mesh

    capture = Path(capture)
    root = capture / geom_out
    tag = f"{src['state']}->{dst['state']}"

    # union the verified masks of every src-state object, per source frame
    obj_frames = {}
    for key in object_keys or []:
        if not key.endswith("__" + src["state"]):
            continue
        mi = root / key / "masks_index.json"
        if not mi.exists():
            continue
        for f, meta in json.load(open(mi)).items():
            m = cv2.imread(str(root / key / meta["mask_file"]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            mb = m > 127
            obj_frames[f] = mb if f not in obj_frames else (obj_frames[f] | mb)
    if not obj_frames:
        if verbose:
            print(f"    [{tag}] no source masks")
        return {}

    # vote faces on the source-state mesh
    capo_s, sess_s = G._session(capture, src["session"], src["ref"])
    mesh_s = read_mesh(capo_s.proc_path(src["ref"])
                       / capo_s.sessions[src["ref"]].proc.meshes["mesh"])
    V_s, T_s = np.asarray(mesh_s.vertices), np.asarray(mesh_s.triangles)
    R_s = Renderer(mesh_s)
    L, seen, inside = _vote_faces(sess_s, R_s, len(T_s), obj_frames,
                                  vote_scale, min_views, ratio, verbose, tag)
    R_s = None        # let the weakref finalizer free the scene once (do NOT call release())
    gc.collect()
    if len(L) == 0:
        return {}

    # build the labeled sub-mesh, carry it across the rigid bridge into dst frame
    tris_L = T_s[L]
    used = np.unique(tris_L)
    remap = np.full(len(V_s), -1, np.int64)
    remap[used] = np.arange(len(used))
    local_tris = remap[tris_L]
    V_dst = G._apply_T(bridge, V_s[used])

    # debug: dump the raycast footprint faces colored by vote confidence
    # (inside/seen: red≈`ratio` marginal, green≈1 solid), in source-mesh coords
    # (overlay on the src NavVis mesh) and in bridged/dst coords (as rendered).
    if dbg_dir is not None:
        conf = inside[L] / np.maximum(seen[L], 1)
        dbg = Path(dbg_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        base = f"footprint_{src['state']}_{src['ref']}"
        _save_face_colored_ply(dbg / f"{base}_src.ply", V_s[used], local_tris, conf, ratio, 1.0)
        _save_face_colored_ply(dbg / f"{base}_to_{dst['ref']}.ply", V_dst, local_tris,
                               conf, ratio, 1.0)
        if verbose:
            print(f"    [{tag}] saved footprint mesh ({len(L)} faces) -> {base}_*.ply")

    sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V_dst),
                                    o3d.utility.Vector3iVector(local_tris))
    R_L = Renderer(sub)

    # render the footprint into every dst frame, occlusion-tested vs the dst mesh
    capo_d, sess_d = G._session(capture, dst["session"], dst["ref"])
    R_B = Renderer(read_mesh(capo_d.proc_path(dst["ref"])
                             / capo_d.sessions[dst["ref"]].proc.meshes["mesh"]))
    keys = [k for k in sess_d.images.key_pairs() if "cam0" in str(k[1])]
    keys.sort(key=lambda k: k[0])
    out = {}
    for j, (ts, cam) in enumerate(keys):
        name = str(sess_d.images[ts, cam])
        cam_r, _, _ = G._scaled_camera(sess_d.sensors[cam], render_scale)
        T = sess_d.get_pose(ts, cam)
        _, depth_L = R_L.render_from_capture(T, cam_r)
        if not np.any(depth_L > 0):
            continue
        _, depth_B = R_B.render_from_capture(T, cam_r)
        occ = depth_B > 0
        vis = (depth_L > 0) & (~occ | (depth_L <= depth_B + occ_tol))   # not hidden by dst geom
        if not vis.any():
            continue
        cam_o = sess_d.sensors[cam]
        if vis.shape != (cam_o.height, cam_o.width):
            vis = cv2.resize(vis.astype(np.uint8), (cam_o.width, cam_o.height),
                             interpolation=cv2.INTER_NEAREST) > 0
        out[name] = vis
        if verbose and j and j % 50 == 0:
            print(f"    [{tag}] rendered {j}/{len(keys)} frames ({len(out)} hit)", flush=True)
    R_L = R_B = None        # drop refs; weakref finalizer frees each scene once
    gc.collect()
    if verbose:
        print(f"    [{tag}] footprint in {len(out)} dst frames")
    return out


def build_symmetric(capture, states, bridge_path, geom_out=GEOM_OUT, object_keys=None,
                    vote_scale=0.5, render_scale=1.0, min_views=3, ratio=0.6,
                    occ_tol=0.10, verbose=True, viz=False):
    """SYMMETRIC / MV3DCD GT: native union, plus each state's footprint cross-rendered
    into the other state's frames. Writes change_mask_symmetric/<state>/*.png and
    returns the spec §6 block."""
    import geom_sam_prototype as G
    capture = Path(capture)
    root = capture / geom_out
    if object_keys is None:
        object_keys = sorted(p.name for p in root.iterdir()
                             if p.is_dir() and "__" in p.name)

    unions, prov = _native_unions(capture, geom_out, object_keys, verbose)
    native_snap = {st: dict(fr) for st, fr in unions.items()}  # native before ghost merge

    T = G._load_T(bridge_path)                          # T_<preref>_from_<postref>
    pre, post = states["pre"], states["post"]
    params = dict(vote_scale=vote_scale, render_scale=render_scale, min_views=min_views,
                  ratio=ratio, occ_tol=occ_tol, verbose=verbose)
    if viz:
        params["dbg_dir"] = capture / "changes" / "change_mask_symmetric"
    # pre footprint -> post frames (carry pre-ref pts into post-ref: inverse of the bridge)
    ghosts = {"post": _ghost_for_direction(capture, pre, post, T.inverse(), geom_out,
                                           object_keys, **params),
              "pre": _ghost_for_direction(capture, post, pre, T, geom_out,
                                          object_keys, **params)}
    for state, gmap in ghosts.items():
        sess = states[state]["session"]
        for frame, g in gmap.items():
            acc = unions.setdefault(state, {}).get(frame)
            if acc is None:
                unions[state][frame] = g
            elif g.shape == acc.shape:
                unions[state][frame] = acc | g
            else:
                g = cv2.resize(g.astype(np.uint8), (acc.shape[1], acc.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
                unions[state][frame] = acc | g
            prov.setdefault(state, {}).setdefault(frame, {"session": sess})

    block = _write_masks(capture, "change_mask_symmetric", unions, prov)
    if viz:
        split = {}
        for state, frames in unions.items():
            for frame in frames:
                split.setdefault(state, {})[frame] = (
                    native_snap.get(state, {}).get(frame),
                    ghosts.get(state, {}).get(frame))
        _write_viz(capture, "change_mask_symmetric", unions, prov, split, verbose)
    if verbose:
        for state, frames in block.items():
            print(f"  {state}: {len(frames)} symmetric change frames")
    return block


# ───────────────────────────────── CLI ─────────────────────────────────
def _states_from_args(a):
    return {"pre": {"state": "pre", "session": a.pre_session, "ref": a.pre_ref},
            "post": {"state": "post", "session": a.post_session, "ref": a.post_ref}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture dir holding changes/")
    ap.add_argument("--geom-out", default=GEOM_OUT,
                    help=f"geom-out root relative to capture (default {GEOM_OUT})")
    ap.add_argument("--symmetric", action="store_true",
                    help="also build the MV3DCD-suited symmetric GT (needs lamar_env)")
    ap.add_argument("--viz", action="store_true",
                    help="also write [RGB | overlay | mask] panels under "
                         "<mask_dir>/<state>_viz/ (native=green, cross-projected=red)")
    ap.add_argument("--pre-session", default="aria_a_rgb")
    ap.add_argument("--pre-ref", default="navvis_a")
    ap.add_argument("--post-session", default="aria_b_rgb")
    ap.add_argument("--post-ref", default="navvis_b")
    ap.add_argument("--bridge", default=None,
                    help="T_<preref>_from_<postref>.txt; default derived from refs")
    ap.add_argument("--vote-scale", type=float, default=0.5,
                    help="render scale for face voting (speed; faces are coarse)")
    ap.add_argument("--render-scale", type=float, default=1.0,
                    help="render scale for the output footprint masks")
    ap.add_argument("--min-views", type=int, default=3,
                    help="min source views that must see a face to keep it")
    ap.add_argument("--ratio", type=float, default=0.6,
                    help="min inside/seen ratio to keep a face (multi-view vote)")
    ap.add_argument("--occ-tol", type=float, default=0.10,
                    help="depth tolerance (m) for the target-mesh occlusion test")
    args = ap.parse_args()

    cap = Path(args.capture)
    # when building symmetric, its viz supersedes the native one (native ⊆ symmetric)
    native = build_change_masks(args.capture, geom_out=args.geom_out,
                                viz=args.viz and not args.symmetric)
    blocks = {"change_mask": native}

    if args.symmetric:
        bridge = args.bridge or str(cap / "changes" / f"{args.post_ref}_to_{args.pre_ref}"
                                    / f"T_{args.pre_ref}_from_{args.post_ref}.txt")
        sym = build_symmetric(args.capture, _states_from_args(args), bridge,
                              geom_out=args.geom_out, vote_scale=args.vote_scale,
                              render_scale=args.render_scale, min_views=args.min_views,
                              ratio=args.ratio, occ_tol=args.occ_tol, viz=args.viz)
        blocks["change_mask_symmetric"] = sym

    seg_path = cap / "changes" / "segments.json"
    if seg_path.exists():
        seg = json.load(open(seg_path))
        seg.update(blocks)
        json.dump(seg, open(seg_path, "w"), indent=1)
        print(f"updated {seg_path} with {', '.join(blocks)}")
    for name, b in blocks.items():
        print(f"{name}: {sum(len(v) for v in b.values())} frames -> {cap/'changes'/name}")


if __name__ == "__main__":
    main()
