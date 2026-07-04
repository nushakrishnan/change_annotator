"""Symmetric change GT from per-object meshes built out of the RAW lidar cloud.

change_mask.py --symmetric can only select faces that survived NavVis meshing,
and the meshing step drops most changed objects -> holey/missing ghosts.
lidar_tsdf_prototype.py repaired that with DA3 mono depth fused into a TSDF.
This prototype drops the mono-depth dependency entirely: the raw lidar scan
still contains the changed objects, so the verified 2D masks can select the
object's own scan points directly:

  - per object (<geom_out>/<id>__<state>), project the state's raw cloud into
    every ANNOTATED frame, z-buffered so points hidden behind nearer scan
    surface cannot vote, and keep points that fall inside the verified mask in
    enough views (the point analog of change_mask._vote_faces);
  - reconstruct a small per-object mesh from the kept points (alpha shape by
    default, Poisson optional);
  - optionally trim reconstruction overfill with the mask EXTERIOR: run
    change_mask._vote_faces on the object mesh and drop faces that project
    outside the masks. Only annotated frames may trim -- in an unannotated
    frame the absence of a mask proves nothing;
  - render the bridged object meshes into the other state's frames with the
    same occlusion-tested loop as change_mask._ghost_for_direction. Output
    goes to changes/change_mask_symmetric_points/ (originals untouched).

Stages (lamar_env + PYTHONPATH=~/repos/lamaria-indoor:., no GPU, no DA3 cache):

  # 1. select + mesh (+ trim) each object's lidar points:
  python point_ghost_prototype.py objects --capture .../cnb_e100
  #    -> changes/point_ghost/<key>/{points.ply,mesh.ply,stats.json}

  # 2. symmetric GT with the object meshes as the ghost source:
  python point_ghost_prototype.py symmetric --capture .../cnb_e100 --viz
  #    -> changes/change_mask_symmetric_points/<state>/*.png (+ _viz)

  # 3. old-vs-new comparison panels + source-frame IoU table:
  python point_ghost_prototype.py compare --capture .../cnb_e100
  #    -> changes/point_ghost_compare/<state>/*.jpg + iou.json
"""
import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

import change_mask as CM
import geom_sam_prototype as G
import mono_footprint_prototype as M

VOTE_SCALE = 0.5           # 2560x1920 -> 1280x960 for point voting; below this,
                           # thin structure (chair tube frames) is 2-4 px wide and
                           # the mask erosion wipes it out before it can vote
CLOUD_VOXEL = 0.01         # lidar downsample (m); plenty for object-scale meshing
SPLAT = 1                  # z-buffer min-filter radius (px), same idea as
                           # lidar_tsdf_prototype.LidarAnchoredDepth.lidar_depth
POINT_OCC_TOL = 0.08       # a point votes only if within this of the z-buffer front
ALPHA = 0.06               # alpha-shape radius (m); ~6x the 1 cm point spacing
MASK_ERODE = 1             # mask erosion (vote-res px) before voting: the verified
                           # masks are consistently fat at the object-floor contact,
                           # which otherwise votes in a sheet of floor points
MIN_MASK_PX = 1200         # skip frames whose eroded mask is smaller (vote-res px):
                           # distant views alias fore/background into the tiny mask
CLUSTER_EPS = 0.08         # DBSCAN radius for the post-vote cluster filter
CLUSTER_MERGE = 0.30       # keep clusters this close (m) to the dominant one


def _out_root(capture):
    return Path(capture) / "changes" / "point_ghost"


def _obj_keys(capture, geom_out, state=None, ghosts=False):
    """Object dirs under geom_out. The <id>_ghost__<state> pseudo-objects the
    `ghosts` stage writes for GUI correction are EXCLUDED unless ghosts=True --
    they are derived output, not annotations."""
    root = Path(capture) / geom_out
    keys = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and "__" in p.name
                  and (p / "masks_index.json").exists()
                  and (ghosts or "_ghost__" not in p.name))
    if state is not None:
        keys = [k for k in keys if k.endswith("__" + state)]
    return keys


def _obj_frames(capture, geom_out, key):
    """{frame: bool mask} of one object's verified masks."""
    root = Path(capture) / geom_out / key
    out = {}
    for f, meta in json.load(open(root / "masks_index.json")).items():
        m = cv2.imread(str(root / meta["mask_file"]), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            out[f] = m > 127
    return out


def _cam_by_frame(sess):
    return {str(sess.images[k]): k for k in sess.images.key_pairs()}


def _load_cloud(capture, ref, voxel, verbose):
    import open3d as o3d
    p = Path(capture) / "sessions" / ref / "raw_data" / "pointcloud.ply"
    if verbose:
        print(f"  [{ref}] loading lidar {p} ...", flush=True)
    pcd = o3d.io.read_point_cloud(str(p))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    P = np.asarray(pcd.points, dtype=np.float32)
    if verbose:
        print(f"  [{ref}] {len(P)} points @ voxel {voxel}m", flush=True)
    return P


# ─────────────────────── stage 1: objects (select+mesh) ───────────────────────
def _project(P, T, cam):
    """(idx, u, v, z) of the points that land in front of the camera (z > 0.2)
    and inside the image; idx indexes back into P."""
    R = np.asarray(T.R, np.float32)
    t = np.asarray(T.t, np.float32).reshape(3)
    pc = (P - t) @ R
    z = pc[:, 2]
    m = z > 0.2
    idx = np.nonzero(m)[0]
    fx, fy, cx, cy = M._K(cam)
    u = np.rint(fx * pc[m, 0] / z[m] + cx - 0.5).astype(np.int64)
    v = np.rint(fy * pc[m, 1] / z[m] + cy - 0.5).astype(np.int64)
    zi = z[m]
    ok = (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
    return idx[ok], u[ok], v[ok], zi[ok]


def _zbuffer(u, v, z, W, H, splat):
    D = np.full(H * W, np.inf, np.float32)
    order = np.argsort(z)[::-1]                      # nearest point written last
    D[(v * W + u)[order]] = z[order]
    D = D.reshape(H, W)
    if splat:
        k = 2 * splat + 1
        D = cv2.erode(D, np.ones((k, k), np.uint8))  # local min fills gaps
    return D


def _vote_mask(mask, W, H, mask_erode):
    mk = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    if mask_erode:
        k = 2 * mask_erode + 1
        mk = cv2.erode(mk, np.ones((k, k), np.uint8))
    return mk > 0


def _vote_points(P, sess, cbf, frames, scale, splat, occ_tol, min_views, ratio,
                 mask_erode, min_mask_px, verbose, tag):
    """Point analog of change_mask._vote_faces: a cloud point is kept if it is on
    the z-buffer front surface in >= min_views annotated frames AND falls inside
    the mask in >= ratio of those. Masks are eroded by `mask_erode` px first and
    frames whose eroded mask is under `min_mask_px` are skipped entirely (distant
    views alias adjacent-ray surfaces into the few mask pixels; they must not add
    seen counts either). Returns (keep_idx, seen, inside, used_frames)."""
    seen = np.zeros(len(P), np.int32)
    inside = np.zeros(len(P), np.int32)
    cams = {}
    used = {}
    for j, (frame, mask) in enumerate(frames.items()):
        ts, cam = cbf[frame]
        cam_v = cams.setdefault(cam, G._scaled_camera(sess.sensors[cam], scale)[0])
        W, H = cam_v.width, cam_v.height
        mk = _vote_mask(mask, W, H, mask_erode)
        if mk.sum() < min_mask_px:
            continue
        used[frame] = mask
        T = sess.get_pose(ts, cam)
        idx, u, v, zi = _project(P, T, cam_v)
        D = _zbuffer(u, v, zi, W, H, splat)
        vis = zi <= D[v, u] + occ_tol
        seen[idx[vis]] += 1
        inside[idx[vis & mk[v, u]]] += 1
        if verbose and j and j % 25 == 0:
            print(f"    [{tag}] voted {j}/{len(frames)} frames", flush=True)
    keep = (seen >= min_views) & (inside >= ratio * np.maximum(seen, 1))
    K = np.where(keep)[0]
    if verbose:
        print(f"    [{tag}] kept {len(K)}/{len(P)} points from {len(used)} views "
              f"({len(frames) - len(used)} skipped as too small)")
    return K, seen, inside, used


def _cluster_filter(P_sel, eps, merge, verbose, tag):
    """Keep the dominant DBSCAN cluster plus any cluster within `merge` metres of
    the kept set (a chair back and seat may split; floor sheets and distant-view
    leakage do not survive). Returns a bool keep mask over P_sel."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_sel.astype(np.float64)))
    lab = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=5))
    if lab.max() < 0:
        return np.ones(len(P_sel), bool)
    sizes = np.bincount(lab[lab >= 0])
    kept = {int(np.argmax(sizes))}
    cand = set(range(len(sizes))) - kept
    grew = True
    while grew and cand:
        grew = False
        tree = cKDTree(P_sel[np.isin(lab, list(kept))])
        for c in sorted(cand):
            if tree.query(P_sel[lab == c], k=1)[0].min() <= merge:
                kept.add(c)
                cand.discard(c)
                grew = True
                break
    keep = np.isin(lab, list(kept))
    if verbose:
        print(f"    [{tag}] cluster filter: kept {keep.sum()}/{len(P_sel)} pts "
              f"({len(kept)}/{len(sizes)} clusters)")
    return keep


def _reconstruct(P_sel, method, alpha, poisson_depth, density_q, tag):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P_sel.astype(np.float64)))
    if method == "poisson":
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                                  max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(30)
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth)
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.quantile(dens, density_q))
    else:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    return mesh


def _submesh(V, F, L):
    import open3d as o3d
    tris = F[L]
    used = np.unique(tris)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V[used]),
                                     o3d.utility.Vector3iVector(remap[tris]))


def _trim_mesh(mesh, sess, frames, scale, min_views, ratio, verbose, tag):
    """Mask-EXTERIOR trim: drop reconstructed faces that project outside the
    verified masks in too many annotated views (dual use of _vote_faces).
    Faces never visible from the annotated views are dropped too (backfaces of
    the shell); the remaining shell still spans the object's silhouette."""
    from scantools.proc.rendering import Renderer
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if len(F) == 0:
        return mesh, 0
    rd = Renderer(mesh)
    L, _, _ = CM._vote_faces(sess, rd, len(F), frames, scale, min_views, ratio,
                             verbose, tag + " trim")
    rd = None       # weakref finalizer frees the raybender scene (no release())
    gc.collect()
    if len(L) == 0:
        print(f"    [{tag}] WARNING: trim removed everything -- keeping untrimmed")
        return mesh, 0
    return _submesh(V, F, L), len(F) - len(L)


def cmd_objects(args):
    import open3d as o3d
    for st in _states(args).values():
        keys = _obj_keys(args.capture, args.geom_out, st["state"])
        if args.objects:
            keys = [k for k in keys if k in set(args.objects)]
        if not keys:
            continue
        _, sess = G._session(args.capture, st["session"], st["ref"])
        cbf = _cam_by_frame(sess)
        P = _load_cloud(args.capture, st["ref"], args.voxel, True)
        for key in keys:
            frames = _obj_frames(args.capture, args.geom_out, key)
            if not frames:
                print(f"  skip {key}: no readable masks")
                continue
            K, seen, inside, used = _vote_points(
                P, sess, cbf, frames, args.vote_scale, args.splat,
                args.point_occ_tol, args.min_views, args.ratio,
                args.mask_erode, args.min_mask_px, True, key)
            if not used:
                print(f"  skip {key}: every mask under --min-mask-px")
                continue
            n_voted = int(len(K))
            if args.cluster_filter and len(K):
                K = K[_cluster_filter(P[K], args.cluster_eps, args.cluster_merge,
                                      True, key)]
            odir = _out_root(args.capture) / key
            odir.mkdir(parents=True, exist_ok=True)
            stats = {"n_frames": len(frames), "n_frames_used": len(used),
                     "n_points_voted": n_voted, "n_points": int(len(K)),
                     "voxel": args.voxel, "min_views": args.min_views,
                     "ratio": args.ratio, "mask_erode": args.mask_erode,
                     "min_mask_px": args.min_mask_px, "surface": args.surface,
                     "n_faces": 0, "n_faces_trimmed": 0}
            pcd = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(P[K].astype(np.float64)))
            o3d.io.write_point_cloud(str(odir / "points.ply"), pcd)
            if len(K) < args.min_points:
                print(f"  [{key}] only {len(K)} points -- skipping mesh "
                      f"(degenerate, will be absent from the symmetric GT)")
                json.dump(stats, open(odir / "stats.json", "w"), indent=1)
                continue
            try:
                mesh = _reconstruct(P[K], args.surface, args.alpha,
                                    args.poisson_depth, args.density_q, key)
            except Exception as e:                    # Qhull can fail on slivers
                print(f"  [{key}] {args.surface} reconstruction failed: {e}")
                json.dump(stats, open(odir / "stats.json", "w"), indent=1)
                continue
            if args.trim:
                mesh, n_cut = _trim_mesh(mesh, sess, used, args.vote_scale,
                                         args.trim_min_views, args.trim_ratio,
                                         True, key)
                stats["n_faces_trimmed"] = int(n_cut)
            stats["n_faces"] = len(mesh.triangles)
            o3d.io.write_triangle_mesh(str(odir / "mesh.ply"), mesh)
            json.dump(stats, open(odir / "stats.json", "w"), indent=1)
            print(f"  [{key}] {len(K)} pts -> {len(mesh.triangles)} faces "
                  f"({stats['n_faces_trimmed']} trimmed) -> {odir}")
        P = None
        gc.collect()


# ─────────────────────── stage 2: symmetric GT ───────────────────────
def _merged_object_mesh(capture, keys, state, bridge=None, verbose=True):
    """One (V, F) mesh from every <key>__<state> object mesh, optionally carried
    through the bridge Pose3d. Returns (None, None) if no object has a mesh."""
    import open3d as o3d
    Vs, Fs, off = [], [], 0
    for key in keys:
        if not key.endswith("__" + state):
            continue
        p = _out_root(capture) / key / "mesh.ply"
        if not p.exists():
            if verbose:
                print(f"    [{state}] no object mesh for {key} (run `objects`)")
            continue
        m = o3d.io.read_triangle_mesh(str(p))
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        if len(F) == 0:
            if verbose:
                print(f"    [{state}] empty object mesh for {key}, skipped")
            continue
        Vs.append(G._apply_T(bridge, V) if bridge is not None else V)
        Fs.append(F + off)
        off += len(V)
    if not Vs:
        return None, None
    return np.vstack(Vs), np.vstack(Fs)


def _dst_scene_renderer(capture, dst):
    from scantools.proc.rendering import Renderer
    from scantools.utils.io import read_mesh
    capo_d, sess_d = G._session(capture, dst["session"], dst["ref"])
    R_B = Renderer(read_mesh(capo_d.proc_path(dst["ref"])
                             / capo_d.sessions[dst["ref"]].proc.meshes["mesh"]))
    return sess_d, R_B


def _render_ghost(V, F, sess_d, R_B, render_scale, occ_tol, verbose, tag):
    """Render the (V, F) ghost mesh into every dst cam0 frame, occlusion-tested
    against the dst scene renderer. Returns {dst_frame: bool mask}."""
    import open3d as o3d
    from scantools.proc.rendering import Renderer
    sub = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                    o3d.utility.Vector3iVector(F))
    R_L = Renderer(sub)
    keys_d = [k for k in sess_d.images.key_pairs() if "cam0" in str(k[1])]
    keys_d.sort(key=lambda k: k[0])
    out = {}
    for j, (ts, cam) in enumerate(keys_d):
        name = str(sess_d.images[ts, cam])
        cam_r, _, _ = G._scaled_camera(sess_d.sensors[cam], render_scale)
        T = sess_d.get_pose(ts, cam)
        _, depth_L = R_L.render_from_capture(T, cam_r)
        if not np.any(depth_L > 0):
            continue
        _, depth_B = R_B.render_from_capture(T, cam_r)
        occ = depth_B > 0
        vis = (depth_L > 0) & (~occ | (depth_L <= depth_B + occ_tol))
        if not vis.any():
            continue
        cam_o = sess_d.sensors[cam]
        if vis.shape != (cam_o.height, cam_o.width):
            vis = cv2.resize(vis.astype(np.uint8), (cam_o.width, cam_o.height),
                             interpolation=cv2.INTER_NEAREST) > 0
        out[name] = vis
        if verbose and j and j % 50 == 0:
            print(f"    [{tag}] rendered {j}/{len(keys_d)} frames "
                  f"({len(out)} hit)", flush=True)
    R_L = None
    gc.collect()
    if verbose:
        print(f"    [{tag}] footprint in {len(out)} dst frames")
    return out


def _ghost_from_objects(capture, src, dst, bridge, keys, *, render_scale, occ_tol,
                        verbose, dbg_dir=None):
    """change_mask._ghost_for_direction with the voted-face footprint replaced by
    the per-object point meshes. Returns {dst_frame: bool mask}."""
    import open3d as o3d
    capture = Path(capture)
    tag = f"{src['state']}->{dst['state']}"
    V, F = _merged_object_mesh(capture, keys, src["state"], bridge, verbose)
    if V is None:
        if verbose:
            print(f"    [{tag}] no object meshes")
        return {}
    if dbg_dir is not None:
        dbg = Path(dbg_dir)
        dbg.mkdir(parents=True, exist_ok=True)
        p = dbg / f"objects_{src['state']}_{src['ref']}_to_{dst['ref']}.ply"
        o3d.io.write_triangle_mesh(str(p), o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F)))
        if verbose:
            print(f"    [{tag}] saved bridged object mesh ({len(F)} faces) -> {p}")
    sess_d, R_B = _dst_scene_renderer(capture, dst)
    out = _render_ghost(V, F, sess_d, R_B, render_scale, occ_tol, verbose, tag)
    R_B = None
    gc.collect()
    return out


def cmd_symmetric(args):
    capture = Path(args.capture)
    states = _states(args)
    keys = _obj_keys(capture, args.geom_out)
    unions, prov = CM._native_unions(capture, args.geom_out, keys, True)
    native_snap = {s: dict(fr) for s, fr in unions.items()}
    bridge = args.bridge or str(capture / "changes"
                                / f"{args.post_ref}_to_{args.pre_ref}"
                                / f"T_{args.pre_ref}_from_{args.post_ref}.txt")
    T = G._load_T(bridge)
    dbg = capture / "changes" / "change_mask_symmetric_points"
    params = dict(render_scale=args.render_scale, occ_tol=args.occ_tol,
                  verbose=True, dbg_dir=dbg)
    pre, post = states["pre"], states["post"]
    ghosts = {"post": _ghost_from_objects(capture, pre, post, T.inverse(),
                                          keys, **params),
              "pre": _ghost_from_objects(capture, post, pre, T, keys, **params)}
    for state, gmap in ghosts.items():
        sess = states[state]["session"]
        for frame, g in gmap.items():
            acc = unions.setdefault(state, {}).get(frame)
            if acc is not None and g.shape != acc.shape:
                g = cv2.resize(g.astype(np.uint8), (acc.shape[1], acc.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
            unions[state][frame] = g if acc is None else (acc | g)
            ghosts[state][frame] = g
            prov.setdefault(state, {}).setdefault(frame, {"session": sess})
    block = CM._write_masks(capture, "change_mask_symmetric_points", unions, prov)
    if args.viz:
        split = {s: {f: (native_snap.get(s, {}).get(f), ghosts.get(s, {}).get(f))
                     for f in fr} for s, fr in unions.items()}
        CM._write_viz(capture, "change_mask_symmetric_points", unions, prov,
                      split, True)
    for state, frames in block.items():
        print(f"  {state}: {len(frames)} symmetric change frames")


# ──────────────── stage 2b: GUI-editable ghosts + corrected merge ────────────────
def _bridge_path(args, capture):
    return args.bridge or str(capture / "changes"
                              / f"{args.post_ref}_to_{args.pre_ref}"
                              / f"T_{args.pre_ref}_from_{args.post_ref}.txt")


def _replace_json(path, obj):
    """Atomic write via rename: the shared annotation workspace is group-owned,
    so an existing file may not be writable by this user even though the
    directory is -- os.replace only needs directory write."""
    import os
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def cmd_ghosts(args):
    """Render each object's ghost SEPARATELY into the other state's frames and
    write it as a GUI-editable pseudo-object: <geom_out>/<id>_ghost__<dst_state>/
    masks_index.json in the exact format gui.py's review/edit mode consumes
    (brush/eraser + per-frame delete), plus a gui_objects.json entry flagged
    ghost:true (gui.py hides propagate/seed for these and skips them at export).
    After hand-correcting in the GUI, run `merge` to rebuild the symmetric GT."""
    import shutil
    capture = Path(args.capture)
    states = _states(args)
    keys = _obj_keys(capture, args.geom_out)
    T = G._load_T(_bridge_path(args, capture))
    gobj_p = capture / args.geom_out / "gui_objects.json"
    gobj = json.load(open(gobj_p)) if gobj_p.exists() else {}
    for src, dst, B in ((states["pre"], states["post"], T.inverse()),
                        (states["post"], states["pre"], T)):
        src_keys = [k for k in keys if k.endswith("__" + src["state"])]
        if not src_keys:
            continue
        sess_d, R_B = _dst_scene_renderer(capture, dst)
        for key in src_keys:
            V, F = _merged_object_mesh(capture, [key], src["state"], B, True)
            if V is None:
                continue
            oid = key.rsplit("__", 1)[0] + "_ghost"
            gkey = f"{oid}__{dst['state']}"
            out = _render_ghost(V, F, sess_d, R_B, args.render_scale,
                                args.occ_tol, True, f"{key}->{dst['state']}")
            if not out:
                print(f"  [{gkey}] ghost hits no {dst['state']} frame, skipped")
                continue
            od = capture / args.geom_out / gkey
            if (od / "masks").exists():           # regeneration is idempotent
                shutil.rmtree(od / "masks")
            (od / "masks").mkdir(parents=True, exist_ok=True)
            mi = {}
            for frame in sorted(out):
                flat = frame.replace("/", "_")
                cv2.imwrite(str(od / "masks" / flat),
                            out[frame].astype(np.uint8) * 255)
                mi[frame] = {"session": dst["session"],
                             "mask_file": f"masks/{flat}",
                             "px": int(out[frame].sum())}
            json.dump(mi, open(od / "masks_index.json", "w"), indent=1)
            src_entry = gobj.get(key, {})
            gobj[gkey] = {"id": oid,
                          "label": f"{src_entry.get('label', key.rsplit('__', 1)[0])} (ghost)",
                          "deformability": src_entry.get("deformability", "rigid"),
                          "state": dst["state"], "frame": next(iter(sorted(out))),
                          "points": [], "seed_frames": [], "ghost": True}
            print(f"  [{gkey}] {len(mi)} ghost masks -> {od}")
        R_B = None
        gc.collect()
    _replace_json(gobj_p, gobj)
    print(f"  registered ghost objects in {gobj_p} -- reload the GUI to edit them")


def cmd_merge(args):
    """Rebuild the symmetric GT from the object dirs: native masks UNION the
    (possibly hand-corrected) <id>_ghost__<state> dirs. Run after editing ghosts
    in the GUI; replaces changes/change_mask_symmetric_points/."""
    capture = Path(args.capture)
    keys = _obj_keys(capture, args.geom_out, ghosts=True)
    native_keys = [k for k in keys if "_ghost__" not in k]
    ghost_keys = [k for k in keys if "_ghost__" in k]
    if not ghost_keys:
        raise SystemExit("no <id>_ghost__<state> dirs -- run `ghosts` first")
    unions, prov = CM._native_unions(capture, args.geom_out, native_keys, True)
    native_snap = {s: dict(fr) for s, fr in unions.items()}
    gunions, gprov = CM._native_unions(capture, args.geom_out, ghost_keys, True)
    for state, frames in gunions.items():
        for frame, g in frames.items():
            acc = unions.setdefault(state, {}).get(frame)
            if acc is not None and g.shape != acc.shape:
                g = cv2.resize(g.astype(np.uint8), (acc.shape[1], acc.shape[0]),
                               interpolation=cv2.INTER_NEAREST) > 0
            unions[state][frame] = g if acc is None else (acc | g)
            prov.setdefault(state, {}).setdefault(
                frame, gprov.get(state, {}).get(frame, {}))
    block = CM._write_masks(capture, "change_mask_symmetric_points", unions, prov)
    if args.viz:
        split = {s: {f: (native_snap.get(s, {}).get(f),
                         gunions.get(s, {}).get(f)) for f in fr}
                 for s, fr in unions.items()}
        CM._write_viz(capture, "change_mask_symmetric_points", unions, prov,
                      split, True)
    for state, frames in block.items():
        print(f"  {state}: {len(frames)} symmetric change frames "
              f"(ghosts from {len(ghost_keys)} editable dirs)")


# ─────────────────────── stage 3: compare old vs new ───────────────────────
def _index_mask(capture, idx, state, frame):
    meta = idx.get(state, {}).get(frame)
    if meta is None:
        return None
    m = cv2.imread(str(Path(capture) / "changes" / meta["mask_file"]),
                   cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 127


def _label(img, txt):
    cv2.putText(img, txt, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 3)
    return img


def cmd_compare(args):
    """[RGB | old symmetric | new symmetric] panels for every frame either method
    marked, plus a source-frame IoU table: each method's SOURCE geometry rendered
    back into the annotated frames vs. the verified masks. Old geometry is the
    footprint_*.ply the --viz runs dump; new is the union of object meshes.
    (Neither render is occlusion-tested against the scene, so frames where the
    object is partly hidden under-score both methods equally.)"""
    from scantools.proc.rendering import Renderer
    import open3d as o3d
    capture = Path(args.capture)
    states = _states(args)
    keys = _obj_keys(capture, args.geom_out)
    new_sub = "change_mask_symmetric_points"
    odir = capture / "changes" / "point_ghost_compare"

    # panels
    old_idx_p = capture / "changes" / f"{args.old}_index.json"
    new_idx_p = capture / "changes" / f"{new_sub}_index.json"
    old_idx = json.load(open(old_idx_p)) if old_idx_p.exists() else {}
    new_idx = json.load(open(new_idx_p)) if new_idx_p.exists() else {}
    for state in ("pre", "post"):
        frames = sorted(set(old_idx.get(state, {})) | set(new_idx.get(state, {})))
        if not frames:
            continue
        vdir = odir / state
        vdir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            sess_id = (old_idx.get(state, {}).get(frame) or
                       new_idx.get(state, {}).get(frame))["session"]
            rgb = cv2.imread(str(CM._rgb_path(capture, sess_id, frame)))
            om = _index_mask(capture, old_idx, state, frame)
            nm = _index_mask(capture, new_idx, state, frame)
            ref = om if om is not None else nm
            if ref is None:
                continue
            H, W = ref.shape
            if rgb is None:
                rgb = np.zeros((H, W, 3), np.uint8)
            elif rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H))
            def _ov(m, col):
                if m is None:
                    return _label(rgb.copy(), "missing")
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST) > 0
                o = rgb.copy()
                o[m] = col
                return cv2.addWeighted(rgb, 0.5, o, 0.5, 0)
            panel = np.hstack([rgb, _label(_ov(om, (0, 255, 0)), f"old: {args.old}"),
                               _label(_ov(nm, (0, 0, 255)), "new: points")])
            s = args.panel_scale
            if s < 0.999:
                panel = cv2.resize(panel, (int(panel.shape[1] * s),
                                           int(panel.shape[0] * s)))
            cv2.imwrite(str(vdir / Path(CM._flat_png(frame)).with_suffix(".jpg").name),
                        panel)
        print(f"  {state}: {len(frames)} comparison panels -> {vdir}")

    # source-frame IoU: render each method's geometry into the annotated frames
    unions, _ = CM._native_unions(capture, args.geom_out, keys, False)
    report = {}
    for state, st in states.items():
        frames = unions.get(state, {})
        if not frames:
            continue
        _, sess = G._session(capture, st["session"], st["ref"])
        cbf = _cam_by_frame(sess)
        geoms = {}
        V, F = _merged_object_mesh(capture, keys, state, None, False)
        if V is not None:
            geoms["new_points"] = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
        fp = capture / "changes" / args.old / f"footprint_{state}_{st['ref']}_src.ply"
        if fp.exists():
            geoms[f"old_{args.old}"] = o3d.io.read_triangle_mesh(str(fp))
        else:
            print(f"  [{state}] no {fp.name} in changes/{args.old} "
                  f"(rerun the old method with --viz) -- old IoU skipped")
        for name, mesh in geoms.items():
            rd = Renderer(mesh)
            ious = []
            for frame, mask in frames.items():
                ts, cam = cbf[frame]
                cam_v, _, _ = G._scaled_camera(sess.sensors[cam], args.vote_scale)
                _, depth = rd.render_from_capture(sess.get_pose(ts, cam), cam_v)
                sil = cv2.resize((depth > 0).astype(np.uint8),
                                 (mask.shape[1], mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST) > 0
                inter, union = (sil & mask).sum(), (sil | mask).sum()
                ious.append(inter / union if union else 1.0)
            rd = None
            gc.collect()
            report.setdefault(state, {})[name] = {
                "mean_iou": float(np.mean(ious)), "n_frames": len(ious)}
            print(f"  [{state}] {name}: mean source-frame IoU "
                  f"{np.mean(ious):.3f} over {len(ious)} frames")
    odir.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(odir / "iou.json", "w"), indent=1)
    print(f"  IoU table -> {odir / 'iou.json'}")


# ──────────────────────────────── CLI ────────────────────────────────
def _states(a):
    return {"pre": {"state": "pre", "session": a.pre_session, "ref": a.pre_ref},
            "post": {"state": "post", "session": a.post_session, "ref": a.post_ref}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--capture", required=True)
    common.add_argument("--geom-out", default=CM.GEOM_OUT)
    common.add_argument("--pre-session", default="aria_a_rgb")
    common.add_argument("--pre-ref", default="navvis_a")
    common.add_argument("--post-session", default="aria_b_rgb")
    common.add_argument("--post-ref", default="navvis_b")
    common.add_argument("--vote-scale", type=float, default=VOTE_SCALE)

    o = sub.add_parser("objects", parents=[common],
                       help="mask-vote lidar points per object, mesh + trim")
    o.add_argument("--objects", nargs="*", default=None,
                   help="restrict to these <id>__<state> keys")
    o.add_argument("--voxel", type=float, default=CLOUD_VOXEL)
    o.add_argument("--splat", type=int, default=SPLAT)
    o.add_argument("--point-occ-tol", type=float, default=POINT_OCC_TOL)
    o.add_argument("--min-views", type=int, default=3)
    o.add_argument("--ratio", type=float, default=0.5)
    o.add_argument("--mask-erode", type=int, default=MASK_ERODE,
                   help="erode masks this many vote-res px before voting")
    o.add_argument("--min-mask-px", type=int, default=MIN_MASK_PX,
                   help="skip frames whose eroded mask is smaller (vote-res px)")
    o.add_argument("--no-cluster-filter", dest="cluster_filter",
                   action="store_false")
    o.add_argument("--cluster-eps", type=float, default=CLUSTER_EPS)
    o.add_argument("--cluster-merge", type=float, default=CLUSTER_MERGE)
    o.add_argument("--min-points", type=int, default=50)
    o.add_argument("--surface", choices=("alpha", "poisson"), default="alpha")
    o.add_argument("--alpha", type=float, default=ALPHA)
    o.add_argument("--poisson-depth", type=int, default=8)
    o.add_argument("--density-q", type=float, default=0.05,
                   help="poisson: drop vertices below this density quantile")
    o.add_argument("--no-trim", dest="trim", action="store_false")
    o.add_argument("--trim-min-views", type=int, default=2)
    o.add_argument("--trim-ratio", type=float, default=0.5)
    o.set_defaults(fn=cmd_objects)

    s = sub.add_parser("symmetric", parents=[common],
                       help="symmetric GT from the object meshes -> "
                            "change_mask_symmetric_points/")
    s.add_argument("--bridge", default=None)
    s.add_argument("--render-scale", type=float, default=1.0)
    s.add_argument("--occ-tol", type=float, default=0.10)
    s.add_argument("--viz", action="store_true")
    s.set_defaults(fn=cmd_symmetric)

    g = sub.add_parser("ghosts", parents=[common],
                       help="per-object ghost masks as GUI-editable "
                            "pseudo-objects (<id>_ghost__<state>)")
    g.add_argument("--bridge", default=None)
    g.add_argument("--render-scale", type=float, default=1.0)
    g.add_argument("--occ-tol", type=float, default=0.10)
    g.set_defaults(fn=cmd_ghosts)

    m = sub.add_parser("merge", parents=[common],
                       help="rebuild symmetric GT from native + (corrected) "
                            "ghost dirs")
    m.add_argument("--viz", action="store_true")
    m.set_defaults(fn=cmd_merge)

    c = sub.add_parser("compare", parents=[common],
                       help="old-vs-new panels + source-frame IoU table")
    c.add_argument("--old", default="change_mask_symmetric",
                   help="old symmetric subdir (e.g. change_mask_symmetric_tsdf)")
    c.add_argument("--panel-scale", type=float, default=0.5)
    c.set_defaults(fn=cmd_compare)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
