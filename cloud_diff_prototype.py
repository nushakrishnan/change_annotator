"""Cloud-diff change PROPOSER.

Detect candidate changed objects between two states by differencing their raw
NavVis point clouds, gate/rank them by how much the Aria glasses actually looked
at each one, and hand every survivor to the existing per-frame masking pipeline
(geom_sam_prototype) so it lands in the GUI as a reviewable object.

Why the RAW clouds, not the meshes: NavVis surface meshing drops most changed
objects (see point_ghost_prototype.py's docstring); the raw lidar scan keeps them.
Why a diff works: a moved/added/removed object is orders of magnitude larger than
the cm-scale NavVis->NavVis control-point alignment residual, so a generous
nearest-neighbour distance threshold cleanly isolates the change.
Why the Aria-attention gate is essential: the two NavVis scans cover a whole
building floor, so a raw diff over-proposes everywhere. But GT lives ONLY in Aria
image space -- a change the glasses never looked at is unannotatable and yields
zero masks. So visible-frame count is both the relevance filter and a hard
annotatability test; it collapses the building-wide diff to the walked region.

Pipeline:
  1. load both raw clouds; bring `post` into the `pre` (ref) frame via the rigid
     NavVis->NavVis bridge; restrict to the volume both scans observed;
  2. two-way nearest-neighbour distance (scipy cKDTree), threshold `tau`:
       pre  points with no post neighbour -> old-location set (native to pre)
       post points with no pre  neighbour -> new-location set (native to post)
  3. DBSCAN each side into candidate clusters (keep every cluster >= min size);
  4. per cluster, reproject into its NATIVE state's Aria frames (occlusion-tested,
     geom_sam_prototype._seeds_for_session) -> visible-frame count = attention;
     drop clusters below `min_frames`, keep the rest ranked;
  5. write each survivor's per-frame seeds as a GUI-reviewable object.

MODE B (current): each cluster is proposed as its OWN object (unique id). A moved
object therefore arrives as a pre-state and a post-state object with DIFFERENT
ids; you link them (shared id -> "moved") in the GUI. MODE A (auto-association of
old/new-location pairs into one moved instance) is a later refinement.

Standalone dev harness (the annotator drives this from the GUI button):
  PYTHONPATH=~/repos/lamaria-indoor GEOM_OUT=changes/geom_sam_out_1_2_sangwoo \
    ~/annotator_env/bin/python cloud_diff_prototype.py propose \
      --capture /media/lamaria_indoor/captures/changes/dlab_open_space
"""
import argparse
import json
from pathlib import Path

import numpy as np

import geom_sam_prototype as G


# ───────────────────────────── cloud helpers ─────────────────────────────
def _load_cloud(capture, ref, voxel, verbose=True):
    """Raw NavVis cloud for `ref`, voxel-downsampled. Returns (N,3) float64."""
    import open3d as o3d
    p = Path(capture) / "sessions" / ref / "raw_data" / "pointcloud.ply"
    if verbose:
        print(f"  [{ref}] loading {p} ...", flush=True)
    pcd = o3d.io.read_point_cloud(str(p))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    P = np.asarray(pcd.points, dtype=np.float64)
    if verbose:
        print(f"  [{ref}] {len(P):,} points @ voxel {voxel} m", flush=True)
    return P


def _cluster(P, eps, min_points, min_cluster, verbose, tag):
    """DBSCAN -> list of (M,3) clusters with >= min_cluster points, largest first."""
    import open3d as o3d
    if len(P) == 0:
        return []
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    lab = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    clusters = [P[lab == c] for c in (range(lab.max() + 1) if lab.max() >= 0 else [])]
    clusters = sorted((c for c in clusters if len(c) >= min_cluster), key=len, reverse=True)
    if verbose:
        kept = ", ".join(str(len(c)) for c in clusters) or "none"
        print(f"    [{tag}] -> {len(clusters)} clusters >= {min_cluster} pts: [{kept}]")
    return clusters


def _signature(pts):
    c = pts.mean(0)
    ext = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov((pts - c).T)), 0, None))[::-1]
    return {"n": int(len(pts)),
            "centroid": [round(float(v), 3) for v in c],
            "extent_m": [round(float(v), 3) for v in ext]}


# ───────────────────────────── detect ─────────────────────────────
def detect(capture, pre_ref, post_ref, bridge_path, *, tau=0.10, voxel=0.02,
           eps=0.10, min_points=10, min_cluster=500, overlap_margin=0.0, verbose=True):
    """Two-way cloud diff -> {'old_clusters','new_clusters'} (each cluster's points
    in its NATIVE state's world frame: old in pre/ref, new in post/ref)."""
    from scipy.spatial import cKDTree

    P1 = _load_cloud(capture, pre_ref, voxel, verbose)        # pre (ref) frame
    P2 = _load_cloud(capture, post_ref, voxel, verbose)       # post (ref) frame
    T = G._load_T(bridge_path)                                # T_pre_from_post
    P2in1 = G._apply_T(T, P2)                                 # post cloud in pre frame

    lo = np.maximum(P1.min(0), P2in1.min(0)) - overlap_margin
    hi = np.minimum(P1.max(0), P2in1.max(0)) + overlap_margin
    in1 = np.all((P1 >= lo) & (P1 <= hi), axis=1)
    in2 = np.all((P2in1 >= lo) & (P2in1 <= hi), axis=1)
    P1o, P2o = P1[in1], P2in1[in2]

    d_old = cKDTree(P2o).query(P1o, workers=-1)[0]
    d_new = cKDTree(P1o).query(P2o, workers=-1)[0]
    old = P1o[d_old > tau]                                    # old location, pre frame
    new = G._apply_T(T.inverse(), P2o[d_new > tau])           # new location -> post frame
    if verbose:
        print(f"  changed points  old(pre)={len(old):,}  new(post)={len(new):,} (tau={tau} m)")
    return {"old_clusters": _cluster(old, eps, min_points, min_cluster, verbose, "old/pre"),
            "new_clusters": _cluster(new, eps, min_points, min_cluster, verbose, "new/post")}


def _candidates_from_diff(res, prefix):
    """Flat per-cluster candidates (MODE B): each cluster is its own object with a
    unique id. old -> pre/'removed', new -> post/'added' (final change_type is
    derived from id presence in the GUI; unique ids => removed/added until you
    link a pair by giving them a shared id)."""
    cands = []
    for i, pts in enumerate(res["old_clusters"], 1):
        cands.append({"id": f"{prefix}_pre_{i:02d}", "state": "pre",
                      "change_type": "removed", "pts": pts, "sig": _signature(pts)})
    for i, pts in enumerate(res["new_clusters"], 1):
        cands.append({"id": f"{prefix}_post_{i:02d}", "state": "post",
                      "change_type": "added", "pts": pts, "sig": _signature(pts)})
    return cands


# ───────────────────────── attention (Aria visibility) ─────────────────────────
def _ctx_for_states(capture, states, cands):
    """{state: (capo, sess, renderer)} for the states present among candidates;
    renderer holds that state's NavVis mesh for the occlusion test."""
    from scantools.proc.rendering import Renderer
    from scantools.utils.io import read_mesh
    ctx = {}
    for state in ("pre", "post"):
        if any(c["state"] == state for c in cands):
            ref, sid = states[state]["ref"], states[state]["session"]
            capo, sess = G._session(capture, sid, ref)
            mesh = capo.proc_path(ref) / capo.sessions[ref].proc.meshes["mesh"]
            ctx[state] = (capo, sess, Renderer(read_mesh(mesh)))
    return ctx


def _seed(cands, states, ctx, n, min_vis, occ_scale, dbg_root):
    """For each candidate, reproject its cluster into its native state's Aria
    frames (n=0 = all, N>0 subsample). Sets c['seeds'] (per-frame point+box) and
    c['n_frames'] (the Aria-attention score)."""
    for c in cands:
        capo, sess, renderer = ctx[c["state"]]
        seeds = G._seeds_for_session(c["pts"], sess, renderer, capo,
                                     states[c["state"]]["session"], n, min_vis,
                                     dbg_root, occ_scale=occ_scale, write_dbg=False)
        c["seeds"], c["n_frames"] = seeds, len(seeds)


# ───────────────────────────── propose ─────────────────────────────
def propose(capture, states, bridge_path, *, tau=0.10, voxel=0.02, eps=0.10,
            min_points=10, min_cluster=500, gate_n=150, min_vis=5, occ_scale=0.35,
            min_frames=3, prefix="cd", progress=None, verbose=True):
    """Full proposer up to (not including) SAM: diff -> per-cluster candidates ->
    Aria-attention gate (cheap subsample) -> full-frame seeds on survivors ->
    write <GEOM_OUT>/<id>__<state>/seeds.json. Returns a list of
    (key, session, gui_entry, n_frames) sorted by attention, most-seen first.
    SAM per-frame masks are run by the caller (the GUI reuses its shared model)."""
    say = progress or (lambda s: None)
    dbg_root = Path(capture) / "changes" / "cloud_diff" / "seed_dbg"

    say("differencing point clouds …")
    res = detect(capture, states["pre"]["ref"], states["post"]["ref"], bridge_path,
                 tau=tau, voxel=voxel, eps=eps, min_points=min_points,
                 min_cluster=min_cluster, verbose=verbose)
    cands = _candidates_from_diff(res, prefix)
    say(f"{len(cands)} raw candidates; building geometry contexts …")
    ctx = _ctx_for_states(capture, states, cands)

    say(f"gating {len(cands)} candidates by Aria visibility (~{gate_n} frames) …")
    _seed(cands, states, ctx, gate_n, min_vis, occ_scale, dbg_root)
    survivors = sorted((c for c in cands if c["n_frames"] >= min_frames),
                       key=lambda c: -c["n_frames"])
    say(f"{len(survivors)}/{len(cands)} candidates seen by Aria "
        f"(>= {min_frames} frames); seeding all frames on survivors …")
    _seed(survivors, states, ctx, 0, min_vis, occ_scale, dbg_root)

    out = []
    for c in survivors:
        key = f"{c['id']}__{c['state']}"
        od = G.out_dir(capture, key)                         # respects GEOM_OUT
        json.dump(c["seeds"], open(od / "seeds.json", "w"), indent=1)
        entry = {"id": c["id"], "label": "", "deformability": "rigid",
                 "state": c["state"], "frame": next(iter(c["seeds"])),
                 "points": [], "seed_frames": [], "source": "cloud_diff"}
        out.append((key, states[c["state"]]["session"], entry, c["n_frames"]))
    say(f"wrote seeds for {len(out)} proposals")
    return out


# ───────────────────────────── CLI (dev harness) ─────────────────────────────
def _bridge(capture, pre_ref, post_ref, override):
    return override or str(Path(capture) / "changes" / f"{post_ref}_to_{pre_ref}"
                           / f"T_{pre_ref}_from_{post_ref}.txt")


def _states(a):
    return {"pre": {"session": a.pre_session, "ref": a.pre_ref},
            "post": {"session": a.post_session, "ref": a.post_ref}}


def cmd_diff(a):
    """Geometry-only: diff -> candidate clusters, saved as .ply for inspection."""
    import open3d as o3d
    res = detect(a.capture, a.pre_ref, a.post_ref,
                 _bridge(a.capture, a.pre_ref, a.post_ref, a.bridge),
                 tau=a.tau, voxel=a.voxel, eps=a.eps, min_points=a.min_points,
                 min_cluster=a.min_cluster)
    cands = _candidates_from_diff(res, a.prefix)
    root = Path(a.capture) / "changes" / "cloud_diff"
    root.mkdir(parents=True, exist_ok=True)
    for c in cands:
        o3d.io.write_point_cloud(str(root / f"{c['id']}__{c['state']}.ply"),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c["pts"])))
    json.dump([{k: v for k, v in c.items() if k != "pts"} for c in cands],
              open(root / "candidates.json", "w"), indent=1)
    print(f"\n  {len(cands)} candidates -> {root}")


def cmd_attention(a):
    """diff -> per-cluster -> Aria-visibility ranking (no seeds written)."""
    res = detect(a.capture, a.pre_ref, a.post_ref,
                 _bridge(a.capture, a.pre_ref, a.post_ref, a.bridge),
                 tau=a.tau, voxel=a.voxel, eps=a.eps, min_points=a.min_points,
                 min_cluster=a.min_cluster)
    cands = _candidates_from_diff(res, a.prefix)
    ctx = _ctx_for_states(a.capture, _states(a), cands)
    _seed(cands, _states(a), ctx, a.n, a.min_vis, a.occ_scale,
          Path(a.capture) / "changes" / "cloud_diff" / "seed_dbg")
    print(f"\n  Aria attention (visible frames{'' if a.n == 0 else f', ~{a.n} sampled'}):")
    for c in sorted(cands, key=lambda c: -c["n_frames"]):
        print(f"    {c['id']:12s} [{c['change_type']:7s}] {c['n_frames']:4d}f  "
              f"{c['sig']['n']}pts @ {c['sig']['centroid']}")


def cmd_propose(a):
    """Full propose (writes seeds + merges gui_objects.json in the GEOM_OUT
    workspace). SAM masks are NOT run here -- launch the GUI (or its
    /api/cloud_diff button) to fill them with the shared model."""
    out = propose(a.capture, _states(a),
                  _bridge(a.capture, a.pre_ref, a.post_ref, a.bridge),
                  tau=a.tau, voxel=a.voxel, eps=a.eps, min_points=a.min_points,
                  min_cluster=a.min_cluster, gate_n=a.gate_n, min_vis=a.min_vis,
                  occ_scale=a.occ_scale, min_frames=a.min_frames, prefix=a.prefix,
                  progress=lambda s: print("  ·", s, flush=True))
    gp = G.out_dir(a.capture) / "gui_objects.json"
    gobj = json.load(open(gp)) if gp.exists() else {}
    for key, _sid, entry, _n in out:
        gobj[key] = entry
    json.dump(gobj, open(gp, "w"), indent=1)
    print(f"\n  {len(out)} proposals -> {gp}  (workspace {G.OUT})")
    for key, _sid, _e, n in out:
        print(f"    {key:18s} {n:4d} frames")
    print("  launch the GUI on this workspace and run SAM (button / review) to fill masks.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("diff", "attention", "propose"):
        p = sub.add_parser(name)
        p.add_argument("--capture", required=True)
        p.add_argument("--pre-session", default="dlab_open_space_1_rgb")
        p.add_argument("--pre-ref", default="navvis_1")
        p.add_argument("--post-session", default="dlab_open_space_2_rgb")
        p.add_argument("--post-ref", default="navvis_2")
        p.add_argument("--bridge", default=None)
        p.add_argument("--tau", type=float, default=0.10)
        p.add_argument("--voxel", type=float, default=0.02)
        p.add_argument("--eps", type=float, default=0.10)
        p.add_argument("--min-points", type=int, default=10)
        p.add_argument("--min-cluster", type=int, default=500)
        p.add_argument("--prefix", default="cd")
        if name == "attention":
            p.add_argument("--n", type=int, default=0)
            p.add_argument("--min-vis", type=int, default=5)
            p.add_argument("--occ-scale", type=float, default=0.35)
        if name == "propose":
            p.add_argument("--gate-n", type=int, default=150)
            p.add_argument("--min-vis", type=int, default=5)
            p.add_argument("--occ-scale", type=float, default=0.35)
            p.add_argument("--min-frames", type=int, default=3)
    a = ap.parse_args()
    {"diff": cmd_diff, "attention": cmd_attention, "propose": cmd_propose}[a.cmd](a)


if __name__ == "__main__":
    main()
