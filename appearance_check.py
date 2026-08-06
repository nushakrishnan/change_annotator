"""Texture / appearance change detection on GEOMETRICALLY STATIC surfaces.

The certified-static field (cloud_diff_prototype.compute_fields) proves shape
didn't move — but a poster swapped for a different poster, a screen showing new
content, a re-printed banner are changes the lidar cannot see. This module
closes that hole:

  1. tile the BOTH-states-static surface into cells (0.20 m);
  2. per cell, find its best (nearest, occlusion-tested) view in the PRE walk
     and in the POST walk — the scans' registration is the pairing, no shared
     trajectories needed;
  3. pool DINOv3 dense features (timm convnext_base.dinov3_lvd1689m — robust to
     the lighting/viewpoint differences that kill pixel metrics, sensitive to
     texture/content) over each cell in both views; cosine-compare;
  4. SELF-CALIBRATED flagging: the similarity distribution over all static
     cells absorbs global lighting shifts; genuine content changes are the low
     tail (mean - 3*sigma, floor-capped);
  5. cluster flagged cells into incidents and hand each back as a proposal
     candidate — cluster.npy + seeds + per-frame SAM masks via the existing
     pipeline, so a poster change lands in the GUI as a reviewable object
     (tex_NN__pre / tex_NN__post) exactly like a cloud-diff proposal.

Standalone:
  PYTHONPATH=~/repos/lamaria-indoor GEOM_OUT=changes/geom_sam_out_1_2_sangwoo \
    ~/annotator_env/bin/python appearance_check.py --capture .../climate_day \
      --pre-session climate_day_1_1_rgb --pre-ref navvis_2 \
      --post-session climate_day_2_1_rgb --post-ref navvis_3 [--dry]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import geom_sam_prototype as G
import cloud_diff_prototype as CD

CELL = 0.20            # m, appearance cell size
FRAME_STEP = 5         # sample every k-th frame for best-view search
MAX_VIEW_D = 6.0       # m, beyond this DINO patches are too coarse to judge
IMG_LONG = 1536        # feature-extraction resolution (long side)
MIN_CELLS = 2          # incident = >= this many flagged cells (kills speckle)
N_VIEWS = 4            # embeddings averaged over up to N views per cell — the
                       # per-view nuisance (obliquity, exposure, pose slop) was
                       # measured at sigma 0.17 single-view; averaging divides it
DBSCAN_EPS = 0.45      # m, incident grouping


def _load_dino(progress):
    import timm
    import torch
    progress("loading DINOv3 (convnext_base.dinov3_lvd1689m) …")
    model = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True,
                              num_classes=0)
    model.eval().cuda()
    cfg = timm.data.resolve_model_data_config(model)
    mean = np.array(cfg["mean"], np.float32) * 255
    std = np.array(cfg["std"], np.float32) * 255
    return model, mean, std


CROP_M = 0.45          # physical window compared per cell (m)


def _embed_crops(model, mean, std, crops, batch=192):
    """Normalized pooled embeddings for a list of BGR crops (any sizes)."""
    import torch
    out = []
    for i in range(0, len(crops), batch):
        xs = []
        for c in crops[i:i + batch]:
            img = cv2.resize(c, (224, 224))
            rgb = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) - mean) / std
            xs.append(rgb.transpose(2, 0, 1))
        t = torch.from_numpy(np.stack(xs)).cuda()
        with torch.no_grad():
            e = model(t)                              # (B, C) pooled
        e = torch.nn.functional.normalize(e, dim=1)
        out.append(e.cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 1))


def _best_views(centers, capture, session, ref, step, progress, tag, offset=0):
    """Per cell: (frame_name, px, py, depth) of its nearest unoccluded view.
    `centers` must be in the STATE's native world frame."""
    from scantools.proc.rendering import Renderer
    from scantools.utils.io import read_mesh
    from scantools.utils.geometry import project, sample_depth
    capo, sess = G._session(capture, session, ref)
    mesh = capo.proc_path(ref) / capo.sessions[ref].proc.meshes["mesh"]
    renderer = Renderer(read_mesh(mesh))
    keys = sorted((k for k in sess.images.key_pairs() if "cam0" in str(k[1])),
                  key=lambda k: k[0])[offset::step]
    best = {}                                        # ci -> (d, name, px, py)
    occ_cams = {}
    for j, (ts, cam_t) in enumerate(keys):
        if j % 25 == 0:
            progress(f"[{tag}] best-view scan {j}/{len(keys)}")
        cam = sess.sensors[cam_t]
        if cam_t not in occ_cams:
            occ_cams[cam_t] = G._scaled_camera(cam, 0.5)
        cam_s, sx, sy = occ_cams[cam_t]
        T = sess.get_pose(ts, cam_t)
        name = str(sess.images[ts, cam_t])
        p2d, z, vis = project(centers, cam, pose=T.inverse())
        if not vis.any():
            continue
        _, depth = renderer.render_from_capture(T, cam_s)
        occ_z, occ_ok = sample_depth(p2d[vis] * np.array([sx, sy]), depth)
        good = occ_ok & (z[vis] <= occ_z + 0.08) & (z[vis] < MAX_VIEW_D)
        idx = np.where(vis)[0][good]
        for ci, (px, py), d in zip(idx, p2d[vis][good], z[vis][good]):
            views = best.setdefault(ci, [])
            views.append((float(d), name, float(px), float(py)))
            if len(views) > N_VIEWS:                 # keep the N nearest views
                views.sort()
                del views[N_VIEWS:]
    return best, capo, sess


def _cell_feats(best, sess, capo, session, model, mean, std, progress, tag):
    """ci -> normalized embedding of a FIXED-PHYSICAL-SCALE crop (CROP_M square
    at the cell's depth, resized to 224). Dense-map sampling was measured at
    0.64-0.66 same-walk cosine: best views sit at different distances, so a
    stride-32 feature covered 12 cm of wall in one view and 60 cm in the other
    — scale mismatch, not misalignment. Fixed-metric crops normalize it away."""
    by_frame = {}
    for ci, views in best.items():
        for (d, name, px, py) in views:
            by_frame.setdefault(name, []).append((ci, px, py, d))
    name2key = {str(sess.images[k[0], k[1]]): k for k in sess.images.key_pairs()}
    ids, crops = [], []
    for j, (name, items) in enumerate(sorted(by_frame.items())):
        if j % 20 == 0:
            progress(f"[{tag}] crops {j}/{len(by_frame)} frames")
        bgr = cv2.imread(str(capo.data_path(session) / name))
        k = name2key.get(name)
        if bgr is None or k is None:
            continue
        cam = sess.sensors[k[1]]
        fx = float(getattr(cam, "f", [cam.params[0]])[0])
        H, W = bgr.shape[:2]
        for ci, px, py, d in items:
            half = max(24, int(fx * CROP_M / max(d, 0.3) / 2))
            x0, x1 = int(px) - half, int(px) + half
            y0, y1 = int(py) - half, int(py) + half
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue                              # partial crops compare unfairly
            c = bgr[y0:y1, x0:x1]
            g = cv2.cvtColor(cv2.resize(c, (224, 224)), cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(g, cv2.CV_64F).var() < 12:
                continue                              # motion blur fakes dissimilarity
            ids.append(ci)
            crops.append(c)
    progress(f"[{tag}] embedding {len(crops)} crops …")
    embs = _embed_crops(model, mean, std, crops)
    acc = {}
    for i, ci in enumerate(ids):
        acc.setdefault(ci, []).append(embs[i])
    out = {}
    for ci, es in acc.items():                       # multi-view average, renormalized
        m = np.mean(es, axis=0)
        out[ci] = m / (np.linalg.norm(m) + 1e-8)
    return out


def run(capture, states, workspace, progress=None, dry=False):
    """Full appearance check. Returns candidate dicts (same shape as
    cloud_diff_prototype candidates) for texture-change incidents, and writes
    <workspace>/texture/report.json."""
    say = progress or (lambda s: print("  ·", s, flush=True))
    fdir = Path(workspace) / "fields"
    sp, spo = fdir / "static_pre.npy", fdir / "static_post.npy"
    if not sp.exists():
        raise SystemExit("fields not computed — run compute change fields first")
    pre_ref, post_ref = states["pre"]["ref"], states["post"]["ref"]
    T = G._load_T(CD._bridge(capture, pre_ref, post_ref, None))  # pre <- post

    say("building both-static cells …")
    P_pre = np.load(sp).astype(np.float64)
    P_post = np.load(spo).astype(np.float64)
    P_post_in_pre = G._apply_T(T, P_post)
    k_pre = set(map(tuple, np.floor(P_pre / CELL).astype(np.int64)))
    k_post = set(map(tuple, np.floor(P_post_in_pre / CELL).astype(np.int64)))
    both = sorted(k_pre & k_post)
    centers_pre = (np.array(both, np.float64) + 0.5) * CELL      # pre frame
    centers_post = G._apply_T(T.inverse(), centers_pre)          # post frame
    say(f"{len(both):,} both-static cells")

    best_pre, capo_pre, sess_pre = _best_views(centers_pre, capture,
                                               states["pre"]["session"],
                                               pre_ref, FRAME_STEP, say, "pre")
    best_post, capo_post, sess_post = _best_views(centers_post, capture,
                                                  states["post"]["session"],
                                                  post_ref, FRAME_STEP, say, "post")
    common = sorted(set(best_pre) & set(best_post))
    say(f"{len(common):,} cells visible in BOTH walks")
    if not common:
        return []

    model, mean, std = _load_dino(say)
    f_pre = _cell_feats({c: best_pre[c] for c in common},
                        sess_pre, capo_pre, states["pre"]["session"],
                        model, mean, std, say, "pre")
    f_post = _cell_feats({c: best_post[c] for c in common},
                         sess_post, capo_post, states["post"]["session"],
                         model, mean, std, say, "post")
    cells = sorted(set(f_pre) & set(f_post))
    sims = np.array([float(np.dot(f_pre[c], f_post[c])) for c in cells])
    mu, sd = float(sims.mean()), float(sims.std())
    thr = min(mu - 3 * sd, float(np.percentile(sims, 1.0)))
    flagged = [c for c, s in zip(cells, sims) if s < thr]
    say(f"similarity mean {mu:.3f} ± {sd:.3f} -> threshold {thr:.3f}; "
        f"{len(flagged)} / {len(cells)} cells flagged")
    # ECHO suppression: a flagged cell adjacent to geometric-change evidence is
    # the changed OBJECT contaminating a static background's view (measured:
    # removed shelf/sign-stand re-flagged via the wall behind them) — the cloud
    # diff already owns those. Keep only appearance-ONLY flags.
    if flagged:
        from scipy.spatial import cKDTree
        ch = [np.load(fdir / "changed_pre.npy").astype(np.float64)]
        cpo = np.load(fdir / "changed_post.npy").astype(np.float64)
        ch.append(G._apply_T(T, cpo))                # post changes -> pre frame
        tree = cKDTree(np.vstack(ch))
        fc0 = centers_pre[np.array(flagged, int)]
        d_ch = tree.query(fc0, workers=-1)[0]
        kept = [f for f, d in zip(flagged, d_ch) if d > 0.30]
        say(f"echo suppression: {len(flagged) - len(kept)} cells adjacent to "
            f"geometric change dropped; {len(kept)} appearance-only remain")
        flagged = kept

    # incidents: spatial clustering of flagged cell centres (pre frame)
    import open3d as o3d
    cands = []
    report = {"cells": len(cells), "mean": mu, "std": sd, "thr": thr,
              "incidents": []}
    if flagged:
        # cell-key -> point indices, built ONCE (per-incident scans over the
        # multi-million-point fields would dominate the runtime)
        def index_by_cell(P):
            d = {}
            for i, q in enumerate(map(tuple, np.floor(P / CELL).astype(np.int64))):
                d.setdefault(q, []).append(i)
            return d
        idx_pre = index_by_cell(P_pre)
        idx_post = index_by_cell(P_post_in_pre)
        fc = centers_pre[np.array(flagged, int)]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fc))
        lab = np.asarray(pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=1))
        n_inc = 0
        for li in range(lab.max() + 1):
            cc = fc[lab == li]
            if len(cc) < MIN_CELLS:
                continue
            keys = set(map(tuple, np.floor(cc / CELL).astype(np.int64)))
            ip = sum((idx_pre.get(k, []) for k in keys), [])
            ipo = sum((idx_post.get(k, []) for k in keys), [])
            n_inc += 1
            for st, pts in (("pre", P_pre[ip]), ("post", P_post[ipo])):
                if len(pts) < 20:
                    continue
                cands.append({"id": f"tex_{n_inc:02d}", "state": st,
                              "change_type": "moved", "pts": pts,
                              "sig": CD._signature(pts)})
            report["incidents"].append({"id": f"tex_{n_inc:02d}", "cells": int(len(cc)),
                                        "centroid": [round(float(v), 2) for v in cc.mean(0)]})
    tdir = Path(workspace) / "texture"
    tdir.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(tdir / "report.json", "w"), indent=1)
    say(f"{len(report['incidents'])} texture incident(s) -> {tdir/'report.json'}")
    if dry or not cands:
        return cands if dry else []
    say("seeding incidents (occlusion-tested projection) …")
    ctx = CD._ctx_for_states(capture, states, cands)
    CD._seed(cands, states, ctx, 0, 5, 0.5,
             Path(capture) / "changes" / "cloud_diff" / "seed_dbg",
             occ_tol=0.05, min_frac=0.0)
    out = []
    for c in cands:
        if not c.get("seeds"):
            continue
        key = f"{c['id']}__{c['state']}"
        od = G.out_dir(capture, key)
        json.dump(c["seeds"], open(od / "seeds.json", "w"), indent=1)
        np.save(od / "cluster.npy", c["pts"].astype(np.float32))
        entry = {"id": c["id"], "label": "", "deformability": "rigid",
                 "state": c["state"], "frame": next(iter(c["seeds"])),
                 "points": [], "seed_frames": [], "source": "texture"}
        out.append((key, states[c["state"]]["session"], entry, c.get("n_frames", 0)))
    say(f"wrote seeds for {len(out)} texture proposals")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--pre-session", required=True)
    ap.add_argument("--pre-ref", required=True)
    ap.add_argument("--post-session", required=True)
    ap.add_argument("--post-ref", required=True)
    ap.add_argument("--dry", action="store_true", help="detect + report only, no seeding")
    a = ap.parse_args()
    states = {"pre": {"session": a.pre_session, "ref": a.pre_ref},
              "post": {"session": a.post_session, "ref": a.post_ref}}
    ws = G.out_dir(a.capture)
    res = run(a.capture, states, ws, dry=a.dry)
    for r in res:
        if isinstance(r, dict):
            print(f"  {r['id']}__{r['state']}: {r['sig']}")
        else:
            print(f"  {r[0]} ({r[3]} frames)")


if __name__ == "__main__":
    main()
