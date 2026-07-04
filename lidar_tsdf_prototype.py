"""Symmetric change GT from TSDF-fused mono depth anchored to raw NavVis lidar.

Successor experiment to mono_footprint_prototype.py, which showed that per-frame
affine-aligned DA3 depth is smooth per view but metrically inconsistent between
views (~2-5% systematic scale error pair-to-pair), so the voxel vote degenerates
to an inflated visual hull. Here the mono depth is made consistent BEFORE any
voting, and the original mesh-face pipeline of change_mask.py --symmetric is
reused unchanged on top:

  - per frame, DA3 depth (raw cache shared with mono_footprint_prototype) is
    scale/shift-fitted to the RAW lidar scan points z-buffered into the frame --
    unlike the NavVis mesh render, the raw cloud still contains the changed
    objects the meshing step dropped, so no mask exclusion is needed and far
    fewer frames lose their anchor;
  - the aligned depths of all confident frames are fused into ONE TSDF mesh per
    state -> cross-view consistent by construction, complete where the scan
    never looked (fusion averages the residual per-frame wobble);
  - change_mask.py's face voting + occlusion-tested cross-rendering then runs
    with the fused meshes substituted for the NavVis meshes. Output goes to
    changes/change_mask_symmetric_tsdf/ (originals untouched).

Stages (lamar_env + PYTHONPATH=~/repos/lamaria-indoor:., no GPU needed as long
as changes/mono_depth/<session>_w640/ raw DA3 caches are complete):

  # 1. sanity-check the lidar-anchored alignment:
  python lidar_tsdf_prototype.py check --capture .../cnb_e100 --frames 8
  #    -> changes/lidar_tsdf_check/<state>/*.jpg [RGB | lidar | DA3 aligned | rel err]

  # 2. fuse each state's aligned mono depth into a TSDF mesh:
  python lidar_tsdf_prototype.py fuse --capture .../cnb_e100
  #    -> changes/tsdf_mono/<state>_<ref>.ply (+ render sanity panels)

  # 3. symmetric GT with the fused meshes in place of the NavVis meshes:
  python lidar_tsdf_prototype.py symmetric --capture .../cnb_e100 --viz
  #    -> changes/change_mask_symmetric_tsdf/<state>/*.png (+ _viz)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import change_mask as CM
import geom_sam_prototype as G
import mono_footprint_prototype as M

WORK_SCALE = M.WORK_SCALE
MIN_INLIER = M.MIN_INLIER
TSDF_VOXEL = 0.015
TSDF_TRUNC = 0.06
DEPTH_TRUNC = 8.0


class LidarAnchoredDepth:
    """Per-frame DA3 depth anchored to the state's raw lidar scan.

    Reuses the raw DA3 npz cache written by mono_footprint_prototype
    (changes/mono_depth/<session>_w<W>/); lidar-based (s, b) fits live next to
    it in fits_lidar.json. Same low-confidence handling as AlignedDepth:
    inlier_frac < min_inlier -> s,b interpolated over time from neighbors."""

    def __init__(self, capture, sess, session_id, ref, scale=WORK_SCALE,
                 min_inlier=MIN_INLIER, verbose=True):
        self.capture, self.sess, self.ref = Path(capture), sess, ref
        self.session_id, self.scale = session_id, scale
        self.min_inlier, self.verbose = min_inlier, verbose
        self.cam_by_frame = {str(sess.images[k]): k for k in sess.images.key_pairs()}
        cam0 = sess.sensors[next(iter(self.cam_by_frame.values()))[1]]
        self.cam, _, _ = G._scaled_camera(cam0, scale)
        self.dir = (self.capture / "changes" / "mono_depth"
                    / f"{session_id}_w{self.cam.width}")
        if not self.dir.exists():
            raise SystemExit(f"missing raw DA3 cache {self.dir} -- run "
                             f"mono_footprint_prototype.py check/symmetric first")
        self.fits_path = self.dir / "fits_lidar.json"
        self.fits = (json.load(open(self.fits_path))
                     if self.fits_path.exists() else {})
        self._P = None                     # lazy lidar cloud (N,3) float32
        self._sb = None                    # frame -> resolved (s, b)

    @property
    def cloud(self):
        if self._P is None:
            import open3d as o3d
            p = self.capture / "sessions" / self.ref / "raw_data" / "pointcloud.ply"
            if self.verbose:
                print(f"    [{self.session_id}] loading lidar {p} ...", flush=True)
            self._P = np.asarray(o3d.io.read_point_cloud(str(p)).points,
                                 dtype=np.float32)
        return self._P

    def lidar_depth(self, frame, splat=1):
        """Raw scan z-buffered into the frame (0 where no point). `splat` is the
        radius (px) of the min-filter used to densify single-pixel gaps."""
        T = self.pose(frame)
        R = np.asarray(T.R, np.float32)
        t = np.asarray(T.t, np.float32).reshape(3)
        pc = (self.cloud - t) @ R
        z = pc[:, 2]
        fx, fy, cx, cy = M._K(self.cam)
        W, H = self.cam.width, self.cam.height
        m = z > 0.2
        u = np.rint(fx * pc[m, 0] / z[m] + cx - 0.5).astype(np.int64)
        v = np.rint(fy * pc[m, 1] / z[m] + cy - 0.5).astype(np.int64)
        zi = z[m]
        ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        D = np.full(H * W, np.inf, np.float32)
        order = np.argsort(zi[ok])[::-1]           # nearest point written last
        D[(v[ok] * W + u[ok])[order]] = zi[ok][order]
        D = D.reshape(H, W)
        if splat:
            k = 2 * splat + 1
            D = cv2.erode(D, np.ones((k, k), np.uint8))     # local min fills gaps
        D[~np.isfinite(D)] = 0
        return D

    def raw(self, frame):
        p = self.dir / Path(CM._flat_png(frame)).with_suffix(".npz").name
        if not p.exists():
            raise SystemExit(f"missing cached DA3 depth {p}")
        return np.load(p)["depth"].astype(np.float32)

    def pose(self, frame):
        ts, cam = self.cam_by_frame[frame]
        return self.sess.get_pose(ts, cam)

    def prepare(self):
        """Fit (s, b) against lidar for every frame, then resolve low-confidence
        fits by temporal interpolation from confident neighbors."""
        frames = sorted(self.cam_by_frame, key=lambda f: self.cam_by_frame[f][0])
        todo = [f for f in frames if f not in self.fits]
        for i, f in enumerate(todo):
            _, _, st = M._fit_scale_shift(self.raw(f), self.lidar_depth(f))
            self.fits[f] = st
            if (i + 1) % 25 == 0:
                json.dump(self.fits, open(self.fits_path, "w"))
                if self.verbose:
                    print(f"    [{self.session_id}] lidar fits {i + 1}/{len(todo)}",
                          flush=True)
        if todo:
            json.dump(self.fits, open(self.fits_path, "w"))
        ts = np.array([float(self.cam_by_frame[f][0]) for f in frames])
        ok = np.array([self.fits[f].get("inlier_frac", 0) >= self.min_inlier
                       for f in frames])
        if not ok.any():
            self._sb = {f: (1.0, 0.0) for f in frames}
            print(f"    [{self.session_id}] WARNING: no confident lidar fits",
                  flush=True)
            return
        s = np.interp(ts, ts[ok], [self.fits[f]["scale"] for f in np.array(frames)[ok]])
        b = np.interp(ts, ts[ok], [self.fits[f]["shift"] for f in np.array(frames)[ok]])
        self._sb = {f: (float(s[i]), float(b[i])) for i, f in enumerate(frames)}
        if self.verbose:
            print(f"    [{self.session_id}] lidar scale fits: {ok.sum()}/{len(frames)}"
                  f" confident, {(~ok).sum()} interpolated", flush=True)

    def aligned(self, frame):
        if self._sb is None:
            self.prepare()
        s, b = self._sb[frame]
        return s * self.raw(frame) + b

    def confident(self, frame):
        return self.fits[frame].get("inlier_frac", 0) >= self.min_inlier


def _mk_lad(capture, st, verbose=True):
    _, sess = G._session(capture, st["session"], st["ref"])
    return LidarAnchoredDepth(capture, sess, st["session"], st["ref"],
                              verbose=verbose)


def _fused_mesh_path(capture, st):
    return (Path(capture) / "changes" / "tsdf_mono"
            / f"{st['state']}_{st['ref']}.ply")


# ──────────────────────────────── stages ────────────────────────────────
def cmd_check(args):
    """Lidar-anchored alignment panels, plus comparison with the mesh fits."""
    for st in _states(args).values():
        lad = _mk_lad(args.capture, st)
        frames = sorted(lad.cam_by_frame, key=lambda f: lad.cam_by_frame[f][0])
        picks = [frames[i] for i in
                 np.linspace(0, len(frames) - 1, args.frames, dtype=int)]
        odir = Path(args.capture) / "changes" / "lidar_tsdf_check" / st["state"]
        odir.mkdir(parents=True, exist_ok=True)
        mesh_fits_p = lad.dir / "fits.json"
        mesh_fits = json.load(open(mesh_fits_p)) if mesh_fits_p.exists() else {}
        for f in picks:
            raw, lid = lad.raw(f), lad.lidar_depth(f)
            s, b, stt = M._fit_scale_shift(raw, lid)
            lad.fits[f] = stt
            aligned = (stt.get("scale", 1.0) * raw + stt.get("shift", 0.0)
                       if s is not None else raw)
            rgb = cv2.imread(str(CM._rgb_path(args.capture, st["session"], f)))
            rgb = cv2.resize(rgb, (lad.cam.width, lad.cam.height))
            v = lid > 0
            lo, hi = np.percentile(lid[v], (2, 98)) if v.any() else (0, 1)
            err = np.zeros_like(aligned)
            err[v] = np.abs(aligned[v] - lid[v]) / lid[v]
            panel = np.hstack([rgb, M._turbo(lid, lo, hi), M._turbo(aligned, lo, hi),
                               M._turbo(np.where(v, err, -1), 0, 0.2)])
            mf = mesh_fits.get(f, {})
            txt = (f"lidar: s={stt.get('scale', 0):.3f} b={stt.get('shift', 0):+.2f}m"
                   f" inl={stt.get('inlier_frac', 0):.0%}"
                   f" medrel={stt.get('med_rel_err', 0):.1%}"
                   + (f"   (mesh fit: inl={mf.get('inlier_frac', 0):.0%}"
                      f" medrel={mf.get('med_rel_err', 0):.1%})" if mf else ""))
            cv2.putText(panel, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            cv2.imwrite(str(odir / Path(CM._flat_png(f)).with_suffix(".jpg").name),
                        panel)
            print(f"  [{st['state']}] {f}: {txt}")
        json.dump(lad.fits, open(lad.fits_path, "w"))
        print(f"  panels -> {odir}   [RGB | lidar | DA3 aligned | rel err 0-20%]")


def cmd_fuse(args):
    """TSDF-fuse each state's lidar-aligned mono depth into one mesh."""
    import open3d as o3d
    for st in _states(args).values():
        lad = _mk_lad(args.capture, st)
        lad.prepare()
        frames = sorted(lad.cam_by_frame, key=lambda f: lad.cam_by_frame[f][0])
        use = [f for f in frames if lad.confident(f)]
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=args.voxel, sdf_trunc=args.trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        intr = o3d.camera.PinholeCameraIntrinsic(
            lad.cam.width, lad.cam.height, *M._K(lad.cam))
        for j, f in enumerate(use):
            rgb = cv2.imread(str(CM._rgb_path(args.capture, st["session"], f)))
            rgb = cv2.cvtColor(cv2.resize(rgb, (lad.cam.width, lad.cam.height)),
                               cv2.COLOR_BGR2RGB)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)),
                o3d.geometry.Image(lad.aligned(f)),
                depth_scale=1.0, depth_trunc=DEPTH_TRUNC,
                convert_rgb_to_intensity=False)
            T = lad.pose(f)
            E = np.eye(4)
            R = np.asarray(T.R, np.float64)
            t = np.asarray(T.t, np.float64).reshape(3)
            E[:3, :3], E[:3, 3] = R.T, -R.T @ t          # world -> cam
            vol.integrate(rgbd, intr, E)
            if (j + 1) % 50 == 0:
                print(f"    [{st['state']}] integrated {j + 1}/{len(use)}",
                      flush=True)
        mesh = vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        out = _fused_mesh_path(args.capture, st)
        out.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_triangle_mesh(str(out), mesh)
        print(f"  [{st['state']}] fused {len(use)}/{len(frames)} frames -> {out} "
              f"({len(mesh.triangles)} tris)")

        # sanity: render the fused mesh back into a few frames vs aligned DA3
        from scantools.proc.rendering import Renderer
        rd = Renderer(mesh)
        odir = Path(args.capture) / "changes" / "lidar_tsdf_check" / f"{st['state']}_fused"
        odir.mkdir(parents=True, exist_ok=True)
        for f in [use[i] for i in np.linspace(0, len(use) - 1, 4, dtype=int)]:
            _, dr = rd.render_from_capture(lad.pose(f), lad.cam)
            da = lad.aligned(f)
            rgb = cv2.imread(str(CM._rgb_path(args.capture, st["session"], f)))
            rgb = cv2.resize(rgb, (lad.cam.width, lad.cam.height))
            v = dr > 0
            lo, hi = np.percentile(dr[v], (2, 98)) if v.any() else (0, 1)
            err = np.zeros_like(da)
            err[v] = np.abs(da[v] - dr[v]) / dr[v]
            panel = np.hstack([rgb, M._turbo(dr, lo, hi), M._turbo(da, lo, hi),
                               M._turbo(np.where(v, err, -1), 0, 0.2)])
            cv2.putText(panel, f"fused mesh render vs DA3: med rel "
                        f"{np.median(err[v]):.1%}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imwrite(str(odir / Path(CM._flat_png(f)).with_suffix(".jpg").name),
                        panel)
        print(f"  [{st['state']}] fused-render panels -> {odir}")


def cmd_symmetric(args):
    """change_mask.py --symmetric semantics with the fused meshes swapped in.
    Writes changes/change_mask_symmetric_tsdf/ (never the original output)."""
    import scantools.utils.io as sio
    capture = Path(args.capture)
    states = _states(args)
    override = {}
    for st in states.values():
        capo, _ = G._session(capture, st["session"], st["ref"])
        navvis = capo.proc_path(st["ref"]) / capo.sessions[st["ref"]].proc.meshes["mesh"]
        fused = _fused_mesh_path(capture, st)
        if not fused.exists():
            raise SystemExit(f"missing fused mesh {fused} -- run `fuse` first")
        override[str(navvis)] = fused
    orig_read = sio.read_mesh
    sio.read_mesh = lambda p, *a, **k: orig_read(override.get(str(p), p), *a, **k)
    try:
        root = capture / args.geom_out
        keys = sorted(p.name for p in root.iterdir() if p.is_dir() and "__" in p.name)
        unions, prov = CM._native_unions(capture, args.geom_out, keys, True)
        native_snap = {s: dict(fr) for s, fr in unions.items()}
        bridge = args.bridge or str(capture / "changes"
                                    / f"{args.post_ref}_to_{args.pre_ref}"
                                    / f"T_{args.pre_ref}_from_{args.post_ref}.txt")
        T = G._load_T(bridge)
        params = dict(vote_scale=args.vote_scale, render_scale=args.render_scale,
                      min_views=args.min_views, ratio=args.ratio,
                      occ_tol=args.occ_tol, verbose=True,
                      dbg_dir=capture / "changes" / "change_mask_symmetric_tsdf")
        pre, post = states["pre"], states["post"]
        ghosts = {"post": CM._ghost_for_direction(capture, pre, post, T.inverse(),
                                                  args.geom_out, keys, **params),
                  "pre": CM._ghost_for_direction(capture, post, pre, T,
                                                 args.geom_out, keys, **params)}
    finally:
        sio.read_mesh = orig_read
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
    block = CM._write_masks(capture, "change_mask_symmetric_tsdf", unions, prov)
    if args.viz:
        split = {s: {f: (native_snap.get(s, {}).get(f), ghosts.get(s, {}).get(f))
                     for f in fr} for s, fr in unions.items()}
        CM._write_viz(capture, "change_mask_symmetric_tsdf", unions, prov, split, True)
    for state, frames in block.items():
        print(f"  {state}: {len(frames)} symmetric change frames")


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

    c = sub.add_parser("check", parents=[common],
                       help="lidar-anchored alignment panels")
    c.add_argument("--frames", type=int, default=8)
    c.set_defaults(fn=cmd_check)

    f = sub.add_parser("fuse", parents=[common],
                       help="TSDF-fuse aligned mono depth -> changes/tsdf_mono/")
    f.add_argument("--voxel", type=float, default=TSDF_VOXEL)
    f.add_argument("--trunc", type=float, default=TSDF_TRUNC)
    f.set_defaults(fn=cmd_fuse)

    s = sub.add_parser("symmetric", parents=[common],
                       help="symmetric GT with fused meshes -> "
                            "change_mask_symmetric_tsdf/")
    s.add_argument("--bridge", default=None)
    s.add_argument("--vote-scale", type=float, default=0.5)
    s.add_argument("--render-scale", type=float, default=1.0)
    s.add_argument("--min-views", type=int, default=3)
    s.add_argument("--ratio", type=float, default=0.6)
    s.add_argument("--occ-tol", type=float, default=0.10)
    s.add_argument("--viz", action="store_true")
    s.set_defaults(fn=cmd_symmetric)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
