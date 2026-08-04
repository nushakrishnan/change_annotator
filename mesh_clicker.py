"""3D click annotator: select changed objects directly in the point cloud.

For hard pairs (polymesse: massive change, far covisibility) image-space
annotation breaks down -- you cannot tell WHAT changed by scrubbing frames.
Here you annotate in 3D instead, where the change is obvious at a glance:

  view    : open a state's geometry in an Open3D window -- the NavVis mesh
            (--mesh, nicest to look at; polymesse's meshes verifiably retain
            the changed objects) or the raw cloud (default; always contains
            everything, use it on captures whose meshing drops objects).
            Ctrl+click points on an object; the selection region-grows live
            from your clicks. Save writes the object's 3D points into the
            pair workspace.
  project : for every saved 3D object, reproject its points into its state's
            Aria frames (occlusion-tested against the NavVis mesh, the same
            geom_sam_prototype._seeds_for_session used everywhere else),
            write per-frame seeds + a GUI object entry, and optionally run
            SAM per-frame masks (--sam). The object then shows up in the GUI
            exactly like a cloud-diff proposal, ready for review.

SceneDiff-style GT (every change annotated in BOTH sequences) falls out of
the naming convention plus the existing point_ghost machinery:
  moved   : click it in BOTH states with the SAME name -> two native
            annotations, one per sequence, linked by shared id;
  removed / added : click it in the one state it exists in. Each save also
            writes the object's surface (NavVis submesh in --mesh mode,
            alpha-shape at project time otherwise) as <key>/mesh.ply -- the
            exact input point_ghost_prototype.py `ghosts` consumes to render
            the occlusion-tested footprint into the OTHER sequence as a
            GUI-editable ghost object, and `merge` folds native + ghost masks
            into changes/change_mask_symmetric_points/ (mv3dcd-style GT).

Because every clip pair of a capture shares the same two NavVis states, one
3D annotation pass serves ALL pairs: rerun `project` per pair workspace.

view keys:
  Ctrl+click  add a click on the surface (selection re-grows immediately)
  U           undo last click          +/-   grow reach (default 1.2 m)
  P           toggle plane lock (floor/wall points refuse to grow; ON default)
  S           save object -> workspace     N   next object (discards unsaved)
  Q           quit

Dev harness:
  PYTHONPATH=~/repos/lamaria-indoor GEOM_OUT=changes/geom_sam_out_polymesse_p00 \\
    ~/annotator_env/bin/python mesh_clicker.py view \\
      --capture /media/lamaria_indoor/captures/changes/polymesse/pairs \\
      --ref navvis_1 --state pre --crop-session hg_poly1_4_rgb_p00
  ... then:
    ~/annotator_env/bin/python mesh_clicker.py project \\
      --capture .../pairs --pre-session hg_poly1_4_rgb_p00 --pre-ref navvis_1 \\
      --post-session hg_poly2_4_rgb_p00 --post-ref navvis_2 --sam
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

import geom_sam_prototype as G


# ───────────────────────────── cloud / geometry ─────────────────────────────
def _load_state_cloud(capture, ref, voxel, crop_center=None, crop_margin=6.0):
    """Raw NavVis cloud of `ref`, voxel-downsampled, optionally cropped to an
    axis-aligned box around `crop_center` points (e.g. a clip's camera path)."""
    import open3d as o3d
    p = Path(capture) / "sessions" / ref / "raw_data" / "pointcloud.ply"
    print(f"  [{ref}] loading {p} ...", flush=True)
    pcd = o3d.io.read_point_cloud(str(p))
    if voxel:
        pcd = pcd.voxel_down_sample(voxel)
    if crop_center is not None and len(crop_center):
        lo = crop_center.min(0) - crop_margin
        hi = crop_center.max(0) + crop_margin
        pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    print(f"  [{ref}] {len(pcd.points):,} points @ voxel {voxel} m", flush=True)
    return pcd


def _load_state_mesh(capture, ref, crop_center=None, crop_margin=6.0):
    """NavVis mesh of `ref`, optionally cropped like _load_state_cloud. Returns
    (mesh, vertices) -- clicking/growing runs on the vertex set, which at NavVis
    density is as dense as a ~1-2 cm cloud."""
    import open3d as o3d
    p = Path(capture) / "sessions" / ref / "proc" / "meshes" / "mesh.ply"
    print(f"  [{ref}] loading mesh {p} (minutes) ...", flush=True)
    mesh = o3d.io.read_triangle_mesh(str(p))
    if crop_center is not None and len(crop_center):
        lo = crop_center.min(0) - crop_margin
        hi = crop_center.max(0) + crop_margin
        mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    print(f"  [{ref}] {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles", flush=True)
    return mesh, np.asarray(mesh.vertices)


def _camera_centers(capture, sid, ref):
    """Camera centers of a clip session (world/ref frame) -- used to crop the
    cloud to the region the pair actually covers."""
    capo, sess = G._session(capture, sid, ref)
    keys = [k for k in sess.images.key_pairs() if "cam0" in str(k[1])]
    return np.array([sess.get_pose(ts, cam).t for ts, cam in keys])


def _plane_lock(P, voxel, n_planes=4, min_inliers=60_000, dist=0.03):
    """Bool mask of points on the dominant planes (floor/ceiling/big walls).
    Locked points refuse to region-grow, so a click on a chair does not flood
    the floor. RANSAC runs on a coarse proxy; inliers transfer by distance."""
    import open3d as o3d
    from scipy.spatial import cKDTree
    proxy = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    proxy = proxy.voxel_down_sample(max(0.05, 2 * voxel))
    lock = np.zeros(len(P), bool)
    for _ in range(n_planes):
        if len(proxy.points) < min_inliers:
            break
        model, inl = proxy.segment_plane(dist, 3, 200)
        if len(inl) < min_inliers:
            break
        a, b, c, d = model
        lock |= np.abs(P @ np.array([a, b, c]) + d) < dist
        proxy = proxy.select_by_index(inl, invert=True)
    return lock


def _grow(P, tree, lock, click_idx, reach=1.2, r_grow=None, cap=400_000, voxel=0.03):
    """Region-grow a selection from clicked points: BFS over an r_grow-radius
    neighbour graph, never entering plane-locked points (a click ON a locked
    point still takes a small local patch so planar objects stay clickable),
    never straying more than `reach` from the nearest click."""
    from scipy.spatial import cKDTree
    if not click_idx:
        return np.zeros(0, np.int64)
    r_grow = r_grow or max(2.5 * voxel, 0.06)
    clicks = P[click_idx]
    near_click = tree.query_ball_point(clicks, reach)
    allowed = np.zeros(len(P), bool)
    for idx in near_click:
        allowed[idx] = True
    allowed &= ~lock
    seen = np.zeros(len(P), bool)
    frontier = []
    for ci in click_idx:
        if lock[ci]:  # planar object: take a local patch, do not spread on the plane
            patch = tree.query_ball_point(P[ci], 0.12)
            seen[patch] = True
        else:
            frontier.append(ci)
    frontier = [i for i in frontier if not seen[i]]
    seen[frontier] = True
    while frontier and seen.sum() < cap:
        nbrs = tree.query_ball_point(P[frontier], r_grow)
        nxt = np.unique(np.concatenate([np.asarray(n, np.int64) for n in nbrs]))
        nxt = nxt[allowed[nxt] & ~seen[nxt]]
        seen[nxt] = True
        frontier = nxt.tolist()
    return np.where(seen)[0]


def _next_id(state):
    """First free mc_<state>_NN id in the workspace."""
    taken = set()
    ws = _ws_root()
    if ws.exists():
        for d in ws.iterdir():
            if d.name.startswith(f"mc_{state}_"):
                taken.add(d.name.split("__")[0])
    i = 1
    while f"mc_{state}_{i:02d}" in taken:
        i += 1
    return f"mc_{state}_{i:02d}"


_CAPTURE = None  # set by cmd_* so workspace helpers can resolve out_dir


def _ws_root():
    return G.out_dir(_CAPTURE)


# ───────────────────────────── view (GUI) ─────────────────────────────
class Clicker:
    def __init__(self, geom, P, ref, state, voxel, is_mesh=False, faces=None):
        import open3d as o3d
        import open3d.visualization.gui as gui
        import open3d.visualization.rendering as rendering
        from scipy.spatial import cKDTree
        self.o3d, self.gui, self.rendering = o3d, gui, rendering

        self.P = P                       # clickable/growable point set
        self.tree = cKDTree(self.P)
        print("  fitting dominant planes (grow lock) ...", flush=True)
        self.lock = _plane_lock(self.P, voxel)
        self.voxel, self.ref, self.state = voxel, ref, state
        self.faces = faces
        self.use_lock, self.reach = True, 1.2
        self.clicks, self.sel = [], np.zeros(0, np.int64)
        self.obj_id = _next_id(state)

        app = gui.Application.instance
        app.initialize()
        self.window = app.create_window(f"mesh clicker — {ref} ({state})", 1600, 1000)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.widget.scene.set_background([0.09, 0.09, 0.11, 1.0])
        mat = rendering.MaterialRecord()
        if is_mesh:
            mat.shader = "defaultLit"
            if not geom.has_vertex_normals():
                geom.compute_vertex_normals()
        else:
            mat.shader = "defaultUnlit"
            mat.point_size = 2.0
            if not geom.has_colors():
                z = self.P[:, 2]
                t = (z - z.min()) / max(z.ptp(), 1e-6)
                geom.colors = o3d.utility.Vector3dVector(
                    np.stack([0.35 + 0.3 * t, 0.35 + 0.3 * t, 0.4 + 0.3 * t], 1))
        self.widget.scene.add_geometry("cloud", geom, mat)
        b = geom.get_axis_aligned_bounding_box()
        self.widget.setup_camera(60, b, b.get_center())
        self.window.add_child(self.widget)

        panel = gui.Vert(0, gui.Margins(8, 8, 8, 8))
        self.status = gui.Label("")
        panel.add_child(self.status)
        panel.add_child(gui.Label("name (same name in both states = moved):"))
        self.name_edit = gui.TextEdit()
        panel.add_child(self.name_edit)
        panel.add_child(gui.Label(
            "ctrl+click add point   U undo   +/- reach\n"
            "P plane-lock   S save   N next   Q quit\n"
            "(click into the 3D view after typing a name)"))
        self.panel = panel
        self.window.add_child(panel)
        self.window.set_on_layout(self._layout)
        self.widget.set_on_mouse(self._mouse)
        self.widget.set_on_key(self._key)
        self._refresh()

    def _layout(self, ctx):
        r = self.window.content_rect
        self.widget.frame = r
        pref = self.panel.calc_preferred_size(ctx, self.gui.Widget.Constraints())
        self.panel.frame = self.gui.Rect(r.x + 8, r.y + 8, pref.width, pref.height)

    # ── selection display ──
    def _refresh(self):
        gui, o3d, rendering = self.gui, self.o3d, self.rendering
        for name in ("sel", "clicks"):
            if self.widget.scene.has_geometry(name):
                self.widget.scene.remove_geometry(name)
        if len(self.sel):
            sp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.P[self.sel]))
            sp.paint_uniform_color([0.95, 0.15, 0.15])
            m = rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = 4.0
            self.widget.scene.add_geometry("sel", sp, m)
        if self.clicks:
            cp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.P[self.clicks]))
            cp.paint_uniform_color([1.0, 0.9, 0.1])
            m = rendering.MaterialRecord(); m.shader = "defaultUnlit"; m.point_size = 10.0
            self.widget.scene.add_geometry("clicks", cp, m)
        self.status.text = (f"object {self.obj_id} [{self.state}]   "
                            f"clicks {len(self.clicks)}   selected {len(self.sel):,}\n"
                            f"reach {self.reach:.1f} m   plane-lock "
                            f"{'ON' if self.use_lock else 'OFF'}")
        self.window.post_redraw()

    def _regrow(self):
        lock = self.lock if self.use_lock else np.zeros(len(self.P), bool)
        self.sel = _grow(self.P, self.tree, lock, self.clicks,
                         reach=self.reach, voxel=self.voxel)
        self._refresh()

    # ── events ──
    def _mouse(self, ev):
        gui = self.gui
        if (ev.type == gui.MouseEvent.Type.BUTTON_DOWN
                and ev.is_modifier_down(gui.KeyModifier.CTRL)):
            frame = self.widget.frame

            def depth_cb(img):
                x, y = ev.x - frame.x, ev.y - frame.y
                d = np.asarray(img)[y, x]
                if d >= 1.0:                       # clicked empty space
                    return
                w = self.widget.scene.camera.unproject(
                    x, y, d, frame.width, frame.height)
                dist, idx = self.tree.query(np.asarray(w))
                if dist > 0.15:
                    return
                def apply():
                    self.clicks.append(int(idx))
                    self._regrow()
                gui.Application.instance.post_to_main_thread(self.window, apply)

            self.widget.scene.scene.render_to_depth_image(depth_cb)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def _key(self, ev):
        gui = self.gui
        if ev.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        k = ev.key
        if k == ord('u') and self.clicks:
            self.clicks.pop(); self._regrow()
        elif k in (ord('+'), ord('=')):
            self.reach = min(4.0, self.reach + 0.2); self._regrow()
        elif k == ord('-'):
            self.reach = max(0.2, self.reach - 0.2); self._regrow()
        elif k == ord('p'):
            self.use_lock = not self.use_lock; self._regrow()
        elif k == ord('s'):
            self._save()
        elif k == ord('n'):
            self.clicks, self.sel = [], np.zeros(0, np.int64)
            self.obj_id = _next_id(self.state); self._refresh()
        elif k == ord('q'):
            gui.Application.instance.quit()
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    def _save(self):
        import re
        import open3d as o3d
        if not len(self.sel):
            return
        name = re.sub(r"[^a-z0-9_]+", "_",
                      self.name_edit.text_value.strip().lower()).strip("_")
        oid = name or self.obj_id
        key = f"{oid}__{self.state}"
        od = G.out_dir(_CAPTURE, key)
        o3d.io.write_point_cloud(str(od / "points3d.ply"),
            o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.P[self.sel])))
        json.dump({"id": oid, "name": name, "state": self.state, "ref": self.ref,
                   "voxel": self.voxel, "reach": self.reach,
                   "clicks": self.P[self.clicks].tolist()},
                  open(od / "clicks.json", "w"), indent=1)
        n_faces = 0
        if self.faces is not None:                # NavVis submesh -> ghost input
            import point_ghost_prototype as PG
            selm = np.zeros(len(self.P), bool)
            selm[self.sel] = True
            keep = selm[self.faces].all(1)
            if keep.any():
                sub = PG._submesh(self.P, self.faces, keep)
                o3d.io.write_triangle_mesh(str(od / "mesh.ply"), sub)
                n_faces = int(keep.sum())
        print(f"  saved {key}: {len(self.sel):,} pts"
              + (f", {n_faces:,}-face submesh" if n_faces else "")
              + f" -> {od}", flush=True)
        self.clicks, self.sel = [], np.zeros(0, np.int64)
        self.obj_id = _next_id(self.state)
        self.name_edit.text_value = ""
        self._refresh()


def cmd_view(a):
    global _CAPTURE
    _CAPTURE = a.capture
    crop = _camera_centers(a.capture, a.crop_session, a.ref) \
        if a.crop_session else None
    if a.mesh:
        geom, P = _load_state_mesh(a.capture, a.ref, crop, a.crop_margin)
        c = Clicker(geom, P, a.ref, a.state, voxel=0.02, is_mesh=True,
                    faces=np.asarray(geom.triangles))
    else:
        pcd = _load_state_cloud(a.capture, a.ref, a.voxel, crop, a.crop_margin)
        c = Clicker(pcd, np.asarray(pcd.points), a.ref, a.state, a.voxel)
    c.gui.Application.instance.run()


# ───────────────────────────── project ─────────────────────────────
def cmd_project(a):
    """3D objects -> per-frame seeds (+ optional SAM masks) in their native
    state's clip session; registers each in the workspace gui_objects.json."""
    global _CAPTURE
    _CAPTURE = a.capture
    from cloud_diff_prototype import _renderer
    states = {"pre": (a.pre_session, a.pre_ref), "post": (a.post_session, a.post_ref)}

    objs = []
    for d in sorted(_ws_root().iterdir()):
        cj = d / "clicks.json"
        if cj.exists() and (a.obj in (None, d.name.split("__")[0])):
            objs.append((d, json.load(open(cj))))
    if not objs:
        print("no clicked objects in workspace"); return

    ctx = {}   # state -> (capo, sess, renderer)
    gp = _ws_root() / "gui_objects.json"
    gobj = json.load(open(gp)) if gp.exists() else {}
    import open3d as o3d
    for d, meta in objs:
        st = meta["state"]
        sid, ref = states[st]
        if st not in ctx:
            capo, sess = G._session(a.capture, sid, ref)
            ctx[st] = (capo, sess, _renderer(a.capture, ref))
        capo, sess, ren = ctx[st]
        pts = np.asarray(o3d.io.read_point_cloud(str(d / "points3d.ply")).points)
        if not (d / "mesh.ply").exists():         # cloud-mode object: alpha-shape
            import point_ghost_prototype as PG    # -> ghost/symmetric input
            alpha = max(PG.ALPHA, 2.5 * meta.get("voxel", 0.03))
            m = PG._reconstruct(pts, "alpha", alpha, 9, 0.3, d.name)
            if len(m.triangles):
                o3d.io.write_triangle_mesh(str(d / "mesh.ply"), m)
        print(f"  {d.name}: {len(pts):,} pts -> seeding {sid} ...", flush=True)
        seeds = G._seeds_for_session(pts, sess, ren, capo, sid, 0, a.min_vis,
                                     _ws_root() / "seed_dbg", occ_scale=a.occ_scale,
                                     write_dbg=False)
        if not seeds:
            print(f"    not visible in {sid} (0 frames) — skipped"); continue
        json.dump(seeds, open(d / "seeds.json", "w"), indent=1)
        gobj[d.name] = {"id": meta["id"], "label": meta.get("name", ""),
                        "deformability": "rigid", "state": st,
                        "frame": next(iter(seeds)), "points": [],
                        "seed_frames": [], "source": "mesh_click"}
        print(f"    {len(seeds)} frames seeded")
    json.dump(gobj, open(gp, "w"), indent=1)

    if a.sam:
        pair = G._load_sam_image()
        G._load_sam_image = lambda: pair
        for d, meta in objs:
            if not (d / "seeds.json").exists():
                continue
            sid = states[meta["state"]][0]
            print(f"  SAM masks: {d.name} ...", flush=True)
            G.cmd_perframe(argparse.Namespace(capture=a.capture, session=sid,
                                              obj=d.name, save_masks=True))
    print(f"\n  done — open the GUI on workspace {G.OUT} to review the native masks."
          f"\n  for SceneDiff-style masks in the OTHER sequence (removed/added"
          f"\n  footprints), run:  point_ghost_prototype.py ghosts  then  merge"
          f"\n  with the same GEOM_OUT/session args.")


# ───────────────────────────── CLI ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("view")
    v.add_argument("--capture", required=True)
    v.add_argument("--ref", required=True, help="NavVis state to click, e.g. navvis_1")
    v.add_argument("--state", required=True, choices=("pre", "post"))
    v.add_argument("--voxel", type=float, default=0.03)
    v.add_argument("--mesh", action="store_true",
                   help="click on the NavVis mesh instead of the raw cloud "
                        "(verify first that this capture's meshing keeps the "
                        "changed objects; polymesse does)")
    v.add_argument("--crop-session", default=None,
                   help="crop the cloud around this clip session's camera path")
    v.add_argument("--crop-margin", type=float, default=6.0)

    p = sub.add_parser("project")
    p.add_argument("--capture", required=True)
    p.add_argument("--pre-session", required=True)
    p.add_argument("--pre-ref", default="navvis_1")
    p.add_argument("--post-session", required=True)
    p.add_argument("--post-ref", default="navvis_2")
    p.add_argument("--obj", default=None, help="only this object id")
    p.add_argument("--min-vis", type=int, default=5)
    p.add_argument("--occ-scale", type=float, default=0.35)
    p.add_argument("--sam", action="store_true", help="also run SAM per-frame masks")

    a = ap.parse_args()
    {"view": cmd_view, "project": cmd_project}[a.cmd](a)


if __name__ == "__main__":
    main()
