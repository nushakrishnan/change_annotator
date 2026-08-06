"""Minimal click-to-propagate annotation GUI (RGB, single-camera).

A thin Flask front-end on top of the geometry-assisted propagation pipeline
(`geom_sam_prototype.py`). Runs in the single env (setup.sh) and holds one SAM3
*image* model for live click previews; the propagation step runs the geometry
`seeds` stage as a subprocess of the SAME interpreter (scantools on PYTHONPATH)
and finishes the per-frame masks in-process with the same model.

  click point(s) on a source frame  -> SAM3 image mask preview (live)
  add object (label / change_type)   -> first seed mask under geom_sam_out/<id>__<state>/
  + seed (on more spread-out frames) -> extra seed masks; unioned in 3D for a complete object
  propagate                          -> seeds (geometry, unions all seeds) + per-frame SAM masks
  export                             -> changes/segments.json   (annotation_spec.md §6)

Run (single env; scantools on PYTHONPATH for the seeds stage):
  PYTHONPATH=~/repos/lamaria-indoor ~/annotator_env/bin/python gui.py \
      --capture /media/lamaria_indoor/captures/changes/cnb_e100
  # open http://127.0.0.1:5000

Single-user by design (one shared model + a lock). Pinhole RGB sessions
(aria_*_rgb); frames live under the session's images/cam0/ folder.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file, Response

import change_mask as CM
import cloud_diff_prototype as CD
import geom_sam_prototype as G
import propagate_fix as PF

# One interpreter for every stage (defaults to the one running the GUI, i.e. the
# single env from setup.sh). The seeds stage is shelled out only to keep its
# heavy geometry imports out of the long-lived GUI process; it uses this same
# interpreter, with scantools (lamaria-indoor) on PYTHONPATH.
ANNOTATOR_PY = Path(os.environ.get("ANNOTATOR_PY", sys.executable))
LAMARIA_PYTHONPATH = os.environ.get(
    "LAMARIA_INDOOR",
    os.environ.get("LAMAR_PYTHONPATH", str(Path.home() / "repos/lamaria-indoor")))

app = Flask(__name__)

# ── runtime state (single user) ──
CFG = {}              # capture / states / seed params / scene / tier
MODEL = PROC = None
LOCK = threading.Lock()
PENDING = {}          # last click preview: {state, frame, points, neg, mask}
EDIT = {}             # pending review-mode re-segment: {id, frame, session, mask}
OBJECTS = {}          # id -> {label, change_type, deformability, state, frame, points}
JOBS = {}             # job_id -> {status, ...}   (JSON-serializable ONLY: api_job jsonifies it)
PROP_PENDING = {}     # job_id -> {frame_name: bool mask} awaiting human confirm (numpy, NOT in JOBS)
PROP_FLAGS = {}       # job_id -> {frame_name: {"lowconf","legacy"}} for the pending review UI
GEOM_CTX = {}         # state -> (capo, sess, renderer) cache for the geom-mask button
GEOM_KEYMAP = {}      # state -> {frame name: image key} (key_pairs scan is O(session))
GEOM_LOCK = threading.Lock()


def _png_b64(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _frame_path(state, name):
    s = CFG["states"][state]["session"]
    return Path(CFG["capture"]) / "sessions" / s / "raw_data" / name


def _save_working():
    """Atomic write via rename: the shared workspace file may be owned by another
    annotator (group-writable dir, not the file) -- os.replace only needs dir write."""
    p = G.out_dir(CFG["capture"]) / "gui_objects.json"
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    json.dump(OBJECTS, open(tmp, "w"), indent=1)
    os.replace(tmp, p)


def _objdir(key):
    """OBJECTS is keyed by '<id>__<state>' — exactly the geom_sam_out subdir name,
    so the key IS the objdir. One physical object (id) may have a 'pre' and a 'post'
    entry (a moved object); they're linked by id and merged at export."""
    return key if key in OBJECTS else None


def _change_type(in_pre, in_post):
    """Derived from presence per annotation_spec.md §4a — not annotator-set.
    in-both = a change ⇒ 'moved' (downstream mask-displacement refines moved vs static)."""
    if in_pre and not in_post:
        return "removed"
    if in_post and not in_pre:
        return "added"
    if in_pre and in_post:
        return "moved"
    return None


def _overlay_mask(bgr, mask):
    """Red 55%-opacity overlay of a boolean mask onto a BGR frame (in place copy)."""
    ov = bgr.copy()
    ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    return ov


def _mask_png_b64(mask):
    """Red+alpha PNG (transparent background) as a data URL — for the client mask canvas."""
    h, w = mask.shape
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[mask, 2] = 255  # red
    bgra[mask, 3] = 130  # alpha
    ok, buf = cv2.imencode(".png", bgra)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _decode_mask(data_url):
    """Decode a base64 RGBA PNG (from the client mask canvas) to a bool mask (alpha>0)."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[:, :, 3] > 0
    if arr.ndim == 3:
        return arr[:, :, 2] > 127
    return arr > 127


def _add_seed_mask(od, frame, mask, reset=False):
    """Append (or reset to) a seed mask for `frame` in the object's multi-seed store
    (src_masks/ + src_index.json), which cmd_seeds unions into one 3D object.
    Re-seeding the same frame replaces it. Returns the new seed count."""
    sd = od / "src_masks"
    sd.mkdir(exist_ok=True)
    flat = frame.replace("/", "_") + ".png"
    cv2.imwrite(str(sd / flat), (mask * 255).astype(np.uint8))
    idxp = od / "src_index.json"
    items = [] if reset or not idxp.exists() else json.load(open(idxp))
    items = [it for it in items if it["src_name"] != frame]
    items.append({"src_name": frame, "mask_file": f"src_masks/{flat}"})
    json.dump(items, open(idxp, "w"), indent=1)
    return len(items)


# ───────────────────────────── pages / static ─────────────────────────────
@app.route("/")
def index():
    return render_template("gui.html", scene=CFG["scene"],
                           seed_n=CFG["seed"]["n"], seed_min_vis=CFG["seed"]["min_vis"])


@app.route("/api/frames")
def api_frames():
    s = CFG["states"][request.args["state"]]["session"]
    d = Path(CFG["capture"]) / "sessions" / s / "raw_data" / "images" / "cam0"
    # numeric (timestamp) order — must match propagate_fix; lexical order diverges
    # on captures with mixed-width stems (e.g. billiards_*)
    return jsonify(sorted((f"images/cam0/{p.name}" for p in d.glob("*.jpg")),
                          key=lambda n: int(Path(n).stem)))


@app.route("/frame")
def frame():
    return send_file(_frame_path(request.args["state"], request.args["name"]))


@app.route("/results/<obj>/<path:fn>")
def results(obj, fn):
    return send_file(G.out_dir(CFG["capture"], obj) / fn)


# ───────────────────────────── click preview ─────────────────────────────
@app.route("/api/click", methods=["POST"])
def api_click():
    d = request.get_json()
    state, name = d["state"], d["frame"]
    pos = np.array(d.get("points", []), np.float32).reshape(-1, 2)
    neg = np.array(d.get("neg", []), np.float32).reshape(-1, 2)
    if len(pos) == 0:
        return jsonify(error="click a positive point first"), 400
    bgr = cv2.imread(str(_frame_path(state, name)))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts = np.concatenate([pos, neg], 0)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    with LOCK:
        masks, scores = G._segment(MODEL, PROC, img, points=pts, labels=labels, multimask=True)
    best = masks[int(np.argmax(scores))].astype(bool)
    PENDING.update(state=state, frame=name, points=pos.tolist(), neg=neg.tolist(), mask=best)
    ov = bgr.copy()
    ov[best] = (0.5 * ov[best] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    for x, y in pos:
        cv2.circle(ov, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y in neg:
        cv2.circle(ov, (int(x), int(y)), 6, (255, 0, 0), -1)
    return jsonify(overlay=_png_b64(ov), mask=_mask_png_b64(best), px=int(best.sum()), score=float(scores.max()))


@app.route("/api/add_object", methods=["POST"])
def api_add_object():
    d = request.get_json()
    oid = d["id"].strip()
    if not oid:
        return jsonify(error="object id required"), 400
    if d.get("mask"):                                   # client-painted (SAM + brush/eraser)
        mask = _decode_mask(d["mask"])
        state, frame, points = d["state"], d["frame"], d.get("points", [])
    elif PENDING.get("mask") is not None:
        mask, state = PENDING["mask"], PENDING["state"]
        frame, points = PENDING["frame"], PENDING["points"]
    else:
        return jsonify(error="no mask — click or draw the object first"), 400
    if not mask.any():
        return jsonify(error="mask is empty"), 400
    key = f"{oid}__{state}"  # same id in pre + post = a moved object (linked at export)
    od = G.out_dir(CFG["capture"], key)
    cv2.imwrite(str(od / "src_mask.png"), (mask * 255).astype(np.uint8))  # first seed (viz)
    _add_seed_mask(od, frame, mask, reset=True)                          # multi-seed store
    OBJECTS[key] = {"id": oid, "label": d.get("label") or oid,
                    "deformability": d.get("deformability", "rigid"),
                    "state": state, "frame": frame, "points": points,
                    "seed_frames": [frame]}
    PENDING.clear()
    _save_working()
    return jsonify(ok=True)


@app.route("/api/add_seed", methods=["POST"])
def api_add_seed():
    """Add another seed frame (a spread-out view) to an existing object. Lifting
    several masks and unioning their 3D points makes the propagation seed complete
    (one frame only sees one side of the object)."""
    d = request.get_json()
    oid = d["id"]
    if oid not in OBJECTS:
        return jsonify(error="add the object first"), 404
    if OBJECTS[oid].get("ghost"):
        return jsonify(error="ghosts take no seeds -- edit their masks in review mode"), 400
    if d.get("state") and d["state"] != OBJECTS[oid]["state"]:
        return jsonify(error=f"seed must be from the {OBJECTS[oid]['state']} state"), 400
    if d.get("mask"):
        mask, frame = _decode_mask(d["mask"]), d["frame"]
    elif PENDING.get("mask") is not None:
        mask, frame = PENDING["mask"], PENDING["frame"]
    else:
        return jsonify(error="no mask — click or draw the object first"), 400
    if not mask.any():
        return jsonify(error="mask is empty"), 400
    od = G.out_dir(CFG["capture"], oid)
    n = _add_seed_mask(od, frame, mask)
    sf = OBJECTS[oid].setdefault("seed_frames", [OBJECTS[oid].get("frame")])
    if frame not in sf:
        sf.append(frame)
    PENDING.clear()
    _save_working()
    return jsonify(ok=True, n_seeds=n)


# ───────────────────────────── propagate (job) ─────────────────────────────
@app.route("/api/propagate", methods=["POST"])
def api_propagate():
    oid = request.get_json()["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if OBJECTS[oid].get("ghost"):
        return jsonify(error="ghost masks are derived -- regenerate with "
                             "point_ghost_prototype.py ghosts"), 400
    if OBJECTS[oid].get("done"):
        return jsonify(error="object is marked done — uncheck done to re-propagate"), 400
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "obj": oid}
    threading.Thread(target=_propagate_job, args=(job_id, oid), daemon=True).start()
    return jsonify(job_id=job_id)


# ─────────────────────── cloud-diff proposer (job) ───────────────────────
@app.route("/api/cloud_diff", methods=["POST"])
def api_cloud_diff():
    """Kick off the cloud-diff change proposer: diff the two states' NavVis clouds,
    gate candidates by Aria visibility, and drop each survivor in as a reviewable
    object (per-cluster; link moved pairs by giving them a shared id). Background
    job — poll /api/job/<id>."""
    d = request.get_json() or {}
    _snapshot_workspace("predetect")                 # cheap hardlink backup, keeps last 10
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "msg": "starting …"}
    threading.Thread(target=_cloud_diff_job, args=(job_id, d), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/nuke_workspace", methods=["POST"])
def api_nuke_workspace():
    """Start this workspace over: EVERY object, mask, seed, cluster, frontier and
    snapshot is moved aside to <workspace>.nuked_<stamp> (same filesystem, atomic
    rename — recoverable by hand: `mv` it back). Requires confirm='proceed',
    enforced server-side too so nothing can trigger it programmatically."""
    d = request.get_json() or {}
    if (d.get("confirm") or "").strip().lower() != "proceed":
        return jsonify(error="type 'proceed' to confirm the nuke"), 400
    ws = G.out_dir(CFG["capture"])
    dst = ws.with_name(ws.name + f".nuked_{time.strftime('%Y%m%d_%H%M%S')}")
    if ws.exists():
        os.replace(ws, dst)
    ws.mkdir(parents=True, exist_ok=True)
    OBJECTS.clear()
    PROP_PENDING.clear()
    PROP_FLAGS.clear()
    _save_working()
    print(f"workspace NUKED -> {dst}", flush=True)
    return jsonify(ok=True, moved_to=str(dst))


def _snapshot_workspace(tag):
    """Hardlink snapshot of the whole workspace (masks + indexes + gui_objects)
    into <workspace>/snapshots/<stamp>_<tag>/ — near-zero disk/time. Restore with:
    rsync -a <snapshot>/ <workspace>/ . Keeps the newest 10."""
    root = G.out_dir(CFG["capture"])
    if not any(p.is_dir() and "__" in p.name for p in root.iterdir()):
        return                                        # empty workspace: nothing to back up
    snaps = root / "snapshots"
    dest = snaps / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}"
    try:
        shutil.copytree(root, dest, copy_function=os.link,
                        ignore=shutil.ignore_patterns("snapshots", ".trash", "_review"))
        for old in sorted(snaps.iterdir())[:-10]:     # prune beyond the newest 10
            shutil.rmtree(old, ignore_errors=True)
        print(f"snapshot -> {dest}", flush=True)
    except Exception as e:                            # backup must never block work
        print(f"warn: snapshot failed: {e}", flush=True)


def _cloud_diff_job(job_id, opts):
    try:
        cap = CFG["capture"]
        states = CFG["states"]                       # {'pre':{session,ref}, 'post':{...}}
        pre_ref, post_ref = states["pre"]["ref"], states["post"]["ref"]
        bridge = (Path(cap) / "changes" / f"{post_ref}_to_{pre_ref}"
                  / f"T_{pre_ref}_from_{post_ref}.txt")
        if not bridge.exists():
            raise FileNotFoundError(f"bridge not found: {bridge}")
        cb = lambda s: JOBS[job_id].update(msg=s)
        mc = int(opts.get("min_cluster", 500))
        proposals = CD.propose(
            cap, states, str(bridge),
            tau=float(opts.get("tau", 0.10)),
            voxel=float(opts.get("voxel", 0.02)),
            eps=float(opts.get("eps", 0.10)),
            remove_floor=bool(opts.get("remove_floor", True)),
            # strict depth-based occlusion: crisper depth, tighter tolerance, and a
            # min visible-fraction so changes hidden behind geometry (a wall) aren't
            # logged. Off => the old lenient test.
            occ_scale=0.5 if opts.get("strict_occ", True) else 0.35,
            occ_tol=0.05 if opts.get("strict_occ", True) else 0.10,
            min_frac=0.3 if opts.get("strict_occ", True) else 0.0,
            min_cluster=mc,
            # DBSCAN core density scales with cluster size so small objects (few
            # points) still form a cluster instead of being read as noise.
            min_points=max(4, min(10, mc // 10)),
            # min_frames counts visibility on a ~150-frame SUBSAMPLE, so keep it
            # low (~5) or small objects (seen in tens of frames) get excluded.
            min_frames=int(opts.get("min_frames", 5)),
            progress=cb, verbose=True)
        # register each survivor, then fill its per-frame SAM masks (shared model)
        for i, (key, session, entry, n_frames) in enumerate(proposals, 1):
            OBJECTS[key] = entry
            JOBS[job_id].update(msg=f"segmenting proposal {i}/{len(proposals)}: "
                                    f"{key} ({n_frames} frames) …")
            _perframe_inproc(key, session)
        _save_working()
        # ── combined pipeline: ensure change fields, then TEXTURE diff ──
        # (appearance changes on geometrically-static surfaces; proposals are
        # named tex_NN__<state>, distinct from the cd_* geometric ones)
        fdir = G.out_dir(CFG["capture"]) / "fields"
        if not (fdir / "static_pre.npy").exists():
            JOBS[job_id].update(msg="computing change fields …")
            CD.compute_fields(cap, pre_ref, post_ref, fdir,
                              tau=float(opts.get("tau", 0.10)),
                              tau_lo=opts.get("tau_lo") or None,
                              progress=lambda s: JOBS[job_id].update(msg="fields: " + s))
            SCENE_SPLATS.clear()
        tex = []
        try:
            import appearance_check as AC
            tex = AC.run(cap, states, G.out_dir(CFG["capture"]),
                         progress=lambda s: JOBS[job_id].update(msg="texture: " + s))
        except Exception as te:                      # texture is additive, never fatal
            JOBS[job_id].update(msg=f"texture diff failed (non-fatal): {te}")
        for i, (key, session, entry, n_frames) in enumerate(tex, 1):
            OBJECTS[key] = entry
            JOBS[job_id].update(msg=f"segmenting texture proposal {i}/{len(tex)}: {key} …")
            _perframe_inproc(key, session)
        if tex:
            _save_working()
        JOBS[job_id].update(status="done", n_proposals=len(proposals),
                            n_texture=len(tex),
                            keys=[k for k, *_ in proposals] + [k for k, *_ in tex])
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


# ─────────────────── label -> concept remask (job) ───────────────────
@app.route("/api/remask", methods=["POST"])
def api_remask():
    """Re-segment an object with SAM3's CONCEPT path from its (human-typed) label,
    to capture thin structure (legs/base) the point+box mask drops. Background job."""
    d = request.get_json()
    oid, label = d["id"], (d.get("label") or "").strip()
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if OBJECTS[oid].get("done"):
        return jsonify(error="object is marked done — uncheck done to remask"), 400
    if not label:
        return jsonify(error="type a label first (e.g. chair, table)"), 400
    if not (G.out_dir(CFG["capture"], oid) / "seeds.json").exists():
        return jsonify(error="no seeds for this object — propagate/detect it first"), 400
    OBJECTS[oid]["label"] = label
    _save_working()
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "obj": oid, "msg": "starting …"}
    threading.Thread(target=_remask_job, args=(job_id, oid, label), daemon=True).start()
    return jsonify(job_id=job_id)


def _remask_job(job_id, oid, label):
    try:
        cap = Path(CFG["capture"])
        od = G.out_dir(CFG["capture"], oid)
        seeds = json.load(open(od / "seeds.json"))
        mi_path = od / "masks_index.json"
        mi = json.load(open(mi_path)) if mi_path.exists() else {}
        (od / "masks").mkdir(exist_ok=True)
        names = list(seeds.keys())
        # co-GT protection: hand frames and everything at/behind the object's g
        # (verified frontier) are human territory — never re-masked, not even run.
        frontier = OBJECTS[oid].get("verified_until")
        f_ts = int(Path(frontier).stem) if frontier else -1
        tiles, accepted, kept = [], 0, 0
        for i, name in enumerate(names, 1):
            if mi.get(name, {}).get("src") == "hand" or int(Path(name).stem) <= f_ts:
                kept += 1
                continue
            s = seeds[name]
            sid, gb = s.get("session"), s["box"]
            bgr = cv2.imread(str(cap / "sessions" / sid / "raw_data" / name))
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with LOCK:
                masks, boxes, scores = G._segment_concept(MODEL, PROC, img, label)
            chosen = G.pick_concept_instance(masks, boxes, scores, gb)
            flat = name.replace("/", "_")
            if chosen is not None and chosen.any():          # concept matched -> improve
                cv2.imwrite(str(od / "masks" / flat), (chosen * 255).astype(np.uint8))
                mi[name] = {"session": sid, "mask_file": f"masks/{flat}",
                            "px": int(chosen.sum()), "src": "concept"}
                m = chosen
                accepted += 1
            else:                                            # fallback: keep existing mask
                kept += 1
                m = None
                if name in mi:
                    prev = cv2.imread(str(od / mi[name]["mask_file"]), 0)   # None if file missing
                    if prev is not None:
                        m = prev > 127
            ov = bgr.copy()
            if m is not None:
                ov[m] = (0.45 * ov[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            tiles.append(cv2.resize(ov, (242, 242)))
            if i % 20 == 0:
                JOBS[job_id].update(msg=f"remasking {i}/{len(names)} · {accepted} improved")
        json.dump(mi, open(mi_path, "w"), indent=1)
        if tiles:
            while len(tiles) % 4:
                tiles.append(np.zeros((242, 242, 3), np.uint8))
            rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
            cv2.imwrite(str(od / "result_contact.png"), np.concatenate(rows, 0))
        JOBS[job_id].update(status="done", accepted=accepted, kept=kept, n_masks=len(mi),
                            contact=f"/results/{oid}/result_contact.png")
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


# ─────────────── propagate this fix (anchored tracker, preview->confirm) ───────────────
@app.route("/api/propagate_fix", methods=["POST"])
def api_propagate_fix():
    """Carry the anchor frame's SAVED mask to temporal neighbours (SAM3 video tracker,
    geometry-gated; see propagate_fix.py). Results are held for PREVIEW — nothing is
    written until /api/propagate_apply confirms."""
    d = request.get_json()
    oid, frame = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if OBJECTS[oid].get("done"):
        return jsonify(error="object is marked done — uncheck done to propagate"), 400
    mi_path = G.out_dir(CFG["capture"], oid) / "masks_index.json"
    mi = json.load(open(mi_path)) if mi_path.exists() else {}
    if frame not in mi:
        return jsonify(error="this frame has no saved mask — fix & save it first"), 400
    mode = d.get("mode", "forward")                  # forward = never write behind the click
    # evict any prior pending preview for this object (its masks pin ~0.5GB/span)
    # and cap total pending jobs — abandoned previews must not leak for the
    # process lifetime.
    for jid in [j for j in list(PROP_PENDING)
                if (JOBS.get(j) or {}).get("obj") == oid]:
        PROP_PENDING.pop(jid, None)
        PROP_FLAGS.pop(jid, None)
        JOBS.get(jid, {}).update(status="superseded")
    while len(PROP_PENDING) > 3:                     # oldest first (dict insertion order)
        old = next(iter(PROP_PENDING))
        PROP_PENDING.pop(old, None)
        PROP_FLAGS.pop(old, None)
        JOBS.get(old, {}).update(status="superseded")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "obj": oid, "msg": "starting …", "mode": mode}
    threading.Thread(target=_propagate_fix_job,
                     args=(job_id, oid, frame, mode), daemon=True).start()
    return jsonify(job_id=job_id)


def _propagate_fix_job(job_id, oid, frame, mode):
    try:
        od = G.out_dir(CFG["capture"], oid)
        cb = lambda s: JOBS[job_id].update(msg=s)
        cb("loading video tracker (first use takes ~1 min) …")
        results, flags, meta = PF.propagate(
            CFG["capture"], od, frame, mode=mode,
            frontier_name=OBJECTS[oid].get("verified_until"), progress=cb)
        if not results:
            JOBS[job_id].update(status="done", n=0,
                                msg="nothing to fill (span already hand-covered?)")
            return
        mi = json.load(open(od / "masks_index.json"))
        # NOTE: proposals are the tracker's own masks, untouched. A geometry
        # "snap" stage that auto-corrected them was tried and rolled back
        # (2026-07-24): partial clusters/lifted shells clamped SAM's pose-driven
        # evolution. Geometry never modifies propagation output.
        # shrink guard: replacing an existing mask with one under 60% of its size
        # is a suspicious downgrade — flag orange, the human decides at preview.
        n_shrink = 0
        for n in results:
            e = mi.get(n)
            if e and e.get("px") and int(results[n].sum()) < 0.6 * e["px"]:
                flags.setdefault(n, {})["shrink"] = True
                n_shrink += 1
        # preview grid: up to 24 sampled frames; YELLOW border = low-confidence fill,
        # ORANGE = geometry disagreed / suspicious shrink vs the existing mask,
        # CYAN = overwrites a legacy (pre-provenance) mask. Hand frames never appear.
        names = sorted(results)
        sel = names[::max(1, len(names) // 24)][:24]
        tiles = []
        for n in sel:
            sid = mi.get(n, mi[frame])["session"]
            bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / n))
            m = results[n]
            bgr[m] = (0.45 * bgr[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
            t = cv2.resize(bgr, (330, 330))
            f = flags.get(n, {})
            if f.get("lowconf"):
                cv2.rectangle(t, (0, 0), (329, 329), (0, 220, 255), 8)
            elif f.get("shrink"):
                cv2.rectangle(t, (0, 0), (329, 329), (0, 128, 255), 6)
            elif f.get("legacy"):
                cv2.rectangle(t, (0, 0), (329, 329), (255, 200, 0), 5)
            cv2.rectangle(t, (0, 0), (330, 20), (0, 0, 0), -1)
            tag = ("  LOWCONF" if f.get("lowconf") else
                   "  SHRINK" if f.get("shrink") else "")
            cv2.putText(t, Path(n).stem[-8:] + tag,
                        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            tiles.append(t)
        while len(tiles) % 4:
            tiles.append(np.zeros((330, 330, 3), np.uint8))
        rows = [np.concatenate(tiles[j:j + 4], 1) for j in range(0, len(tiles), 4)]
        prev = f"prop_preview_{job_id}.png"
        cv2.imwrite(str(od / prev), np.concatenate(rows, 0))
        PROP_PENDING[job_id] = results
        PROP_FLAGS[job_id] = flags
        n_low = sum(1 for f in flags.values() if f.get("lowconf"))
        n_leg = sum(1 for f in flags.values() if f.get("legacy"))
        JOBS[job_id].update(status="done", n=len(results), n_lowconf=n_low,
                            n_legacy=n_leg, anchors=meta["anchors"], span=meta["span"],
                            n_shrink=n_shrink,
                            resumes=len(meta.get("resumes", [])),
                            invisible=meta.get("invisible", 0),
                            stops=[s["stopped"] for s in meta["stops"]],
                            preview=f"/results/{oid}/{prev}")
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


@app.route("/api/prop_frames")
def api_prop_frames():
    """Pending-review support: which frames a propagation proposes, with flags —
    drives the maskmap colouring and per-frame veto in the GUI."""
    job_id = request.args["job_id"]
    results = PROP_PENDING.get(job_id)
    if results is None:
        return jsonify(error="no pending propagation"), 404
    flags = PROP_FLAGS.get(job_id, {})
    return jsonify(frames=[{"name": n, **flags.get(n, {})} for n in sorted(results)])


@app.route("/api/prop_overlay")
def api_prop_overlay():
    """Pending-review frame: PROPOSED mask as red fill, CURRENT stored mask as a
    green contour — one glance answers 'is the proposal better than what's there'."""
    job_id, name = request.args["job_id"], request.args["name"]
    results = PROP_PENDING.get(job_id)
    job = JOBS.get(job_id) or {}
    oid = job.get("obj")
    if results is None or name not in results or oid is None:
        return jsonify(error="no pending mask for this frame"), 404
    od, _, _, cur = _mask_entry(oid, name)
    sid = cur["session"] if cur else CFG["states"][OBJECTS[oid]["state"]]["session"]
    bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / name))
    if bgr is None:
        return jsonify(error="frame not found"), 404
    m = results[name]
    bgr[m] = (0.45 * bgr[m] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
    if cur:
        prev = cv2.imread(str(od / cur["mask_file"]), 0)
        if prev is not None:
            cnts, _ = cv2.findContours((prev > 127).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(bgr, cnts, -1, (0, 255, 0), 3)
    ok, buf = cv2.imencode(".jpg", bgr)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/propagate_apply", methods=["POST"])
def api_propagate_apply():
    """Write a confirmed propagation. src='hand' frames are co-ground-truth and are
    NEVER overwritten (results shouldn't contain them; skipped here regardless).
    Every overwritten mask is backed up under masks/.bak_<job>/."""
    d = request.get_json()
    job_id = d["job_id"]
    results = PROP_PENDING.get(job_id)
    job = JOBS.get(job_id) or {}
    oid = job.get("obj")
    if results is None or oid is None:
        return jsonify(error="no pending propagation for this job"), 404
    # apply-time REVALIDATION: the preview may be stale — done/frontier can have
    # changed between preview and apply, and protection must hold at WRITE time.
    if OBJECTS.get(oid, {}).get("done"):
        return jsonify(error="object was marked done after this preview — uncheck done to apply"), 400
    f_ts = G.frontier_ts(OBJECTS.get(oid, {}).get("verified_until"))
    exclude = set(d.get("exclude") or [])            # per-frame vetoes from pending review
    od = G.out_dir(CFG["capture"], oid)
    mi_path = od / "masks_index.json"
    mi = json.load(open(mi_path))
    bak = od / "masks" / f".bak_{job_id}"
    written, skipped = 0, 0
    for name, mask in sorted(results.items()):
        cur = mi.get(name)
        if (name in exclude or (cur and cur.get("src") == "hand")
                or int(Path(name).stem) <= f_ts):    # veto / co-GT / current frontier
            skipped += 1
            continue
        flat = name.replace("/", "_")
        if cur and (od / cur["mask_file"]).exists():         # backup before overwrite
            bak.mkdir(parents=True, exist_ok=True)
            shutil.copy(od / cur["mask_file"], bak / flat)
        cv2.imwrite(str(od / "masks" / flat), (mask * 255).astype(np.uint8))
        sid = cur["session"] if cur else mi[sorted(mi)[0]]["session"]
        mi[name] = {"session": sid, "mask_file": f"masks/{flat}",
                    "px": int(mask.sum()), "src": "prop"}
        written += 1
    json.dump(mi, open(mi_path, "w"), indent=1)
    PROP_PENDING.pop(job_id, None)
    PROP_FLAGS.pop(job_id, None)
    return jsonify(ok=True, written=written, skipped_hand=skipped, n_masks=len(mi))


@app.route("/api/propagate_discard", methods=["POST"])
def api_propagate_discard():
    jid = request.get_json().get("job_id")
    PROP_PENDING.pop(jid, None)
    PROP_FLAGS.pop(jid, None)
    return jsonify(ok=True)


# ───────────────── geom mask (dense lidar-cluster silhouette) ─────────────────
def _geom_ctx(state):
    """(capo, sess, renderer) for a state's mesh, built once (GEOM_LOCK guards
    the build: impatient double-clicks must not build two renderers)."""
    with GEOM_LOCK:
        if state not in GEOM_CTX:
            from scantools.proc.rendering import Renderer
            from scantools.utils.io import read_mesh
            st = CFG["states"][state]
            capo, sess = G._session(CFG["capture"], st["session"], st["ref"])
            mesh = capo.proc_path(st["ref"]) / capo.sessions[st["ref"]].proc.meshes["mesh"]
            GEOM_CTX[state] = (capo, sess, Renderer(read_mesh(mesh)))
    return GEOM_CTX[state]


def _frame_key(state, sess, name):
    """Image key for a frame name (per-state cache; key_pairs scan is O(session))."""
    km = GEOM_KEYMAP.setdefault(state, {})
    if not km:
        for k in sess.images.key_pairs():
            km[str(sess.images[k[0], k[1]])] = k
    return km.get(name)


DEPTH_EDGES = {}      # (state, frame) -> png bytes: mesh depth-discontinuity outline
SCENE_SPLATS = {}     # (state, frame) -> (static bool HxW, changed bool HxW) projections


@app.route("/api/texture_check", methods=["POST"])
def api_texture_check():
    """DINOv3 appearance check on geometrically-static surfaces: poster/screen/
    banner content changes the lidar can't see become tex_NN proposals with
    seeds + SAM masks, reviewable like cloud-diff objects."""
    _snapshot_workspace("pretexture")
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "msg": "starting texture check …"}
    threading.Thread(target=_texture_job, args=(job_id,), daemon=True).start()
    return jsonify(job_id=job_id)


def _texture_job(job_id):
    try:
        import appearance_check as AC
        proposals = AC.run(CFG["capture"], CFG["states"], G.out_dir(CFG["capture"]),
                           progress=lambda s: JOBS[job_id].update(msg=s))
        for i, (key, session, entry, n_frames) in enumerate(proposals, 1):
            OBJECTS[key] = entry
            JOBS[job_id].update(msg=f"segmenting texture proposal {i}/{len(proposals)}: {key} …")
            _perframe_inproc(key, session)
        _save_working()
        JOBS[job_id].update(status="done", n_proposals=len(proposals),
                            keys=[k for k, *_ in proposals])
    except (Exception, SystemExit) as e:
        JOBS[job_id].update(status="error", error=str(e))


@app.route("/api/compute_fields", methods=["POST"])
def api_compute_fields():
    """Background job: diff the two scans ONCE and persist, per state, the
    CERTIFIED-STATIC point field (matched within tau_lo both ways — geometric
    evidence of NO change; matching is far more reliable than change detection)
    and the CHANGED-CANDIDATE field (full hysteresis set, unclustered, no size
    or attention gates — everything the lidar suspects). These feed the scene
    view's green/magenta layers and the coverage stats."""
    d = request.get_json() or {}
    tau_lo = d.get("tau_lo") or None                 # None -> tau/3 (0.033 m)
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "msg": "loading clouds …"}
    threading.Thread(target=_fields_job, args=(job_id, tau_lo), daemon=True).start()
    return jsonify(job_id=job_id)


def _fields_job(job_id, tau_lo=None):
    try:
        out = G.out_dir(CFG["capture"]) / "fields"
        n = CD.compute_fields(CFG["capture"], CFG["states"]["pre"]["ref"],
                              CFG["states"]["post"]["ref"], out, tau_lo=tau_lo,
                              progress=lambda s: JOBS[job_id].update(msg=s))
        SCENE_SPLATS.clear()                             # stale projections
        JOBS[job_id].update(status="done",
                            msg="fields written: " +
                                ", ".join(f"{k} {v:,}" for k, v in n.items()))
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


def _union_mask(state, frame):
    """Union of every object's saved mask on this frame (bool full-res or None)."""
    u = None
    for key, o in OBJECTS.items():
        if o.get("state") != state or o.get("ghost"):
            continue
        od = G.out_dir(CFG["capture"], key)
        mi_path = od / "masks_index.json"
        if not mi_path.exists():
            continue
        e = json.load(open(mi_path)).get(frame)
        if not e:
            continue
        m = cv2.imread(str(od / e["mask_file"]), 0)
        if m is None:
            continue
        u = (m > 127) if u is None else (u | (m > 127))
    return u


def _scene_splats(state, frame):
    """(static bool, changed bool) projections for a frame, occlusion-tested,
    cached. Raises FileNotFoundError until compute_fields has run."""
    key = (state, frame)
    if key in SCENE_SPLATS:
        return SCENE_SPLATS[key]
    fdir = G.out_dir(CFG["capture"]) / "fields"
    sp = fdir / f"static_{state}.npy"
    chp = fdir / f"changed_{state}.npy"
    if not sp.exists() or not chp.exists():
        raise FileNotFoundError("fields not computed")
    from scantools.utils.geometry import project, sample_depth
    capo, sess, renderer = _geom_ctx(state)
    k = _frame_key(state, sess, frame)
    if k is None:
        raise KeyError("frame not in session")
    cam = sess.sensors[k[1]]
    T = sess.get_pose(k[0], k[1])
    cam_s, sx, sy = G._scaled_camera(cam, 0.5)
    _, depth = renderer.render_from_capture(T, cam_s)
    H, W = cam.height, cam.width
    outs = []
    dis = G.out_dir(CFG["capture"]) / "fields" / f"dismissed_{state}.npy"
    for p in (sp, chp):
        P = np.load(p)
        if p is sp and dis.exists():                 # dismissed = certified no-change
            P = np.vstack([P, np.load(dis)])
        p2d, z, vis = project(P.astype(np.float64), cam, pose=T.inverse())
        m = np.zeros((H // 2, W // 2), np.uint8)
        if vis.any():
            occ_z, occ_ok = sample_depth(p2d[vis] * np.array([sx, sy]), depth)
            pv = p2d[vis][occ_ok & (z[vis] <= occ_z + 0.05)] / 2.0
            pv = pv[(pv[:, 0] >= 0) & (pv[:, 0] < W // 2)
                    & (pv[:, 1] >= 0) & (pv[:, 1] < H // 2)].astype(np.int32)
            m[pv[:, 1], pv[:, 0]] = 255
            m = cv2.dilate(m, np.ones((7, 7), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        outs.append(cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0)
    while len(SCENE_SPLATS) > 200:
        SCENE_SPLATS.pop(next(iter(SCENE_SPLATS)))
    SCENE_SPLATS[key] = (outs[0], outs[1])
    return SCENE_SPLATS[key]


@app.route("/api/scene_overlay")
def api_scene_overlay():
    """Scene-view overlay PNG. mode 1: RED = union of all annotated masks;
    mode 2: + GREEN certified-static; mode 3: + MAGENTA machine-changed
    candidates. Uncoloured pixels in mode 3 = no lidar evidence OR missed
    change — the completeness check, visually."""
    state, frame = request.args["state"], request.args["frame"]
    mode = int(request.args.get("mode", 1))
    u = _union_mask(state, frame)
    static = changed = None
    if mode >= 2:
        try:
            static, changed = _scene_splats(state, frame)
        except FileNotFoundError:
            return jsonify(error="run 'compute change fields' first (cloud panel)"), 404
        except KeyError:
            return jsonify(error="frame not in this state's session"), 404
    if u is None and static is None:
        return jsonify(error="nothing to show on this frame"), 404
    ref = u if u is not None else static
    H, W = ref.shape
    rgba = np.zeros((H, W, 4), np.uint8)
    if static is not None:                               # green: certified static
        s = static & ~(u if u is not None else False)
        rgba[s] = (60, 200, 60, 70)
    if mode >= 3 and changed is not None:                # magenta: unclaimed change
        c = changed & ~(u if u is not None else False)
        if static is not None:
            c = c & ~static
        rgba[c] = (200, 0, 220, 110)
    if u is not None:                                    # red: annotated union (wins)
        rgba[u] = (0, 0, 255, 100)
    ok, buf = cv2.imencode(".png", rgba)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/scene_coverage")
def api_scene_coverage():
    """Coverage stats for the scene view status line."""
    state, frame = request.args["state"], request.args["frame"]
    u = _union_mask(state, frame)
    try:
        static, changed = _scene_splats(state, frame)
    except Exception:
        return jsonify(error="fields not computed"), 404
    H, W = static.shape
    tot = H * W
    ub = u if u is not None else np.zeros((H, W), bool)
    unclaimed = changed & ~ub & ~static
    none = ~ub & ~static & ~changed
    return jsonify(annotated=round(100 * ub.mean(), 1),
                   static=round(100 * (static & ~ub).mean(), 1),
                   unclaimed=round(100 * unclaimed.mean(), 1),
                   no_evidence=round(100 * none.mean(), 1))


SEED_BOXES = {}       # key -> (seeds.json mtime, {frame: box}) for who-is-here fallback


@app.route("/api/claim_purple", methods=["POST"])
def api_claim_purple():
    """Merge the UNCLAIMED-CHANGE (purple) connected component at (x,y) into an
    existing object's mask on this frame — shift-click in scene view while
    reviewing the target. SAM-refined inside the component's envelope; the
    result is written as src='prop' (survives re-seed, freely improvable).
    Co-GT rules hold: hand frames, at/behind-frontier frames and done objects
    refuse."""
    d = request.get_json()
    oid, frame = d["id"], d["frame"]
    x, y = int(d["x"]), int(d["y"])
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    o = OBJECTS[oid]
    if o.get("done"):
        return jsonify(error="object is done — uncheck to edit"), 400
    if int(Path(frame).stem) <= G.frontier_ts(o.get("verified_until")):
        return jsonify(error="frame is at/behind the frontier"), 400
    state = o["state"]
    try:
        static, changed = _scene_splats(state, frame)
    except Exception:
        return jsonify(error="fields not computed"), 404
    u = _union_mask(state, frame)
    free = changed & ~(u if u is not None else False)
    if not (0 <= y < free.shape[0] and 0 <= x < free.shape[1]) or not free[y, x]:
        return jsonify(error="no unclaimed purple at this pixel"), 400
    ncomp, lab = cv2.connectedComponents(free.astype(np.uint8))
    comp = lab == lab[y, x]
    od = G.out_dir(CFG["capture"], oid)
    mi_path = od / "masks_index.json"
    mi = json.load(open(mi_path)) if mi_path.exists() else {}
    e = mi.get(frame)
    if e and e.get("src") == "hand":
        return jsonify(error="this frame's mask is hand co-GT — edit it manually"), 400
    sid = e["session"] if e else CFG["states"][state]["session"]
    # SAM-refine the blobby splat to the image edge, bounded to its envelope
    add = comp
    try:
        bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / frame))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        clip = cv2.dilate(comp.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        r = _sam_refine_geom(img, comp, clip)
        inter, union = (r & comp).sum(), (r | comp).sum()
        if r.sum() >= 200 and union and inter / union >= 0.3:
            add = r
    except Exception:
        pass
    if d.get("preview"):                             # arm step: show, don't write
        H, W = add.shape
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[add] = (0, 0, 150, 210)                 # darker red = armed mergee
        ok, buf = cv2.imencode(".png", rgba)
        import base64
        return jsonify(preview="data:image/png;base64," + base64.b64encode(buf).decode(),
                       px=int(add.sum()))
    cur = None
    if e:
        m0 = cv2.imread(str(od / e["mask_file"]), 0)
        cur = (m0 > 127) if m0 is not None else None
    merged = add if cur is None else (cur | add)
    flat = frame.replace("/", "_")
    (od / "masks").mkdir(exist_ok=True)
    if e and (od / e["mask_file"]).exists():         # backup before overwrite
        bak = od / "masks" / ".bak_claim"
        bak.mkdir(parents=True, exist_ok=True)
        shutil.copy(od / e["mask_file"], bak / flat)
    cv2.imwrite(str(od / "masks" / flat), (merged * 255).astype(np.uint8))
    mi[frame] = {"session": sid, "mask_file": f"masks/{flat}",
                 "px": int(merged.sum()), "src": "prop"}
    json.dump(mi, open(mi_path, "w"), indent=1)
    return jsonify(ok=True, added=int(add.sum()), px=int(merged.sum()))


@app.route("/api/dismiss_object", methods=["POST"])
def api_dismiss_object():
    """Human-certified NO-CHANGE: the object's masks are removed (recoverable,
    .trash) and its 3D cluster joins the workspace's dismissed field, painted
    GREEN by the scene view from now on. Confirmed client-side with 'g' after
    an explicit review warning — dismissal applies to the WHOLE sequence."""
    oid = request.get_json()["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if OBJECTS[oid].get("done"):
        return jsonify(error="object is done — uncheck to dismiss"), 400
    state = OBJECTS[oid]["state"]
    od = G.out_dir(CFG["capture"], oid)
    greened = False
    for cn in ("cluster.npy", "cluster_enriched.npy"):
        cp = od / cn
        if cp.exists():
            fdir = G.out_dir(CFG["capture"]) / "fields"
            fdir.mkdir(parents=True, exist_ok=True)
            dp = fdir / f"dismissed_{state}.npy"
            P = np.load(cp).astype(np.float32)
            if dp.exists():
                P = np.vstack([np.load(dp), P])
            np.save(dp, P)
            greened = True
            break
    OBJECTS.pop(oid)
    _save_working()
    if od.exists():
        trash = Path(CFG["capture"]) / G.OUT / ".trash"
        trash.mkdir(exist_ok=True)
        dest = trash / oid
        if dest.exists():
            dest = trash / f"{oid}__{uuid.uuid4().hex[:8]}"
        shutil.move(str(od), str(dest))
    SCENE_SPLATS.clear()                             # green layer changed
    return jsonify(ok=True, greened=greened)


@app.route("/api/purple_blobs", methods=["GET", "POST"])
def api_purple_blobs():
    """POST: (re)compute unresolved purple blobs per state — changed-field
    points minus everything explained (near an object's cluster or dismissed),
    DBSCAN'd into blob entries persisted under fields/blobs/. GET: list them.
    Every blob must be RESOLVED: promote (new object), merge (into an existing
    object's cluster), or dismiss (certified no-change, green)."""
    fdir = G.out_dir(CFG["capture"]) / "fields"
    bj = fdir / "blobs.json"
    if request.method == "GET":
        return jsonify(blobs=json.load(open(bj)) if bj.exists() else [])
    from scipy.spatial import cKDTree
    import open3d as o3d
    blobs = []
    (fdir / "blobs").mkdir(parents=True, exist_ok=True)
    for state in ("pre", "post"):
        chp = fdir / f"changed_{state}.npy"
        if not chp.exists():
            return jsonify(error="fields not computed"), 404
        P = np.load(chp).astype(np.float64)
        explained = []
        for key, o in OBJECTS.items():
            if o.get("state") != state or o.get("ghost"):
                continue
            for cn in ("cluster.npy", "cluster_enriched.npy"):
                cp = G.out_dir(CFG["capture"], key) / cn
                if cp.exists():
                    explained.append(np.load(cp).astype(np.float64))
                    break
        dp = fdir / f"dismissed_{state}.npy"
        if dp.exists():
            explained.append(np.load(dp).astype(np.float64))
        if explained:
            d = cKDTree(np.vstack(explained)).query(P, workers=-1)[0]
            P = P[d > 0.30]
        if len(P) < 40:
            continue
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        lab = np.asarray(pcd.cluster_dbscan(eps=0.12, min_points=8))
        for li in range(lab.max() + 1):
            pts = P[lab == li]
            if len(pts) < 60:
                continue
            bid = f"blob_{state}_{len(blobs):03d}"
            np.save(fdir / "blobs" / f"{bid}.npy", pts.astype(np.float32))
            blobs.append({"id": bid, "state": state, "n": int(len(pts)),
                          "sig": CD._signature(pts)})
    blobs.sort(key=lambda b: -b["n"])
    json.dump(blobs, open(bj, "w"), indent=1)
    return jsonify(blobs=blobs)


@app.route("/api/resolve_blob", methods=["POST"])
def api_resolve_blob():
    """Resolve one purple blob: action=dismiss (points join the green dismissed
    field), merge (points join TARGET object's cluster.npy — re-run its
    propagate to regenerate masks), or promote (becomes a new proposal object
    with seeds + SAM masks, like a cloud-diff survivor)."""
    d = request.get_json()
    bid, action = d["id"], d["action"]
    fdir = G.out_dir(CFG["capture"]) / "fields"
    bj = fdir / "blobs.json"
    blobs = json.load(open(bj)) if bj.exists() else []
    b = next((x for x in blobs if x["id"] == bid), None)
    bp = fdir / "blobs" / f"{bid}.npy"
    if b is None or not bp.exists():
        return jsonify(error="unknown blob"), 404
    pts = np.load(bp)
    state = b["state"]
    if action == "dismiss":
        dp = fdir / f"dismissed_{state}.npy"
        P = np.vstack([np.load(dp), pts]) if dp.exists() else pts
        np.save(dp, P)
        msg = "dismissed — region certified no-change (green)"
    elif action == "merge":
        target = d.get("target") or ""
        if target not in OBJECTS:
            return jsonify(error=f"unknown target object '{target}'"), 404
        od = G.out_dir(CFG["capture"], target)
        cp = od / "cluster.npy"
        P = np.vstack([np.load(cp), pts]) if cp.exists() else pts
        np.save(cp, P.astype(np.float32))
        msg = f"merged into {target}'s cluster — run propagate on it to regenerate masks"
    elif action == "promote":
        cand = {"id": bid, "state": state, "change_type": "moved",
                "pts": pts.astype(np.float64), "sig": b["sig"]}
        ctx = CD._ctx_for_states(CFG["capture"], CFG["states"], [cand])
        CD._seed([cand], CFG["states"], ctx, 0, 5, 0.5,
                 Path(CFG["capture"]) / "changes" / "cloud_diff" / "seed_dbg",
                 occ_tol=0.05, min_frac=0.0)
        if not cand.get("seeds"):
            return jsonify(error="blob not visible in the walk (occlusion) — dismiss instead?"), 400
        key = f"{bid}__{state}"
        od = G.out_dir(CFG["capture"], key)
        json.dump(cand["seeds"], open(od / "seeds.json", "w"), indent=1)
        np.save(od / "cluster.npy", pts.astype(np.float32))
        OBJECTS[key] = {"id": bid, "label": "", "deformability": "rigid",
                        "state": state, "frame": next(iter(cand["seeds"])),
                        "points": [], "seed_frames": [], "source": "purple_blob"}
        _save_working()
        _perframe_inproc(key, CFG["states"][state]["session"])
        msg = f"promoted to object {key} ({cand['n_frames']} visible frames)"
    else:
        return jsonify(error="unknown action"), 400
    blobs = [x for x in blobs if x["id"] != bid]
    json.dump(blobs, open(bj, "w"), indent=1)
    bp.unlink(missing_ok=True)
    SCENE_SPLATS.clear()
    return jsonify(ok=True, msg=msg, remaining=len(blobs))


@app.route("/api/who_is_here", methods=["POST"])
def api_who_is_here():
    """Which object(s) own this pixel on this frame — scene-view click-to-identify.
    Two tiers: masks (the pixel is annotated), else seed-box territory (a
    proposal's cluster projects here but has no mask on this frame — exactly
    what an unclaimed MAGENTA pixel usually is)."""
    d = request.get_json()
    state, frame = d["state"], d["frame"]
    x, y = int(d["x"]), int(d["y"])
    hits, territory = [], []
    for key, o in OBJECTS.items():
        if o.get("state") != state or o.get("ghost"):
            continue
        od = G.out_dir(CFG["capture"], key)
        mi_path = od / "masks_index.json"
        if mi_path.exists():
            e = json.load(open(mi_path)).get(frame)
            if e:
                m = cv2.imread(str(od / e["mask_file"]), 0)
                if (m is not None and 0 <= y < m.shape[0] and 0 <= x < m.shape[1]
                        and m[y, x] > 127):
                    hits.append(key)
                    continue
        sp = od / "seeds.json"
        if not sp.exists():
            continue
        mt = sp.stat().st_mtime
        if key not in SEED_BOXES or SEED_BOXES[key][0] != mt:
            SEED_BOXES[key] = (mt, {n: s["box"] for n, s in json.load(open(sp)).items()})
        bx = SEED_BOXES[key][1].get(frame)
        if bx and bx[0] <= x <= bx[2] and bx[1] <= y <= bx[3]:
            territory.append(key)
    return jsonify(keys=hits, territory=territory)


@app.route("/api/depth_edges")
def api_depth_edges():
    """Thin cyan outline of mesh depth discontinuities for a frame — geometry is
    immune to exposure, so black furniture and shadowed edges that vanish in RGB
    are still crisp. Display-only overlay; nothing downstream sees it."""
    state, name = request.args["state"], request.args["frame"]
    key = (state, name)
    if key not in DEPTH_EDGES:
        capo, sess, renderer = _geom_ctx(state)
        k = _frame_key(state, sess, name)
        if k is None:
            return jsonify(error="frame not in this state's session"), 404
        cam = sess.sensors[k[1]]
        T = sess.get_pose(k[0], k[1])
        cam_s, sx, sy = G._scaled_camera(cam, 0.5)
        _, depth = renderer.render_from_capture(T, cam_s)
        d = np.asarray(depth, np.float32)
        valid = d > 0
        gx = np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        gy = np.abs(np.diff(d, axis=0, prepend=d[:1]))
        rel = np.maximum(gx, gy) / np.maximum(d, 0.3)     # depth-relative step
        edge = ((rel > 0.04) & valid).astype(np.uint8) * 255
        edge = cv2.dilate(edge, np.ones((2, 2), np.uint8))
        H, W = cam.height, cam.width
        edge = cv2.resize(edge, (W, H), interpolation=cv2.INTER_NEAREST)
        rgba = np.zeros((H, W, 4), np.uint8)
        rgba[..., 0] = 255                                 # BGRA: cyan outline
        rgba[..., 1] = 255
        rgba[..., 3] = edge
        ok, buf = cv2.imencode(".png", rgba)
        if not ok:
            return jsonify(error="encode failed"), 500
        while len(DEPTH_EDGES) > 300:                      # LRU-ish cap (~small PNGs)
            DEPTH_EDGES.pop(next(iter(DEPTH_EDGES)))
        DEPTH_EDGES[key] = buf.tobytes()
    return Response(DEPTH_EDGES[key], mimetype="image/png")


def _project_silhouette(oid, name, cur_px=0, pure_only=False):
    """Project the object's stored 3D geometry into frame `name` -> dense bool
    silhouette (or None). Pure diff cluster first (clean on-object lidar); the
    seed-enriched union (cluster_enriched.npy, written by re-seeding) covers
    holes and gives manual objects geometry too — per-frame fallback, sparse-
    guard failures retry with the union. pure_only skips that fallback (the
    lift-enriched shell must never SHAPE masks — geom-snap). Returns
    (sil, n_points, error_msg)."""
    od = G.out_dir(CFG["capture"], oid)
    cpath = od / "cluster.npy"
    epath = od / "cluster_enriched.npy"
    if pure_only:
        epath = cpath
    if not cpath.exists():
        cpath = epath
    if not cpath.exists():
        return None, 0, ("no 3D geometry stored for this object — run propagate "
                         "(re-seed) or detect-changes first")
    state = OBJECTS[oid]["state"]
    from scantools.utils.geometry import project, sample_depth
    capo, sess, renderer = _geom_ctx(state)
    key = _frame_key(state, sess, name)
    if key is None:
        return None, 0, "frame not in this state's session"
    cam = sess.sensors[key[1]]
    T = sess.get_pose(key[0], key[1])
    cam_s, sx, sy = G._scaled_camera(cam, 0.5)
    _, depth = renderer.render_from_capture(T, cam_s)
    H, W = cam.height, cam.width
    # sparsity guard (validated on the chair, frame 301412: sliver -> IoU 0.10)
    # applies per source.
    sil = pv = None
    tried = []
    for cp in dict.fromkeys([cpath, epath]):         # unique, order-preserving
        if not cp.exists():
            continue
        P = np.load(cp)
        p2d, z, vis = project(P, cam, pose=T.inverse())
        if not vis.any():
            tried.append(f"{cp.stem}: not visible"); continue
        occ_z, occ_ok = sample_depth(p2d[vis] * np.array([sx, sy]), depth)
        pv_c = p2d[vis][occ_ok & (z[vis] <= occ_z + 0.05)]
        if len(pv_c) < 10:
            tried.append(f"{cp.stem}: occluded"); continue
        sil_c = np.zeros((H, W), np.uint8)
        for x, y in pv_c:
            if 0 <= int(x) < W and 0 <= int(y) < H:
                cv2.circle(sil_c, (int(x), int(y)), 9, 255, -1)
        sil_c = cv2.morphologyEx(sil_c, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8)) > 0
        spx = int(sil_c.sum())
        if spx < 800 or (cur_px >= 1000 and spx < 0.35 * cur_px):
            tried.append(f"{cp.stem}: too sparse ({spx}px)"); continue
        sil, pv = sil_c, pv_c
        break
    if sil is None:
        return None, 0, ("geometry unusable in this view (" + "; ".join(tried)
                         + ") — use smart brush / line here")
    return sil, len(pv), None


def _sam_refine_geom(img, seed, clip):
    """SAM refine seeded by a geometry-derived mask (mask_input + core positives
    + box) so the boundary snaps to the image edge where contrast exists and
    keeps geometry where not; clipped to the envelope (anti-runaway). May raise."""
    core = cv2.erode(seed.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    ys, xs = np.where(core if core.any() else seed)
    sel = np.linspace(0, len(xs) - 1, 5).astype(int)
    pos = [[float(xs[k]), float(ys[k])] for k in sel]
    ys2, xs2 = np.where(seed)
    box = [float(xs2.min()) - 40, float(ys2.min()) - 40,
           float(xs2.max()) + 40, float(ys2.max()) + 40]
    with LOCK:
        refined = G._segment_refine(MODEL, PROC, img, pos, [1] * len(pos), box, seed)
    return refined & clip


@app.route("/api/geom_mask", methods=["POST"])
def api_geom_mask():
    """Project the object's raw-cloud diff cluster into the frame -> dense
    silhouette as a starting mask. For objects where visual cues offer no hint
    (curtain vs wall, thin desk vs clutter) the lidar silhouette needs none.
    Measured on the dlab desk: ~2x the signal of the mesh-lift path."""
    d = request.get_json()
    oid, name = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    _, _, _, cur = _mask_entry(oid, name)
    sil, npts, err = _project_silhouette(oid, name, cur.get("px", 0) if cur else 0)
    if sil is None:
        return jsonify(error=err), (400 if "unusable" in err else 404)
    state = OBJECTS[oid]["state"]
    mode_used = "geom+SAM"
    out = sil
    try:
        bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions"
                          / CFG["states"][state]["session"] / "raw_data" / name))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        clip = cv2.dilate(sil.astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
        refined = _sam_refine_geom(img, sil, clip)
        if refined.sum() >= 200:
            out = refined
        else:
            mode_used = "raw silhouette (SAM returned ~empty)"
    except Exception as e:                           # SAM hiccup -> raw silhouette
        mode_used = f"raw silhouette (refine failed: {e})"
    return jsonify(mask=_mask_png_b64(out), px=int(out.sum()), pts=int(npts),
                   mode=mode_used)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    return jsonify(JOBS.get(job_id, {"status": "unknown"}))


def _propagate_job(job_id, oid):
    try:
        _snapshot_workspace("prereseed")   # geom regeneration has no preview gate —
        o = OBJECTS[oid]                   # a cheap hardlink snapshot is the undo
        state = o["state"]
        objdir = oid  # OBJECTS is keyed by '<id>__<state>' = the objdir
        st = CFG["states"][state]
        # 1) geometry seeds (subprocess of the same env; within-source walk only).
        # cmd_seeds unions: explicit src_masks + ALL hand masks + cluster.npy if the
        # object came from cloud diff — the completed geometry covers the holes the
        # cluster alone missed. Cluster objects use the strict occlusion defaults.
        env = dict(os.environ, PYTHONPATH=LAMARIA_PYTHONPATH)  # scantools on path
        seed_cmd = [str(ANNOTATOR_PY), str(G.__file__), "seeds",
                    "--capture", CFG["capture"], "--session", st["session"],
                    "--ref", st["ref"], "--src-name", o["frame"], "--obj", objdir,
                    "--n", str(CFG["seed"]["n"]), "--min-vis", str(CFG["seed"]["min_vis"]),
                    "--no-cross"]
        if (G.out_dir(CFG["capture"], objdir) / "cluster.npy").exists():
            seed_cmd += ["--occ-scale", "0.5", "--occ-tol", "0.05", "--min-frac", "0.3"]
        subprocess.run(seed_cmd, check=True, env=env)
        # 2) per-frame SAM masks (in-process, shared model)
        n = _perframe_inproc(objdir, st["session"])
        # 3) collect viz for review (under the workspace, no external dir)
        od = G.out_dir(CFG["capture"], objdir)
        viz = G.out_dir(CFG["capture"]) / "_review" / objdir
        viz.mkdir(parents=True, exist_ok=True)
        if (od / "result_contact.png").exists():
            shutil.copy(od / "result_contact.png", viz / "result_contact.png")
        if (od / "src_mask.png").exists():
            shutil.copy(od / "src_mask.png", viz / "src_mask.png")
        JOBS[job_id].update(status="done", n_masks=n,
                            contact=f"/results/{objdir}/result_contact.png")
    except subprocess.CalledProcessError as e:
        JOBS[job_id].update(status="error", error=f"seeds failed (rc={e.returncode})")
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


def _perframe_inproc(objdir, default_session):
    od = G.out_dir(CFG["capture"], objdir)
    seeds = json.load(open(od / "seeds.json"))
    masks_dir = od / "masks"
    masks_dir.mkdir(exist_ok=True)
    cap = Path(CFG["capture"])
    # co-GT protection + stale-entry cleanup (shared rules, see G.index_keep_protected):
    # hand/frontier frames and non-geom work survive untouched; stale geom entries
    # for frames no longer seeded are DROPPED so they can't reach the export.
    mi_path = od / "masks_index.json"
    old_index = json.load(open(mi_path)) if mi_path.exists() else {}
    f_ts = G.frontier_ts((OBJECTS.get(objdir) or {}).get("verified_until"))
    mask_index = G.index_keep_protected(old_index, set(seeds), f_ts)

    def _live_protected(name):
        """Protection must hold at WRITE time, not job-start time: this loop
        runs for minutes, during which the annotator may hand-save this very
        frame or advance the frontier past it. The job-start snapshot must
        never authorize a write over fresher co-GT."""
        le = (json.load(open(mi_path)) if mi_path.exists() else {}).get(name)
        f_now = G.frontier_ts((OBJECTS.get(objdir) or {}).get("verified_until"))
        return (le and le.get("src") == "hand") or int(Path(name).stem) <= f_now

    new_entries = {}
    tiles = []
    for name, s in seeds.items():
        if name in mask_index:                    # protected — do not regenerate
            continue
        sid = s.get("session", default_session)
        bgr = cv2.imread(str(cap / "sessions" / sid / "raw_data" / name))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with LOCK:
            masks, _ = G._segment(MODEL, PROC, img, points=s["points"], labels=s["labels"],
                                  box=s["box"], multimask=False)
        mask = masks[0].astype(bool)
        flat = name.replace("/", "_")
        if _live_protected(name):                 # re-check right before the PNG write
            continue
        cv2.imwrite(str(masks_dir / flat), (mask * 255).astype(np.uint8))
        new_entries[name] = {"session": sid, "mask_file": f"masks/{flat}",
                             "px": int(mask.sum()), "src": "geom"}
        ov = bgr.copy()
        ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = [int(v) for v in s["box"]]
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 0), 1)
        tiles.append(cv2.resize(ov, (242, 242)))
    # final write: MERGE onto the LIVE index — never dump the job-start copy,
    # or hand saves / frontier moves made during the job would be reverted.
    live = json.load(open(mi_path)) if mi_path.exists() else {}
    f_now = G.frontier_ts((OBJECTS.get(objdir) or {}).get("verified_until"))
    for name in list(live):                       # stale-geom cleanup, live-guarded
        e = live[name]
        if (e.get("src") == "geom" and name not in seeds
                and int(Path(name).stem) > f_now):
            del live[name]
    for name, e in new_entries.items():
        le = live.get(name)
        if (le and le.get("src") == "hand") or int(Path(name).stem) <= f_now:
            continue
        live[name] = e
    mask_index = live
    json.dump(mask_index, open(od / "masks_index.json", "w"), indent=1)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros((242, 242, 3), np.uint8))
        rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(od / "result_contact.png"), np.concatenate(rows, 0))
    return len(mask_index)


# ───────────────────────── review / edit propagated masks ─────────────────────────
@app.route("/api/objects")
def api_objects():
    """All objects + how many propagated frames each has (drives the sidebar list)."""
    states_by_id = {}
    for o in OBJECTS.values():
        states_by_id.setdefault(o.get("instance", o["id"]), set()).add(o["state"])
    out = {}
    for key, o in OBJECTS.items():
        mi = G.out_dir(CFG["capture"], key) / "masks_index.json"
        n = 0
        if mi.exists():
            try:
                n = len(json.load(open(mi)))
            except Exception:
                n = 0
        idxp = G.out_dir(CFG["capture"], key) / "src_index.json"
        try:
            n_seeds = len(json.load(open(idxp))) if idxp.exists() else 1
        except Exception:
            n_seeds = 1
        sts = states_by_id[o.get("instance", o["id"])]   # change type reflects the shared instance id
        in_pre, in_post = "pre" in sts, "post" in sts
        out[key] = {**o, "n_masks": n, "n_seeds": n_seeds, "in_pre": in_pre, "in_post": in_post,
                    "change_type": "ghost" if o.get("ghost") else _change_type(in_pre, in_post)}
    return jsonify(objects=out)


@app.route("/api/object_frames")
def api_object_frames():
    """ALL source-state frames (sorted by timestamp), each flagged has_mask. Lets the
    reviewer scrub the whole walk and spot frames the propagation missed."""
    oid = request.args["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    sid = CFG["states"][OBJECTS[oid]["state"]]["session"]
    fdir = Path(CFG["capture"]) / "sessions" / sid / "raw_data" / "images" / "cam0"
    names = sorted((f"images/cam0/{p.name}" for p in fdir.glob("*.jpg")),
                   key=lambda n: int(Path(n).stem))     # timestamp order, matches propagation
    mi_path = G.out_dir(CFG["capture"], _objdir(oid)) / "masks_index.json"
    mi = json.load(open(mi_path)) if mi_path.exists() else {}
    frames = [{"name": n, "session": mi[n]["session"] if n in mi else sid,
               "has_mask": n in mi, "px": mi[n].get("px") if n in mi else None}
              for n in names]
    return jsonify(frames=frames, count=len(frames), masked=len(mi),
                   state=OBJECTS[oid]["state"])


def _mask_entry(oid, name):
    od = G.out_dir(CFG["capture"], _objdir(oid))
    mi_path = od / "masks_index.json"
    mi = json.load(open(mi_path)) if mi_path.exists() else {}
    return od, mi_path, mi, mi.get(name)


@app.route("/api/overlay")
def api_overlay():
    """JPEG of frame `name`; overlay object `id`'s mask if it has one there, else plain."""
    oid, name = request.args["id"], request.args["name"]
    o = OBJECTS.get(oid)
    if o is None:
        return jsonify(error="unknown object"), 404
    od, _, _, m = _mask_entry(oid, name)
    sid = m["session"] if m else CFG["states"][o["state"]]["session"]
    bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / name))
    if bgr is None:
        return jsonify(error="frame not found"), 404
    if m:
        mm = cv2.imread(str(od / m["mask_file"]), 0)     # None if the mask file is missing
        if mm is not None:
            bgr = _overlay_mask(bgr, mm > 127)
    ok, buf = cv2.imencode(".jpg", bgr)
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/api/edit_click", methods=["POST"])
def api_edit_click():
    """Re-segment one already-propagated frame from fresh clicks; returns a preview."""
    d = request.get_json()
    oid, name = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    pos = np.array(d.get("points", []), np.float32).reshape(-1, 2)
    neg = np.array(d.get("neg", []), np.float32).reshape(-1, 2)
    if len(pos) == 0:
        return jsonify(error="click a positive point first"), 400
    _, _, _, m = _mask_entry(oid, name)
    sid = m["session"] if m else CFG["states"][OBJECTS[oid]["state"]]["session"]
    bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / name))
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pts = np.concatenate([pos, neg], 0)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    with LOCK:
        masks, scores = G._segment(MODEL, PROC, img, points=pts, labels=labels, multimask=True)
    best = masks[int(np.argmax(scores))].astype(bool)
    EDIT.update(id=oid, frame=name, session=sid, mask=best)
    ov = _overlay_mask(bgr, best)
    for x, y in pos:
        cv2.circle(ov, (int(x), int(y)), 6, (0, 255, 0), -1)
    for x, y in neg:
        cv2.circle(ov, (int(x), int(y)), 6, (255, 0, 0), -1)
    return jsonify(overlay=_png_b64(ov), mask=_mask_png_b64(best), px=int(best.sum()), score=float(scores.max()))


@app.route("/api/refine", methods=["POST"])
def api_refine():
    """Smart brush: refine the CURRENT canvas mask with rough pos/neg scribble points.
    SAM seeds from the mask (mask_input) and snaps to the object edge — a dab pulls a
    missed part in, a stroke removes bleed. state+frame based (works in edit & annotate)."""
    d = request.get_json()
    state, name = d["state"], d["frame"]
    pos = np.array(d.get("pos", []), np.float32).reshape(-1, 2)
    neg = np.array(d.get("neg", []), np.float32).reshape(-1, 2)
    if len(pos) + len(neg) == 0:
        return jsonify(error="scribble something first"), 400
    sid = CFG["states"][state]["session"]
    bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / name))
    if bgr is None:
        return jsonify(error="frame not found"), 404
    H, W = bgr.shape[:2]
    prior = _decode_mask(d["mask"]) if d.get("mask") else np.zeros((H, W), bool)
    if prior.shape != (H, W):
        prior = cv2.resize(prior.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    # box = bbox of (prior ∪ scribbles), padded — so an added dab is inside the prompt box
    ys, xs = np.where(prior)
    xr = np.concatenate([xs, pos[:, 0], neg[:, 0]]) if len(xs) else np.concatenate([pos[:, 0], neg[:, 0]])
    yr = np.concatenate([ys, pos[:, 1], neg[:, 1]]) if len(ys) else np.concatenate([pos[:, 1], neg[:, 1]])
    pad = 30
    box = [max(0, int(xr.min()) - pad), max(0, int(yr.min()) - pad),
           min(W, int(xr.max()) + pad), min(H, int(yr.max()) + pad)]
    pts = np.concatenate([pos, neg], 0)
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with LOCK:
        refined = G._segment_refine(MODEL, PROC, img, pts, labels, box, prior)
    return jsonify(mask=_mask_png_b64(refined), px=int(refined.sum()))


@app.route("/api/save_edit", methods=["POST"])
def api_save_edit():
    """Persist an edited mask (SAM re-segment and/or brush/eraser), overwriting the frame's mask."""
    d = request.get_json()
    oid, name = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    od, mi_path, mi, m = _mask_entry(oid, name)
    if d.get("mask"):                                    # client-painted mask
        mask = _decode_mask(d["mask"])
        sid = m["session"] if m else CFG["states"][OBJECTS[oid]["state"]]["session"]
    elif EDIT.get("id") == oid and EDIT.get("frame") == name and EDIT.get("mask") is not None:
        mask, sid = EDIT["mask"], EDIT["session"]
    else:
        return jsonify(error="no pending edit for this frame"), 400
    (od / "masks").mkdir(exist_ok=True)
    flat = name.replace("/", "_")
    if mask.any():
        cv2.imwrite(str(od / "masks" / flat), (mask * 255).astype(np.uint8))
        # src="hand": human-verified — propagation must never silently overwrite these
        mi[name] = {"session": sid, "mask_file": f"masks/{flat}",
                    "px": int(mask.sum()), "src": "hand"}
    else:                                                # erased to empty -> drop the frame
        f = od / (m["mask_file"] if m else f"masks/{flat}")
        if f.exists():
            f.unlink()
        mi.pop(name, None)
    json.dump(mi, open(mi_path, "w"), indent=1)
    EDIT.clear()
    return jsonify(ok=True, px=int(mask.sum()))


@app.route("/api/mask_png")
def api_mask_png():
    """The object's saved mask for `name` as a red+alpha PNG (transparent if none) —
    loaded into the client mask canvas when entering edit on a frame."""
    oid, name = request.args["id"], request.args["name"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    od, _, _, m = _mask_entry(oid, name)
    sid = m["session"] if m else CFG["states"][OBJECTS[oid]["state"]]["session"]
    mask = None
    if m:
        mm = cv2.imread(str(od / m["mask_file"]), 0)     # None if the mask file is missing
        if mm is not None:
            mask = mm > 127
    if mask is None:
        bgr = cv2.imread(str(Path(CFG["capture"]) / "sessions" / sid / "raw_data" / name))
        mask = np.zeros(bgr.shape[:2], bool)
    h, w = mask.shape
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[mask, 2] = 255
    bgra[mask, 3] = 130
    ok, buf = cv2.imencode(".png", bgra)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/api/delete_mask", methods=["POST"])
def api_delete_mask():
    """Drop one frame's mask. Co-GT frames (src=hand, or at/behind the g frontier)
    need an explicit force=true (the GUI asks for confirmation first); every
    deleted mask is moved to masks/.deleted/ instead of unlinked — recoverable."""
    d = request.get_json()
    oid, name = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    od, mi_path, mi, m = _mask_entry(oid, name)
    if m:
        f_ts = G.frontier_ts(OBJECTS[oid].get("verified_until"))
        protected = m.get("src") == "hand" or int(Path(name).stem) <= f_ts
        if protected and not d.get("force"):
            return jsonify(error="this frame is human-verified (hand/confirmed) — "
                                 "confirm to delete it", protected=True), 400
        f = od / m["mask_file"]
        if f.exists():                               # backup, never hard-delete
            trash = od / "masks" / ".deleted"
            trash.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(trash / f"{f.name}.{int(time.time())}"))
        mi.pop(name, None)
        json.dump(mi, open(mi_path, "w"), indent=1)
    return jsonify(ok=True, count=len(mi))


@app.route("/api/delete_object", methods=["POST"])
def api_delete_object():
    """Remove an erroneous object: drop it from the working set and move its
    on-disk workspace (seeds + masks) to a .trash folder so it stays recoverable
    rather than being permanently deleted."""
    oid = request.get_json()["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    OBJECTS.pop(oid)
    _save_working()
    od = Path(CFG["capture"]) / G.OUT / oid
    if od.exists():
        trash = Path(CFG["capture"]) / G.OUT / ".trash"
        trash.mkdir(exist_ok=True)
        dest = trash / oid
        if dest.exists():
            dest = trash / f"{oid}__{uuid.uuid4().hex[:8]}"
        shutil.move(str(od), str(dest))
    return jsonify(ok=True)


@app.route("/api/set_meta", methods=["POST"])
def api_set_meta():
    """Edit an object's id/label/deformability. The 'instance' id is what links a
    moved object: give a moved pair's pre & post the SAME instance id and they export
    as one 'moved' object. The on-disk dir/key is NEVER renamed (only this editable
    instance id changes), so mask paths stay valid."""
    d = request.get_json()
    key = d["id"]
    if key not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if "instance" in d:
        OBJECTS[key]["instance"] = (d.get("instance") or "").strip() or OBJECTS[key]["id"]
    if "label" in d:
        OBJECTS[key]["label"] = (d.get("label") or "").strip()
    if d.get("deformability") in ("rigid", "deformable"):
        OBJECTS[key]["deformability"] = d["deformability"]
        # a moved pair is ONE physical object: keep deformability consistent across
        # both halves (export merges them; a mismatch would be silently last-wins)
        iid = OBJECTS[key].get("instance", OBJECTS[key]["id"])
        for k2, o2 in OBJECTS.items():
            if k2 != key and o2.get("instance", o2.get("id")) == iid:
                o2["deformability"] = d["deformability"]
    # two-person workflow flags: 'done' (annotator: masks finished) and 'reviewed'
    # (second person verified). Independent booleans, persisted in gui_objects.json
    # so both annotators sharing the workspace see them.
    for flag in ("done", "reviewed"):
        if flag in d:
            OBJECTS[key][flag] = bool(d[flag])
    # verified frontier: "confirmed up to here" — frames at/behind it are co-GT
    # (scrub-approved) and are never touched by propagation.
    if "verified_until" in d:
        OBJECTS[key]["verified_until"] = d.get("verified_until") or None
    _save_working()
    return jsonify(ok=True, instance=OBJECTS[key].get("instance"),
                   done=OBJECTS[key].get("done", False),
                   reviewed=OBJECTS[key].get("reviewed", False))


# ───────────────────────────── grouping ─────────────────────────────
@app.route("/api/group", methods=["POST"])
def api_group():
    """Group 2+ objects as ONE physical object: every member gets the FIRST key's
    instance id (e.g. a curtain annotated as left/right parts, or a moved pair plus
    a third part). Non-destructive: on-disk dirs/masks are untouched and members
    stay independently editable; the export merges a group into a single object.
    Reverse with /api/ungroup."""
    d = request.get_json()
    keys = d.get("keys") or []
    if len(keys) < 2:
        return jsonify(error="select at least two objects to group"), 400
    for k in keys:
        if k not in OBJECTS:
            return jsonify(error=f"unknown object {k}"), 404
        if OBJECTS[k].get("ghost"):
            return jsonify(error="ghost objects cannot be grouped"), 400
    first = OBJECTS[keys[0]]
    iid = first.get("instance") or first["id"]
    deform = first.get("deformability", "rigid")
    for k in keys:
        OBJECTS[k]["instance"] = iid
        OBJECTS[k]["deformability"] = deform   # one physical object (same sync as set_meta)
    _save_working()
    return jsonify(ok=True, instance=iid, n=len(keys))


@app.route("/api/ungroup", methods=["POST"])
def api_ungroup():
    """Dissolve a group: every member's instance id reverts to its own object id."""
    d = request.get_json()
    iid = d.get("instance")
    n = 0
    for o in OBJECTS.values():
        if not o.get("ghost") and (o.get("instance") or o["id"]) == iid:
            o["instance"] = o["id"]
            n += 1
    if not n:
        return jsonify(error="unknown instance"), 404
    _save_working()
    return jsonify(ok=True, n=n)


# ───────────────────────────── export ─────────────────────────────
def _composite_group_mask(iid, state, frame, paths):
    """A group (several GUI objects = one physical object) can have 2+ members with a
    mask on the SAME frame+state (e.g. curtain left/right parts at the seam). The
    schema stays one mask_file per frame, so union the members' masks into a single
    PNG under <GEOM_OUT>/_groups/ and reference that. Returns the file path, or None
    if no member's mask was readable."""
    acc = None
    for p in paths:
        m = cv2.imread(str(p), 0)
        if m is None:
            print(f"warn: group '{iid}' {state} {frame}: unreadable {p}, skipped", flush=True)
            continue
        if acc is not None and m.shape != acc.shape:
            m = cv2.resize(m, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_NEAREST)
        acc = (m > 127) if acc is None else (acc | (m > 127))
    if acc is None:
        return None
    od = G.out_dir(CFG["capture"], f"_groups/{iid.replace('/', '_')}/{state}")
    fp = od / (os.path.splitext(frame.replace("/", "_"))[0] + ".png")
    cv2.imwrite(str(fp), acc.astype(np.uint8) * 255)
    return fp


@app.route("/api/export", methods=["POST"])
def api_export():
    # merge the per-state entries of each shared instance id: a moved pair, or a larger
    # GROUP (several part-annotations of one physical object). Downstream sees ONE
    # object per instance with ONE mask_file per frame; a frame covered by several
    # members gets a union PNG (_composite_group_mask). mask_file paths are relative
    # to <capture>/changes/ (same base as the change_mask block) so they resolve even
    # when the instance id differs from its members' on-disk dir names.
    # ghost pseudo-objects (point_ghost_prototype.py ghosts) are derived output,
    # not annotations -- they never enter segments.json or the native change mask.
    changes_root = Path(CFG["capture"]) / "changes"
    by_id = {}
    for key, o in OBJECTS.items():
        if o.get("ghost"):
            continue
        od = G.out_dir(CFG["capture"], key)
        mi_path = od / "masks_index.json"
        mi = json.load(open(mi_path)) if mi_path.exists() else {}
        iid = o.get("instance") or o["id"]     # group / moved pair = shared instance id
        e = by_id.setdefault(iid, {"label": o["label"],
                                   "deformability": o["deformability"], "contrib": {}})
        e["label"] = o.get("label") or e["label"]          # keep a non-empty label
        e["deformability"] = o.get("deformability", e["deformability"])
        for name, m in mi.items():
            e["contrib"].setdefault(o["state"], {}).setdefault(name, []).append(od / m["mask_file"])
    objects_out = {}
    for oid, e in by_id.items():
        masks = {}
        for state, frames in e["contrib"].items():
            fm = {}
            for name, paths in frames.items():
                fp = paths[0] if len(paths) == 1 else _composite_group_mask(oid, state, name, paths)
                if fp is not None:
                    fm[name] = os.path.relpath(fp, changes_root)
            if fm:
                masks[state] = fm
        in_pre, in_post = "pre" in masks, "post" in masks
        objects_out[oid] = {"label": e["label"], "deformability": e["deformability"],
                            "in_pre": in_pre, "in_post": in_post,
                            "change_type": _change_type(in_pre, in_post),  # derived (spec §4a)
                            "masks": masks}
    segments = {"scene": CFG["scene"], "tier": CFG.get("tier", "instance"), "camera": "cam0",
                "pre": CFG["states"]["pre"], "post": CFG["states"]["post"], "objects": objects_out}
    # per-frame binary change mask = union of object masks (the scored GT, spec §6/§7)
    try:
        segments["change_mask"] = CM.build_change_masks(
            CFG["capture"], geom_out=G.OUT, verbose=False,
            object_keys=[k for k, o in OBJECTS.items() if not o.get("ghost")])
    except Exception as e:
        print(f"warn: change_mask export failed: {e}", flush=True)
    out = Path(CFG["capture"]) / "changes" / "segments.json"
    json.dump(segments, open(out, "w"), indent=1)
    n_cf = sum(len(v) for v in segments.get("change_mask", {}).values())
    return jsonify(ok=True, path=str(out), n=len(objects_out), change_frames=n_cf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--tier", default="instance")
    ap.add_argument("--pre-session", default="aria_a_rgb")
    ap.add_argument("--pre-ref", default="navvis_a")
    ap.add_argument("--post-session", default="aria_b_rgb")
    ap.add_argument("--post-ref", default="navvis_b")
    ap.add_argument("--n", type=int, default=0,
                    help="frames to seed per object: 0 = every frame (thorough, default); "
                         "N>0 evenly subsamples N frames for a quick coarse pass")
    ap.add_argument("--min-vis", type=int, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    CFG.update(capture=args.capture, tier=args.tier,
               scene=args.scene or Path(args.capture).name,
               states={"pre": {"session": args.pre_session, "ref": args.pre_ref},
                       "post": {"session": args.post_session, "ref": args.post_ref}},
               seed={"n": args.n, "min_vis": args.min_vis})

    gobj = G.out_dir(args.capture) / "gui_objects.json"           # workspace-scoped (GEOM_OUT)
    legacy = Path(args.capture) / "changes" / "gui_objects.json"
    if not gobj.exists() and G.OUT == "changes/geom_sam_out" and legacy.exists():
        gobj = legacy                                             # migrate the default workspace
    if gobj.exists():
        try:
            for k, o in json.load(open(gobj)).items():
                o.pop("change_type", None)        # derived now, not stored (spec §4a)
                o.setdefault("id", k.split("__")[0])  # migrate old bare-id keys
                o.setdefault("instance", o["id"])     # editable link id (shared => moved)
                OBJECTS[f"{o['id']}__{o['state']}"] = o  # key by '<id>__<state>'
            print(f"loaded {len(OBJECTS)} object source(s) from {gobj.name}", flush=True)
        except Exception as e:
            print(f"warn: could not load {gobj}: {e}", flush=True)

    global MODEL, PROC
    print("loading SAM3 image model ...", flush=True)
    MODEL, PROC = G._load_sam_image()
    print(f"ready: scene={CFG['scene']}  ->  http://{args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
