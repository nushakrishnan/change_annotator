"""Minimal click-to-propagate annotation GUI (RGB, single-camera).

A thin Flask front-end on top of the geometry-assisted propagation pipeline
(`geom_sam_prototype.py`). Runs in sam3_env and holds one SAM3 *image* model for
live click previews; the propagation step shells out to lamar_env for geometry
(`seeds`) and finishes the per-frame masks in-process with the same model.

  click point(s) on a source frame  -> SAM3 image mask preview (live)
  add object (label / change_type)   -> saves src_mask.png under geom_sam_out/<id>__<state>/
  propagate                          -> seeds (lamar_env) + per-frame SAM masks (within-walk)
  export                             -> changes/segments.json   (annotation_spec.md §6)

Run (sam3_env):
  ~/sam3_env/bin/python gui.py --capture /media/lamaria_indoor/captures/changes/cnb_e100
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
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file, Response

import geom_sam_prototype as G

LAMAR_PY = Path(os.environ.get("LAMAR_PY", Path.home() / "lamar_env/bin/python"))
LAMAR_PYTHONPATH = os.environ.get("LAMAR_PYTHONPATH", str(Path.home() / "repos/lamaria-indoor"))
VIZ_ROOT = Path(os.environ.get("VIZ_ROOT", "/media/lamaria_indoor/annotation_exploration"))

app = Flask(__name__)

# ── runtime state (single user) ──
CFG = {}              # capture / states / seed params / scene / tier
MODEL = PROC = None
LOCK = threading.Lock()
PENDING = {}          # last click preview: {state, frame, points, neg, mask}
EDIT = {}             # pending review-mode re-segment: {id, frame, session, mask}
OBJECTS = {}          # id -> {label, change_type, deformability, state, frame, points}
JOBS = {}             # job_id -> {status, ...}


def _png_b64(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def _frame_path(state, name):
    s = CFG["states"][state]["session"]
    return Path(CFG["capture"]) / "sessions" / s / "raw_data" / name


def _save_working():
    json.dump(OBJECTS, open(Path(CFG["capture"]) / "changes" / "gui_objects.json", "w"), indent=1)


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


# ───────────────────────────── pages / static ─────────────────────────────
@app.route("/")
def index():
    return render_template("gui.html", scene=CFG["scene"],
                           seed_n=CFG["seed"]["n"], seed_min_vis=CFG["seed"]["min_vis"])


@app.route("/api/frames")
def api_frames():
    s = CFG["states"][request.args["state"]]["session"]
    d = Path(CFG["capture"]) / "sessions" / s / "raw_data" / "images" / "cam0"
    return jsonify(sorted(f"images/cam0/{p.name}" for p in d.glob("*.jpg")))


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
    cv2.imwrite(str(od / "src_mask.png"), (mask * 255).astype(np.uint8))
    OBJECTS[key] = {"id": oid, "label": d.get("label") or oid,
                    "deformability": d.get("deformability", "rigid"),
                    "state": state, "frame": frame, "points": points}
    PENDING.clear()
    _save_working()
    return jsonify(ok=True)


# ───────────────────────────── propagate (job) ─────────────────────────────
@app.route("/api/propagate", methods=["POST"])
def api_propagate():
    oid = request.get_json()["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "obj": oid}
    threading.Thread(target=_propagate_job, args=(job_id, oid), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    return jsonify(JOBS.get(job_id, {"status": "unknown"}))


def _propagate_job(job_id, oid):
    try:
        o = OBJECTS[oid]
        state = o["state"]
        objdir = oid  # OBJECTS is keyed by '<id>__<state>' = the objdir
        st = CFG["states"][state]
        # 1) geometry seeds (lamar_env subprocess; within-source walk only)
        env = dict(os.environ, PYTHONPATH=LAMAR_PYTHONPATH)
        subprocess.run([str(LAMAR_PY), str(G.__file__), "seeds",
                        "--capture", CFG["capture"], "--session", st["session"],
                        "--ref", st["ref"], "--src-name", o["frame"], "--obj", objdir,
                        "--n", str(CFG["seed"]["n"]), "--min-vis", str(CFG["seed"]["min_vis"]),
                        "--no-cross"], check=True, env=env)
        # 2) per-frame SAM masks (in-process, shared model)
        n = _perframe_inproc(objdir, st["session"])
        # 3) collect viz for review
        od = G.out_dir(CFG["capture"], objdir)
        viz = VIZ_ROOT / CFG["scene"] / objdir
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
    mask_index, tiles = {}, []
    for name, s in seeds.items():
        sid = s.get("session", default_session)
        bgr = cv2.imread(str(cap / "sessions" / sid / "raw_data" / name))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with LOCK:
            masks, _ = G._segment(MODEL, PROC, img, points=s["points"], labels=s["labels"],
                                  box=s["box"], multimask=False)
        mask = masks[0].astype(bool)
        flat = name.replace("/", "_")
        cv2.imwrite(str(masks_dir / flat), (mask * 255).astype(np.uint8))
        mask_index[name] = {"session": sid, "mask_file": f"masks/{flat}", "px": int(mask.sum())}
        ov = bgr.copy()
        ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        x0, y0, x1, y1 = [int(v) for v in s["box"]]
        cv2.rectangle(ov, (x0, y0), (x1, y1), (0, 255, 0), 1)
        tiles.append(cv2.resize(ov, (242, 242)))
    json.dump(mask_index, open(od / "masks_index.json", "w"), indent=1)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros((242, 242, 3), np.uint8))
        rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(od / "result_contact.png"), np.concatenate(rows, 0))
    return len(mask_index)


# ───────────────────────── refine: SAM3 video per-span ─────────────────────────
@app.route("/api/refine", methods=["POST"])
def api_refine():
    """Refine an object's geom masks with the SAM3 video tracker (per visible span)."""
    oid = request.get_json()["id"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    if not (G.out_dir(CFG["capture"], _objdir(oid)) / "masks_index.json").exists():
        return jsonify(error="propagate first"), 400
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "obj": oid}
    threading.Thread(target=_refine_job, args=(job_id, oid), daemon=True).start()
    return jsonify(job_id=job_id)


def _refine_job(job_id, oid):
    try:
        objdir = _objdir(oid)
        sess = CFG["states"][OBJECTS[oid]["state"]]["session"]
        # separate process so the SAM3 *video* model's VRAM is released on exit
        # (the GUI process already holds the SAM3 *image* model for live clicks)
        subprocess.run([sys.executable, str(G.__file__), "track",
                        "--capture", CFG["capture"], "--obj", objdir,
                        "--session", sess], check=True)
        mi = G.out_dir(CFG["capture"], objdir) / "masks_index.json"
        n = len(json.load(open(mi))) if mi.exists() else 0
        JOBS[job_id].update(status="done", n_masks=n)
    except subprocess.CalledProcessError as e:
        JOBS[job_id].update(status="error", error=f"video refine failed (rc={e.returncode})")
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e))


# ───────────────────────── review / edit propagated masks ─────────────────────────
@app.route("/api/objects")
def api_objects():
    """All objects + how many propagated frames each has (drives the sidebar list)."""
    states_by_id = {}
    for o in OBJECTS.values():
        states_by_id.setdefault(o["id"], set()).add(o["state"])
    out = {}
    for key, o in OBJECTS.items():
        mi = G.out_dir(CFG["capture"], key) / "masks_index.json"
        n = 0
        if mi.exists():
            try:
                n = len(json.load(open(mi)))
            except Exception:
                n = 0
        sts = states_by_id[o["id"]]            # change type reflects the id across both states
        in_pre, in_post = "pre" in sts, "post" in sts
        out[key] = {**o, "n_masks": n, "in_pre": in_pre, "in_post": in_post,
                    "change_type": _change_type(in_pre, in_post)}
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
    names = sorted(f"images/cam0/{p.name}" for p in fdir.glob("*.jpg"))
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
        mask = cv2.imread(str(od / m["mask_file"]), 0) > 127
        bgr = _overlay_mask(bgr, mask)
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
        mi[name] = {"session": sid, "mask_file": f"masks/{flat}", "px": int(mask.sum())}
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
    if m:
        mask = cv2.imread(str(od / m["mask_file"]), 0) > 127
    else:
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
    """Drop one frame's mask from the object (bad propagation)."""
    d = request.get_json()
    oid, name = d["id"], d["frame"]
    if oid not in OBJECTS:
        return jsonify(error="unknown object"), 404
    od, mi_path, mi, m = _mask_entry(oid, name)
    if m:
        f = od / m["mask_file"]
        if f.exists():
            f.unlink()
        mi.pop(name, None)
        json.dump(mi, open(mi_path, "w"), indent=1)
    return jsonify(ok=True, count=len(mi))


# ───────────────────────────── export ─────────────────────────────
@app.route("/api/export", methods=["POST"])
def api_export():
    # merge the per-state entries of each object id (pre + post = a moved object)
    by_id = {}
    for key, o in OBJECTS.items():
        mi_path = G.out_dir(CFG["capture"], key) / "masks_index.json"
        state_masks = ({name: m["mask_file"] for name, m in json.load(open(mi_path)).items()}
                       if mi_path.exists() else {})
        e = by_id.setdefault(o["id"], {"label": o["label"],
                                       "deformability": o["deformability"], "masks": {}})
        e["label"], e["deformability"] = o["label"], o["deformability"]
        if state_masks:
            e["masks"][o["state"]] = state_masks
    objects_out = {}
    for oid, e in by_id.items():
        in_pre, in_post = "pre" in e["masks"], "post" in e["masks"]
        objects_out[oid] = {"label": e["label"], "deformability": e["deformability"],
                            "in_pre": in_pre, "in_post": in_post,
                            "change_type": _change_type(in_pre, in_post),  # derived (spec §4a)
                            "masks": e["masks"]}
    segments = {"scene": CFG["scene"], "tier": CFG.get("tier", "instance"), "camera": "cam0",
                "pre": CFG["states"]["pre"], "post": CFG["states"]["post"], "objects": objects_out}
    out = Path(CFG["capture"]) / "changes" / "segments.json"
    json.dump(segments, open(out, "w"), indent=1)
    return jsonify(ok=True, path=str(out), n=len(objects_out))


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

    gobj = Path(args.capture) / "changes" / "gui_objects.json"
    if gobj.exists():
        try:
            for k, o in json.load(open(gobj)).items():
                o.pop("change_type", None)        # derived now, not stored (spec §4a)
                o.setdefault("id", k.split("__")[0])  # migrate old bare-id keys
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
