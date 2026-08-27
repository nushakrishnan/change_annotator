"""Render pre/post change-overlay videos from saved GUI masks (SceneDiff-style).

Reads the per-object masks the GUI wrote under
``<capture>/changes/geom_sam_out/<id>__<state>/`` (``masks/`` + ``masks_index.json``)
and renders one overlay video per state: every frame of that state's camera walk,
with each annotated object filled in its own colour (consistent across *pre* and
*post*) at 50% opacity — matching the look of the SceneDiff dataset's
``video1.mp4`` / ``video2.mp4``.

No SAM / torch needed — just cv2 + numpy + ffmpeg on PATH.

Run (lamar_env):
  ~/lamar_env/bin/python make_change_videos.py \
      --capture /media/lamaria_indoor/captures/changes/cnb_e100
  # -> <capture>/changes/pre.mp4  and  <capture>/changes/post.mp4

Object id/label/state come from ``changes/gui_objects.json`` when present, else
they are parsed from the ``geom_sam_out/<id>__<state>`` directory names, so this
works whether or not you ran Export in the GUI.

Options: --fps, --rotate {0,90,180,270}, --alpha, --labels, --states, --out-dir.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# matplotlib "tab10" as RGB (0-1); index 7 darkened to match the old propagate_offline.py
TAB10 = [
    (0.122, 0.467, 0.706),   # 0 blue
    (1.000, 0.498, 0.055),   # 1 orange
    (0.173, 0.627, 0.173),   # 2 green
    (0.839, 0.153, 0.157),   # 3 red
    (0.580, 0.404, 0.741),   # 4 purple
    (0.549, 0.337, 0.294),   # 5 brown
    (0.890, 0.467, 0.761),   # 6 pink
    (0.249, 0.249, 0.249),   # 7 dark gray (darkened)
    (0.737, 0.741, 0.133),   # 8 olive
    (0.090, 0.745, 0.812),   # 9 cyan
]

OUT = "changes/geom_sam_out"      # mirrors geom_sam_prototype.OUT
FRAMES_SUBDIR = "raw_data"        # frames live at sessions/<session>/raw_data/<frame_name>


def load_objects(capture):
    """Return {key '<id>__<state>': {'id','label','state'}} from gui_objects.json,
    falling back to the geom_sam_out dir names."""
    gobj = Path(capture) / "changes" / "gui_objects.json"
    if gobj.exists():
        raw = json.load(open(gobj))
        out = {}
        for key, o in raw.items():
            oid = o.get("id", key.split("__")[0])
            state = o.get("state", key.split("__")[-1])
            out[key] = {"id": oid, "label": o.get("label", oid), "state": state}
        return out
    out = {}
    for d in sorted((Path(capture) / OUT).glob("*__*")):
        oid, _, state = d.name.rpartition("__")
        out[d.name] = {"id": oid, "label": oid, "state": state}
    return out


def assign_colors(objects):
    """One BGR (0-255) colour per object *id*, stable across states, by sorted id."""
    ids = sorted({o["id"] for o in objects.values()})
    colors = {}
    for i, oid in enumerate(ids):
        r, g, b = TAB10[i % len(TAB10)]
        colors[oid] = np.array([b, g, r]) * 255.0   # BGR for cv2
    return colors


def state_session(capture, keys):
    """The session a state's frames come from — read from any object's masks_index."""
    for key in keys:
        mi = Path(capture) / OUT / key / "masks_index.json"
        if mi.exists():
            idx = json.load(open(mi))
            sessions = {v.get("session") for v in idx.values() if v.get("session")}
            if sessions:
                return sorted(sessions)[0]
    return None


def frame_masks(capture, keys, colors, objects):
    """Build {frame_key: [(color_bgr, abs_mask_path, label), ...]} for a state's objects."""
    fm = defaultdict(list)
    for key in keys:
        od = Path(capture) / OUT / key
        mi = od / "masks_index.json"
        if not mi.exists():
            continue
        o = objects[key]
        color = colors[o["id"]]
        for frame_key, ent in json.load(open(mi)).items():
            fm[frame_key].append((color, od / ent["mask_file"], o["label"]))
    return fm


def open_ffmpeg(out_path, w, h, fps):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "-", "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        # guarantee even dimensions (libx264 / yuv420p requirement)
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def render_state(capture, state, keys, colors, objects, out_dir, fps, alpha, rotate, labels):
    session = state_session(capture, keys)
    if session is None:
        print(f"[{state}] no masks found for any object — skipping", flush=True)
        return None
    frame_dir = Path(capture) / "sessions" / session / FRAMES_SUBDIR / "images" / "cam0"
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        print(f"[{state}] no frames under {frame_dir} — skipping", flush=True)
        return None
    fm = frame_masks(capture, keys, colors, objects)

    probe = cv2.imread(str(frame_paths[0]))
    if rotate in _ROT:
        probe = cv2.rotate(probe, _ROT[rotate])
    h, w = probe.shape[:2]

    out_path = Path(out_dir) / f"{state}.mp4"
    proc = open_ffmpeg(out_path, w, h, fps)
    n_masked = 0
    for i, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if rotate in _ROT:
            bgr = cv2.rotate(bgr, _ROT[rotate])
        frame_key = f"images/cam0/{fp.name}"
        entries = fm.get(frame_key)
        if entries:
            n_masked += 1
            overlay = np.zeros_like(bgr, dtype=np.float32)
            visible = np.zeros((h, w), dtype=bool)
            for color, mask_path, _label in entries:
                m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if rotate in _ROT:
                    m = cv2.rotate(m, _ROT[rotate])
                mb = m > 127                       # masks are lossy JPGs -> threshold
                overlay[mb] = color                # last object wins on overlap
                visible |= mb
            bgr[visible] = np.clip(
                (1 - alpha) * bgr[visible] + alpha * overlay[visible], 0, 255
            ).astype(np.uint8)
            if labels:
                for color, mask_path, label in entries:
                    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    if rotate in _ROT:
                        m = cv2.rotate(m, _ROT[rotate])
                    ys, xs = np.where(m > 127)
                    if len(xs):
                        cx, cy = int(xs.mean()), int(ys.mean())
                        cv2.putText(bgr, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, (0, 0, 0), 5, cv2.LINE_AA)
                        cv2.putText(bgr, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.2, (255, 255, 255), 2, cv2.LINE_AA)
        proc.stdin.write(bgr.tobytes())
        if (i + 1) % 100 == 0 or i + 1 == len(frame_paths):
            print(f"[{state}] {i + 1}/{len(frame_paths)} frames "
                  f"({n_masked} with masks)", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"[{state}] wrote {out_path}  ({session}, {len(frame_paths)} frames, "
          f"{n_masked} masked)", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture dir (holds sessions/ and changes/)")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <capture>/changes)")
    ap.add_argument("--states", nargs="+", default=["pre", "post"], help="states to render")
    ap.add_argument("--fps", type=int, default=10, help="output fps (Aria RGB is ~10)")
    ap.add_argument("--alpha", type=float, default=0.5, help="mask fill opacity")
    ap.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                    help="rotate frames+masks clockwise before rendering")
    ap.add_argument("--labels", action="store_true", help="draw object labels at mask centroids")
    args = ap.parse_args()

    capture = args.capture
    out_dir = Path(args.out_dir) if args.out_dir else Path(capture) / "changes"
    out_dir.mkdir(parents=True, exist_ok=True)

    objects = load_objects(capture)
    if not objects:
        sys.exit(f"no objects found under {Path(capture) / OUT} or gui_objects.json")
    colors = assign_colors(objects)

    by_state = defaultdict(list)
    for key, o in objects.items():
        by_state[o["state"]].append(key)

    print("colour key:", {objects[k]["id"]: None for k in objects})  # ids in play
    for oid in sorted({o["id"] for o in objects.values()}):
        c = colors[oid]
        print(f"  {oid:12s} -> BGR{tuple(int(v) for v in c)}", flush=True)

    made = []
    for state in args.states:
        keys = by_state.get(state)
        if not keys:
            print(f"[{state}] no objects in this state — skipping", flush=True)
            continue
        p = render_state(capture, state, keys, colors, objects, out_dir,
                         args.fps, args.alpha, args.rotate, args.labels)
        if p:
            made.append(p)

    print("\ndone:", *[str(p) for p in made], sep="\n  " if made else " ")


if __name__ == "__main__":
    main()
