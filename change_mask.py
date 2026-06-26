"""Derive per-frame binary change masks (union of object masks) for evaluation.

The instance pipeline (geom_sam_prototype + gui.py) produces per-OBJECT per-frame
masks. The ground truth a change-detection method (e.g. SceneDiff, MV3DCD) is
actually scored against is the per-frame UNION over all objects -- the binary
"did this pixel change" map (annotation_spec.md §6/§7). This module derives it.

Invariant (spec §6): change_mask[state][frame] == OR over objects of masks[state][frame].

Source of truth is each object's `masks_index.json` under the geom-out root
(`<geom_out>/<id>__<state>/masks_index.json`, frame -> {mask_file, session, px}),
so this runs with no SAM / scantools / segments.json dependency -- just cv2+numpy.

Outputs, under `<capture>/changes/`:
  change_mask/<state>/<flat>.png   binary union per frame (255 = changed)
  change_mask_index.json           {state: {frame: {mask_file, session, px}}}

Standalone:
  python change_mask.py --capture /path/to/captures/changes/cnb_e100
If a segments.json exists, a top-level "change_mask" block ({state: {frame: relpath}},
relpath relative to the `changes/` dir where segments.json lives) is added to it.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# geom-out root relative to the capture (mirrors geom_sam_prototype.OUT / gui G.OUT)
GEOM_OUT = "changes/geom_sam_out"


def _flat_png(frame):
    """'images/cam0/1668..jpg' -> 'images_cam0_1668...png' (matches the per-object
    flat naming; forced to lossless PNG since this is scored GT)."""
    return Path(frame.replace("/", "_")).with_suffix(".png").name


def build_change_masks(capture, geom_out=GEOM_OUT, object_keys=None, verbose=True):
    """Union per-object masks into one binary change mask per (state, frame).

    object_keys: explicit '<id>__<state>' dirs to include (e.g. the currently
    exported set); if None, every '<id>__<state>' dir under <capture>/<geom_out>.

    Writes PNGs + change_mask_index.json under <capture>/changes/ and returns the
    spec §6 block: {state: {frame_name: '<relpath under changes/>'}}.
    """
    capture = Path(capture)
    root = capture / geom_out
    if object_keys is None:
        object_keys = sorted(p.name for p in root.iterdir()
                             if p.is_dir() and "__" in p.name)

    # state -> frame -> accumulated boolean union (+ provenance for the index)
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
            else:
                if mb.shape != acc.shape:  # frames share a camera; should not happen
                    mb = cv2.resize(mb.astype(np.uint8), (acc.shape[1], acc.shape[0]),
                                    interpolation=cv2.INTER_NEAREST) > 0
                    if verbose:
                        print(f"  warn {key}/{frame}: shape mismatch, resized")
                unions[state][frame] = acc | mb

    out_root = capture / "changes" / "change_mask"
    block, index = {}, {}
    for state, frames in unions.items():
        (out_root / state).mkdir(parents=True, exist_ok=True)
        for frame, mask in frames.items():
            fname = _flat_png(frame)
            cv2.imwrite(str(out_root / state / fname), (mask.astype(np.uint8) * 255))
            rel = f"change_mask/{state}/{fname}"
            block.setdefault(state, {})[frame] = rel
            index.setdefault(state, {})[frame] = {
                "mask_file": rel, "session": prov[state][frame]["session"],
                "px": int(mask.sum())}

    json.dump(index, open(capture / "changes" / "change_mask_index.json", "w"), indent=1)
    if verbose:
        for state, frames in block.items():
            print(f"  {state}: {len(frames)} change frames")
    return block


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture dir holding changes/")
    ap.add_argument("--geom-out", default=GEOM_OUT,
                    help=f"geom-out root relative to capture (default {GEOM_OUT})")
    args = ap.parse_args()

    block = build_change_masks(args.capture, geom_out=args.geom_out)

    seg_path = Path(args.capture) / "changes" / "segments.json"
    if seg_path.exists():
        seg = json.load(open(seg_path))
        seg["change_mask"] = block  # spec §6: {state: {frame: relpath under changes/}}
        json.dump(seg, open(seg_path, "w"), indent=1)
        print(f"updated {seg_path} with change_mask block")
    total = sum(len(v) for v in block.values())
    print(f"done: {total} change frames -> {Path(args.capture)/'changes'/'change_mask'}")


if __name__ == "__main__":
    main()
