"""Read scene-change exclusions without a capture, scan, or pipeline imports."""
import json
from pathlib import Path

import numpy as np


def excluded_keyframes(sequence, timestamps):
    sequence = Path(sequence)
    path = sequence / "changes/changes.json"
    if not path.exists():
        return set()
    changes = json.loads(path.read_text())
    excluded = set(changes.get("stereo", []))
    rgb_excluded = set(changes.get("rgb", []))
    rgb_path = sequence / "trajectory/rgb_frames.json"
    windows = changes.get("rgb_exclusion_windows")
    if rgb_path.exists():
        # Same contiguous-RGB-run rule as depth_gt.changes.rgb_exclusion_windows.
        rgb = np.unique([f["timestamp_ns"] for f in json.loads(rgb_path.read_text())])
        indices = np.flatnonzero(np.isin(rgb, list(rgb_excluded)))
        runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        windows = [[int(rgb[run[0]]), int(rgb[run[-1]])] for run in runs if len(run)]
    elif windows is None:
        # Legacy changes.json also stores the crossing windows; don't merge
        # separate crossings just because only excluded RGB frames are listed.
        windows = []
        for start, stop in changes.get("windows", []):
            inside = [t for t in rgb_excluded if start <= t <= stop]
            if inside:
                windows.append([min(inside), max(inside)])
        if rgb_excluded and not windows:
            raise ValueError("Cannot resolve RGB exclusion windows: missing RGB trajectory/window metadata")
    stamps = np.unique(np.asarray(timestamps, np.int64))
    for start, stop in windows:
        left = max(0, int(np.searchsorted(stamps, start, side="left")) - 1)
        right = min(len(stamps), int(np.searchsorted(stamps, stop, side="right")) + 1)
        excluded.update(map(int, stamps[left:right]))
    return excluded
