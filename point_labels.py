"""Point-canonical labels: the state's field clouds carry per-point object
labels, frames RENDER them — edit once in 2D, propagate in 3D (anusha's idea,
agreed 2026-08-07 for mass-change scenes under the trimap quality bar).

Substrate: the concatenation [static_<state>; changed_<state>] of the change
fields (voxel-uniform, already occlusion-projectable everywhere). Labels are a
parallel uint16 array (0 = unlabeled, else an index into labels_objects.json
mapping index -> object key). green->red is therefore just labeling STATIC
points (the poster class); ordinary objects label CHANGED points.

Lift = "which cloud points does this 2D stroke image?": project the cloud into
the frame, keep points that land inside the stroke AND pass the mesh occlusion
test. No mesh raycast — the cardboard_boxes decal lesson: rays through mesh
holes land on the wall behind; in-mask + occlusion selection cannot.

Grow = bounded 3D region-growing from the lifted seed (completes the surface
parts this frame can't see, e.g. a poster's margin) — ALWAYS previewed before
commit; a grow across a contact boundary is the 3D form of mask overflow.
"""
import json
from pathlib import Path

import numpy as np


def _paths(fields_dir, state):
    f = Path(fields_dir)
    return (f / f"static_{state}.npy", f / f"changed_{state}.npy",
            f / f"labels_{state}.npy", f / "labels_objects.json",
            f / f"labels_human_{state}.npy")


def load(fields_dir, state):
    """(P cloud, L_auto, L_human, n_static, registry).

    TWO label arrays, same tier discipline as the mask pipeline: `auto` is
    machine consensus (a rebuild may re-decide its own points), `human` is
    explicit 2D->3D edits (a rebuild must never erase them). Both reset to
    zeros if the fields were recomputed since — stale labels must not silently
    misalign with a different cloud."""
    sp, cp, lp, rp, hp = _paths(fields_dir, state)
    S = np.load(sp)
    C = np.load(cp)
    P = np.vstack([S, C]).astype(np.float64)
    def _arr(p):
        a = np.load(p) if p.exists() else np.zeros(len(P), np.uint16)
        return a if len(a) == len(P) else np.zeros(len(P), np.uint16)
    reg = json.load(open(rp)) if rp.exists() else {}
    return P, _arr(lp), _arr(hp), len(S), reg


def save(fields_dir, state, L_auto=None, L_human=None, reg=None):
    sp, cp, lp, rp, hp = _paths(fields_dir, state)
    if L_auto is not None:
        np.save(lp, L_auto.astype(np.uint16))
    if L_human is not None:
        np.save(hp, L_human.astype(np.uint16))
    if reg is not None:
        json.dump(reg, open(rp, "w"), indent=1)


def effective(L_auto, L_human):
    """Human labels win wherever they exist."""
    return np.where(L_human > 0, L_human, L_auto)


def label_index(reg, key):
    """Stable index for an object key (registered on first use)."""
    for k, v in reg.items():
        if v == key:
            return int(k)
    idx = max([int(k) for k in reg] or [0]) + 1
    reg[str(idx)] = key
    return idx


def grow(P, seed, radius=0.06, iters=3, cap_factor=6.0):
    """Bounded region-grow on the cloud from `seed` (bool mask). Stops at
    `iters` rings or when the selection exceeds cap_factor x seed size (a
    runaway grow means the stroke leaked onto a connected structure — the
    preview lets the human catch it, the cap keeps the preview finite)."""
    from scipy.spatial import cKDTree
    sel = seed.copy()
    n0 = max(1, int(seed.sum()))
    tree = cKDTree(P)
    frontier = P[sel]
    for _ in range(iters):
        if not len(frontier) or sel.sum() > cap_factor * n0:
            break
        hits = tree.query_ball_point(frontier, radius, workers=-1)
        flat = sorted({i for lst in hits for i in lst})
        add = np.zeros(len(P), bool)
        add[flat] = True
        add &= ~sel
        if not add.any():
            break
        sel |= add
        frontier = P[add]
    return sel
