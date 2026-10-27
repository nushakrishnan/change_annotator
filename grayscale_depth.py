"""One editable invalid-depth object on a sequence's rectified-left PNG grid.

Pipeline masks are read-only. Export publishes a complete corrected set in the
sequence's grayscale_mask_export/ folder. Blank masks are valid annotations.
"""
import base64
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Response, jsonify, request
from grayscale_exclusions import excluded_keyframes


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.fchmod(f.fileno(), stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o664)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _png(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Could not encode mask PNG")
    return data.tobytes()


class GrayscaleDepth:
    object_id = "invalid_depth__pre"

    def __init__(self, sequence):
        self.sequence = Path(sequence).expanduser().resolve()
        self.stereo = self.sequence / "stereo"
        self.mask_dir = self.stereo / "masks"
        self.export_dir = self.sequence / "grayscale_mask_export"
        self.draft_dir = self.stereo / ".grayscale_annotator"
        self.lock = threading.RLock()
        left = self.stereo / "left"
        paths = sorted(left.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdecimal() else -1)
        self.images = {f"left/{p.name}": p for p in paths if p.stem.isdecimal()}
        if not self.images:
            raise ValueError(f"No timestamped keyframe PNGs in {left}")
        if not self.mask_dir.is_dir():
            raise ValueError(f"Projected mask folder not found: {self.mask_dir}")
        self.excluded = excluded_keyframes(self.sequence, [int(p.stem) for p in self.images.values()])
        self.shapes = {}
        # Validate the exact image/mask grids before allowing any edits. No resize.
        for name, path in self.images.items():
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot read keyframe: {path}")
            self.shapes[name] = image.shape
            self.mask(name)
        self.object = dict(id="invalid_depth", instance="invalid_depth", state="pre",
                           label="Invalid depth", deformability="rigid")

    def frame_path(self, name):
        if name not in self.images:
            raise ValueError("Unknown grayscale keyframe")
        return self.images[name]

    def source_path(self, name, suffix="masked"):
        return self.mask_dir / f"{self.frame_path(name).stem}.{suffix}.png"

    def draft_path(self, name):
        return self.draft_dir / f"{self.frame_path(name).stem}.npz"

    def export_path(self, name):
        return self.export_dir / f"{self.frame_path(name).stem}.masked.png"

    def saved_path(self, name):
        exported = self.export_path(name)
        return exported if exported.exists() else self.source_path(name)

    def _read_draft(self, name):
        with np.load(self.draft_path(name), allow_pickle=False) as data:
            mask = data["mask"].astype(bool)
            baseline = json.loads(str(data["baseline"].item()))
        self._check_shape(name, mask)
        return mask, baseline

    def _check_shape(self, name, mask):
        if mask.shape != self.shapes[name]:
            raise ValueError(f"Mask size {mask.shape} does not match {name}: {self.shapes[name]}")

    def is_excluded(self, name):
        return int(self.frame_path(name).stem) in self.excluded

    def source_mask(self, name, suffix="masked"):
        path = self.source_path(name, suffix)
        if not path.exists():
            return np.zeros(self.shapes[name], bool)
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {path}")
        self._check_shape(name, mask)
        return mask > 127

    def decode_canvas(self, name, data_url):
        self.frame_path(name)
        try:
            raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
            rgba = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        except (ValueError, IndexError, cv2.error) as e:
            raise ValueError("Invalid canvas mask PNG") from e
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            raise ValueError("Expected an RGBA canvas mask")
        mask = rgba[:, :, 3] > 0
        self._check_shape(name, mask)
        return mask

    def mask(self, name):
        if self.is_excluded(name):
            return np.ones(self.shapes[name], bool)
        source = self.saved_path(name)
        if self.draft_path(name).exists():
            return self._read_draft(name)[0]
        if not source.exists():
            return np.zeros(self.shapes[name], bool)
        mask = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Cannot read mask: {source}")
        self._check_shape(name, mask)
        return mask > 127

    def pending(self):
        return [name for name in self.images if self.draft_path(name).exists()]

    def export_delta(self, name, mask):
        previous_delta = self.export_dir / f"{self.frame_path(name).stem}.delta.npz"
        if previous_delta.exists() and self.export_path(name).exists():
            with np.load(previous_delta, allow_pickle=False) as data:
                added, removed = data["added"].astype(bool), data["removed"].astype(bool)
            self._check_shape(name, added); self._check_shape(name, removed)
            if (added & removed).any():
                raise ValueError(f"Conflicting add/remove delta: {previous_delta}")
            if not self.is_excluded(name):
                previous = cv2.imread(str(self.export_path(name)), cv2.IMREAD_UNCHANGED)
                if previous is None:
                    raise ValueError(f"Cannot read exported mask: {self.export_path(name)}")
                self._check_shape(name, previous)
                plus, minus = mask & ~(previous > 127), (previous > 127) & ~mask
                added, removed = (added | plus) & ~minus, (removed | minus) & ~plus
            return added, removed
        if self.is_excluded(name):
            return np.zeros_like(mask), np.zeros_like(mask)
        base = self.source_mask(name)
        return mask & ~base, base & ~mask

    def save(self, name, mask):
        with self.lock:
            self.frame_path(name)
            self._check_shape(name, mask)
            if self.is_excluded(name):
                if not mask.all():
                    raise ValueError("Scene-change-excluded frames must remain whole-image masked")
                return
            draft = self.draft_path(name)
            if draft.exists():
                baseline = self._read_draft(name)[1]
            else:
                baseline = {"mask": _digest(self.saved_path(name))}
            data = io.BytesIO()
            np.savez_compressed(data, mask=mask.astype(bool), baseline=json.dumps(baseline))
            _atomic_write(draft, data.getvalue())

    def export(self):
        with self.lock:
            pending = set(self.pending())
            # Build the complete set before publishing it, so an encoding/write
            # failure leaves the previous export and every draft intact.
            stage = Path(tempfile.mkdtemp(prefix=".grayscale_mask_export-", dir=self.sequence))
            backup = stage.with_name(stage.name + ".previous")
            try:
                os.chmod(stage, stat.S_IMODE(self.export_dir.stat().st_mode)
                         if self.export_dir.exists() else 0o2775)
                frames, deltas = {}, {}
                for name in self.images:
                    mask = self.mask(name)
                    png = _png(mask.astype(np.uint8) * 255)
                    if name in pending and not self.is_excluded(name):
                        _, baseline = self._read_draft(name)
                        # "masked" also accepts drafts from the original version
                        # that exported over the projection masks.
                        expected = baseline.get("mask", baseline.get("masked"))
                        if _digest(self.saved_path(name)) not in (expected, hashlib.sha256(png).hexdigest()):
                            raise ValueError(f"Saved mask changed since this draft was saved: {self.saved_path(name)}. "
                                             "Reconcile the draft before exporting.")
                    stem = self.frame_path(name).stem
                    for suffix in ("masked", "invalid"):
                        _atomic_write(stage / f"{stem}.{suffix}.png", png)
                    strict = self.source_mask(name, "invalid_strict") | mask
                    _atomic_write(stage / f"{stem}.invalid_strict.png", _png(strict.astype(np.uint8) * 255))
                    # Compose this edit's +/- pixels with previous human deltas;
                    # re-exporting after reprojection must not erase an override
                    # just because the new base now happens to agree with it.
                    added, removed = self.export_delta(name, mask)
                    delta = io.BytesIO()
                    np.savez_compressed(delta, added=added, removed=removed)
                    deltas[name] = f"{stem}.delta.npz"
                    _atomic_write(stage / deltas[name], delta.getvalue())
                    frames[name] = f"{stem}.masked.png"
                manifest = dict(sequence=self.sequence.name, camera="rectified_left", frames=frames,
                                schema_version=2, delta_files=deltas, excluded_frames=sorted(self.excluded),
                                original_masks=str(self.mask_dir), mask_values={"0": "clear", "255": "invalid depth"},
                                missing_projected_masks=[name for name in self.images if not self.source_path(name).exists()])
                _atomic_write(stage / "manifest.json", json.dumps(manifest, indent=2).encode())
                if self.export_dir.exists():
                    os.replace(self.export_dir, backup)
                try:
                    os.replace(stage, self.export_dir)
                except OSError:
                    # Keep the backup on disk if restoration itself fails.
                    if backup.exists():
                        os.replace(backup, self.export_dir)
                    raise
            except OSError as e:
                raise OSError(f"Export failed; drafts kept. {e}") from e
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
            # Publication succeeded. Remove drafts only after all masks and the
            # manifest are in place. A retry accepts an already published draft.
            for name in pending:
                self.draft_path(name).unlink()
            if backup.exists():
                shutil.rmtree(backup)
            return len(self.images), len(pending)

    def frames(self):
        return [dict(name=name, session=self.sequence.name, has_mask=True,
                     excluded=self.is_excluded(name),
                     source_missing=not self.saved_path(name).exists(),
                     draft=self.draft_path(name).exists(), px=int(self.mask(name).sum()))
                for name in self.images]

    def dispatch(self, sam_enabled=True):
        """Adapt the existing editor API and reject unrelated RGB/3D writers."""
        passthrough = {"index", "frame", "static", "api_click", "api_edit_click", "api_refine", "api_multi_smart"}
        if request.endpoint in passthrough:
            if request.endpoint in {"api_click", "api_edit_click", "api_refine", "api_multi_smart"} and not sam_enabled:
                return jsonify(error="SAM is disabled (--no-sam); use the manual brushes."), 400
            if request.endpoint in {"frame", "api_click", "api_edit_click", "api_refine", "api_multi_smart"}:
                data = request.args if request.method == "GET" else request.get_json()
                try:
                    self.frame_path(data.get("name", data.get("frame")))
                    if request.endpoint in {"api_click", "api_edit_click", "api_refine"} and self.is_excluded(data["frame"]):
                        raise ValueError("Scene-change-excluded frame: editing is disabled")
                    if data.get("state", "pre") != "pre":
                        raise ValueError("Only the rectified-left sequence is loaded")
                    if "id" in data and data["id"] != self.object_id:
                        raise ValueError("Unknown grayscale object")
                except ValueError as e:
                    return jsonify(error=str(e)), 400
            return None
        try:
            with self.lock:
                return self._dispatch()
        except (ValueError, KeyError, TypeError) as e:
            return jsonify(error=str(e)), 400
        except OSError as e:
            return jsonify(error=str(e)), 500

    def _dispatch(self):
        endpoint = request.endpoint
        allowed = {"api_frames": "GET", "api_objects": "GET", "api_object_frames": "GET",
                   "api_overlay": "GET", "api_mask_png": "GET", "api_save_edit": "POST",
                   "api_export": "POST"}
        if endpoint not in allowed or request.method != allowed[endpoint]:
            return jsonify(error="This action is unavailable in grayscale mask review."), 400
        if endpoint == "api_frames":
            return jsonify(list(self.images))
        if endpoint == "api_objects":
            return jsonify(objects={self.object_id: dict(self.object, n_masks=len(self.images), n_seeds=0)},
                           pending=len(self.pending()))
        if endpoint == "api_export":
            n, updated = self.export()
            return jsonify(ok=True, n=n, updated=updated, pending=len(self.pending()), path=str(self.export_dir),
                           msg=f"Exported all {n} frame masks ({updated} updated) to {self.export_dir}")
        data = request.args if request.method == "GET" else request.get_json()
        if data["id"] != self.object_id:
            raise ValueError("Unknown grayscale object")
        if endpoint == "api_object_frames":
            frames = self.frames()
            return jsonify(frames=frames, count=len(frames), masked=len(frames), state="pre")
        name = data.get("name", data.get("frame"))
        self.frame_path(name)
        if endpoint == "api_save_edit":
            mask = self.decode_canvas(name, data["mask"])
            self.save(name, mask)
            return jsonify(ok=True, px=int(mask.sum()), pending=len(self.pending()), draft=True)
        mask = self.mask(name)
        if endpoint == "api_mask_png":
            image = np.zeros((*mask.shape, 4), np.uint8)
            image[mask] = (0, 0, 255, 130)
        else:
            image = cv2.imread(str(self.frame_path(name)))
            image[mask] = (0.45 * image[mask] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        return Response(_png(image), mimetype="image/png", headers={"Cache-Control": "no-store"})
