"""Behavioral checks for complete exports and read-only sources; no GPU."""
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import gui
from grayscale_depth import GrayscaleDepth


class GrayscaleDepthTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sequence = Path(self.tmp.name) / "sequence"
        self.left = self.sequence / "stereo" / "left"
        self.masks = self.sequence / "stereo" / "masks"
        self.left.mkdir(parents=True)
        self.masks.mkdir()
        self.original = np.zeros((32, 40), np.uint8)
        self.original[2:6, 3:9] = 255
        self.original[20:25, 30:35] = 255
        for ts in (100, 9, 10):
            cv2.imwrite(str(self.left / f"{ts}.png"), np.full((32, 40), 80, np.uint8))
        for ts in (9, 10):
            for suffix in ("masked", "invalid", "uncovered", "invalid_strict"):
                cv2.imwrite(str(self.masks / f"{ts}.{suffix}.png"), self.original)
            np.savez(self.masks / f"{ts}.npz", label=self.original)
        (self.masks / "manifest.json").write_text(json.dumps({"projection": "test"}))
        self.before = self.snapshot()
        self.ws = gui.configure_grayscale(self.sequence, no_sam=True)
        gui.app.config["TESTING"] = True
        self.client = gui.app.test_client()
        self.oid = self.ws.object_id
        self.addCleanup(gui.CFG.clear)
        self.addCleanup(gui.OBJECTS.clear)

    def snapshot(self):
        return {str(p.relative_to(self.sequence)): p.read_bytes()
                for p in self.sequence.rglob("*") if p.is_file()}

    def assert_originals_unchanged(self):
        for name, content in self.before.items():
            self.assertEqual((self.sequence / name).read_bytes(), content, name)

    def export_snapshot(self):
        return {p.name: p.read_bytes() for p in self.ws.export_dir.iterdir()}

    def save(self, mask, name="left/9.png"):
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        rgba[mask.astype(bool)] = (0, 0, 255, 130)
        ok, png = cv2.imencode(".png", rgba)
        self.assertTrue(ok)
        return self.client.post("/api/save_edit", json={"id": self.oid, "frame": name,
                                "mask": "data:image/png;base64," + base64.b64encode(png).decode()})

    def test_loading_is_read_only_single_object_numeric_order(self):
        objects = self.client.get("/api/objects").json["objects"]
        self.assertEqual(list(objects), [self.oid])
        result = self.client.get("/api/object_frames", query_string={"id": self.oid}).json
        self.assertEqual([f["name"] for f in result["frames"]], ["left/9.png", "left/10.png", "left/100.png"])
        self.assertEqual(result["frames"][0]["px"], int((self.original > 0).sum()))
        self.assertTrue(result["frames"][2]["source_missing"])
        for path in ("/api/mask_png", "/api/overlay"):
            r = self.client.get(path, query_string={"id": self.oid, "name": "left/9.png"})
            self.assertEqual(r.status_code, 200)
        self.assertEqual(self.snapshot(), self.before)
        self.assertFalse(self.ws.draft_dir.exists())
        self.assertFalse(self.ws.export_dir.exists())

    def test_drafts_resume_without_changing_sources_and_export_empty(self):
        empty = np.zeros_like(self.original, dtype=bool)
        self.assertEqual(self.save(empty).status_code, 200)
        self.assert_originals_unchanged()
        self.assertFalse(self.ws.export_dir.exists())
        resumed = GrayscaleDepth(self.sequence)
        self.assertFalse(resumed.mask("left/9.png").any())
        self.assertEqual(resumed.pending(), ["left/9.png"])
        r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["n"], 3)
        self.assertEqual(r.json["updated"], 1)
        self.assertEqual(r.json["path"], str(self.sequence / "grayscale_mask_export"))
        for suffix in ("masked", "invalid"):
            mask = cv2.imread(str(self.ws.export_dir / f"9.{suffix}.png"), 0)
            self.assertEqual(mask.shape, self.original.shape)
            self.assertFalse(mask.any())
        self.assert_originals_unchanged()
        self.assertEqual(resumed.pending(), [])
        self.assertFalse(GrayscaleDepth(self.sequence).mask("left/9.png").any())
        after = self.snapshot()
        self.assertEqual(self.client.post("/api/export").json["updated"], 0)
        self.assertEqual(self.snapshot(), after)

    def test_add_and_erase_pixels_are_exact_and_missing_mask_can_be_created(self):
        corrected = self.original > 0
        corrected[3, 4] = False
        corrected[15, 19] = True
        self.assertEqual(self.save(corrected).status_code, 200)
        self.assertEqual(self.save(corrected, "left/100.png").status_code, 200)
        r = self.client.post("/api/export")
        self.assertEqual(r.json["n"], 3)
        self.assertEqual(r.json["updated"], 2)
        for ts in (9, 100):
            mask = cv2.imread(str(self.ws.export_dir / f"{ts}.masked.png"), 0)
            np.testing.assert_array_equal(mask, corrected.astype(np.uint8) * 255)
            np.testing.assert_array_equal(mask, cv2.imread(str(self.ws.export_dir / f"{ts}.invalid.png"), 0))
        np.testing.assert_array_equal(cv2.imread(str(self.ws.export_dir / "10.masked.png"), 0), self.original)
        self.assert_originals_unchanged()

    def test_first_export_without_edits_is_complete_and_records_missing_masks(self):
        r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json["n"], 3)
        self.assertEqual(r.json["updated"], 0)
        self.assertEqual(len(list(self.ws.export_dir.glob("*.png"))), 9)
        np.testing.assert_array_equal(cv2.imread(str(self.ws.export_path("left/9.png")), 0), self.original)
        self.assertFalse(cv2.imread(str(self.ws.export_path("left/100.png")), 0).any())
        manifest = json.loads((self.ws.export_dir / "manifest.json").read_text())
        self.assertEqual(manifest["frames"], {f"left/{ts}.png": f"{ts}.masked.png" for ts in (9, 10, 100)})
        self.assertEqual(manifest["missing_projected_masks"], ["left/100.png"])
        self.assert_originals_unchanged()

    def test_export_deltas_and_strict_mask_superset(self):
        corrected = self.original > 0
        corrected[3, 4] = False; corrected[15, 19] = True
        self.save(corrected)
        self.assertEqual(self.client.post('/api/export').status_code, 200)
        for stem in ('9', '10', '100'):
            masked = cv2.imread(str(self.ws.export_dir / f'{stem}.masked.png'), 0) > 0
            invalid = cv2.imread(str(self.ws.export_dir / f'{stem}.invalid.png'), 0) > 0
            strict = cv2.imread(str(self.ws.export_dir / f'{stem}.invalid_strict.png'), 0) > 0
            np.testing.assert_array_equal(masked, invalid)
            self.assertTrue(strict[masked].all())
            base = self.ws.source_mask(f'left/{stem}.png')
            with np.load(self.ws.export_dir / f'{stem}.delta.npz') as delta:
                np.testing.assert_array_equal((base | delta['added']) & ~delta['removed'], masked)
        self.assert_originals_unchanged()

    def test_reexport_preserves_override_intent_when_new_projection_agrees(self):
        corrected = self.original > 0; corrected[3, 4] = False; corrected[15, 19] = True
        self.save(corrected)
        self.assertEqual(self.client.post('/api/export').status_code, 200)
        with np.load(self.ws.export_dir/'9.delta.npz') as d:
            added, removed = d['added'].copy(), d['removed'].copy()
        cv2.imwrite(str(self.masks/'9.masked.png'), corrected.astype(np.uint8)*255)
        self.assertEqual(self.client.post('/api/export').status_code, 200)
        with np.load(self.ws.export_dir/'9.delta.npz') as d:
            np.testing.assert_array_equal(d['added'], added)
            np.testing.assert_array_equal(d['removed'], removed)

    def test_multi_smart_api_uses_live_canvas_and_writes_nothing(self):
        gui.CFG['no_sam'] = False
        canvas = np.zeros((32, 40, 4), np.uint8)
        canvas[5:20, 5:20] = (0, 0, 255, 130); canvas[10, 10] = 0
        _, png = cv2.imencode('.png', canvas)
        payload = dict(id=self.oid, frame='left/9.png', point=[8, 8], shift=True,
                       mask='data:image/png;base64,' + base64.b64encode(png).decode())
        def sam(model, proc, image, points, labels, box, prior):
            self.assertTrue(prior[15, 15])  # live canvas differs from the file on disk
            return prior
        with patch.object(gui.G, '_segment_refine', side_effect=sam):
            response = self.client.post('/api/multi_smart', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json['added'], 0)
        self.assertEqual(response.json['removed'], 0)
        self.assertEqual(self.snapshot(), self.before)

    def test_exclusion_margin_forces_full_masks_and_blocks_edits(self):
        changes = self.sequence / 'changes'; changes.mkdir()
        (changes / 'changes.json').write_text(json.dumps(dict(rgb=[10], stereo=[], windows=[[10, 10]])))
        self.ws = gui.configure_grayscale(self.sequence, no_sam=False)
        self.assertEqual(self.ws.excluded, {9, 10, 100})
        self.assertTrue(self.ws.mask('left/9.png').all())
        self.assertEqual(self.save(np.zeros_like(self.original, bool)).status_code, 400)
        self.assertEqual(self.client.post('/api/refine', json=dict(frame='left/9.png', state='pre')).status_code, 400)
        self.assertEqual(self.client.post('/api/export').status_code, 200)
        for p in self.ws.export_dir.glob('*.png'):
            self.assertTrue(cv2.imread(str(p), 0).all())
        with np.load(self.ws.export_dir / '9.delta.npz') as delta:
            self.assertFalse(delta['added'].any() or delta['removed'].any())
        self.assert_originals_unchanged()

    def test_later_export_keeps_prior_corrections_and_reads_exports_before_projection(self):
        self.save(np.zeros_like(self.original, bool))
        self.assertEqual(self.client.post("/api/export").status_code, 200)
        self.save(np.ones_like(self.original, bool), "left/10.png")
        r = self.client.post("/api/export")
        self.assertEqual(r.json["n"], 3)
        self.assertEqual(r.json["updated"], 1)
        resumed = GrayscaleDepth(self.sequence)
        self.assertFalse(resumed.mask("left/9.png").any())
        self.assertTrue(resumed.mask("left/10.png").all())
        self.assert_originals_unchanged()
        # A pipeline rerun cannot replace a correction in the export folder.
        cv2.imwrite(str(self.masks / "9.masked.png"), np.full_like(self.original, 255))
        self.assertFalse(GrayscaleDepth(self.sequence).mask("left/9.png").any())

    def test_drafts_from_previous_source_overwrite_version_can_still_export(self):
        import hashlib
        self.ws.draft_dir.mkdir()
        baseline = {suffix: hashlib.sha256((self.masks / f"9.{suffix}.png").read_bytes()).hexdigest()
                    for suffix in ("masked", "invalid")}
        np.savez(self.ws.draft_path("left/9.png"), mask=np.zeros_like(self.original, bool),
                 baseline=json.dumps(baseline))
        self.assertEqual(self.client.post("/api/export").status_code, 200)
        self.assertFalse(GrayscaleDepth(self.sequence).mask("left/9.png").any())
        self.assert_originals_unchanged()

    def test_mismatched_source_and_draft_dimensions_are_rejected(self):
        self.assertEqual(self.save(np.zeros((16, 20), bool)).status_code, 400)
        self.assertEqual(self.snapshot(), self.before)
        cv2.imwrite(str(self.masks / "9.masked.png"), np.zeros((16, 20), np.uint8))
        with self.assertRaisesRegex(ValueError, "does not match"):
            GrayscaleDepth(self.sequence)

    def test_invalid_upload_and_unknown_paths_do_not_write(self):
        r = self.client.post("/api/save_edit", json={"id": self.oid, "frame": "left/9.png", "mask": "bad"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.save(self.original > 0, "../../elsewhere.png").status_code, 400)
        self.assertEqual(self.client.get("/frame?state=pre&name=../manifest.json").status_code, 400)
        self.assertEqual(self.client.get("/frame?state=post&name=left/9.png").status_code, 400)
        self.assertEqual(self.snapshot(), self.before)

    def test_other_writers_are_disabled(self):
        for endpoint in ("add_object", "add_seed", "delete_object", "delete_mask", "nuke_workspace",
                         "depth_propagate", "propagate_fix", "propagate_apply", "remask", "rebuild3d",
                         "set_meta", "group", "cloud_diff"):
            with self.subTest(endpoint=endpoint):
                r = self.client.post("/api/" + endpoint, json={"id": self.oid, "frame": "left/9.png"})
                self.assertEqual(r.status_code, 400)
        self.assertEqual(self.snapshot(), self.before)

    def test_external_source_change_rejects_whole_batch(self):
        self.save(np.zeros_like(self.original, bool))
        self.save(np.zeros_like(self.original, bool), "left/10.png")
        cv2.imwrite(str(self.masks / "10.masked.png"), np.full_like(self.original, 255))
        snapshot = self.snapshot()
        r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 400)
        self.assertIn("Saved mask changed", r.json["error"])
        self.assertEqual(self.snapshot(), snapshot)
        self.assertEqual(len(self.ws.pending()), 2)

    def test_write_failure_preserves_previous_complete_export_and_drafts(self):
        import grayscale_depth
        replace = grayscale_depth.os.replace
        self.assertEqual(self.client.post("/api/export").status_code, 200)
        previous = self.export_snapshot()
        self.save(np.zeros_like(self.original, bool))
        def fail_alias(src, dst):
            if Path(dst).name == "9.invalid.png":
                raise PermissionError("test alias is unwritable")
            return replace(src, dst)
        with patch("grayscale_depth.os.replace", side_effect=fail_alias):
            r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 500)
        self.assertIn("drafts kept", r.json["error"])
        self.assertEqual(self.export_snapshot(), previous)
        self.assertEqual(self.ws.pending(), ["left/9.png"])
        self.assertEqual(self.client.post("/api/export").json["updated"], 1)
        self.assertFalse(cv2.imread(str(self.ws.export_dir / "9.invalid.png"), 0).any())
        self.assert_originals_unchanged()

    def test_failed_publication_restores_previous_export(self):
        import grayscale_depth
        self.assertEqual(self.client.post("/api/export").status_code, 200)
        previous = self.export_snapshot()
        self.save(np.zeros_like(self.original, bool))
        replace = grayscale_depth.os.replace
        def fail_publish(src, dst):
            if Path(dst) == self.ws.export_dir and not str(src).endswith(".previous"):
                raise PermissionError("test publication failure")
            return replace(src, dst)
        with patch("grayscale_depth.os.replace", side_effect=fail_publish):
            r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 500)
        self.assertEqual(self.export_snapshot(), previous)
        self.assertEqual(self.ws.pending(), ["left/9.png"])
        self.assert_originals_unchanged()

    def test_external_export_change_rejects_draft_without_losing_previous_export(self):
        self.assertEqual(self.client.post("/api/export").status_code, 200)
        self.save(np.zeros_like(self.original, bool))
        cv2.imwrite(str(self.ws.export_path("left/9.png")), np.full_like(self.original, 255))
        previous = self.export_snapshot()
        r = self.client.post("/api/export")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.export_snapshot(), previous)
        self.assertEqual(self.ws.pending(), ["left/9.png"])
        self.assert_originals_unchanged()

    def test_sam_uses_grayscale_image_as_three_channels_without_source_writes(self):
        gui.CFG["no_sam"] = False
        def segment(model, proc, image, **kwargs):
            self.assertEqual(image.shape, (32, 40, 3))
            np.testing.assert_array_equal(image[:, :, 0], image[:, :, 1])
            return np.array([self.original > 0]), np.array([0.9])
        with patch.object(gui.G, "_segment", side_effect=segment):
            r = self.client.post("/api/edit_click", json={"id": self.oid, "frame": "left/9.png",
                                                         "points": [[4, 4]]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.snapshot(), self.before)
        gui.CFG["no_sam"] = True
        self.assertEqual(self.client.post("/api/edit_click", json={}).status_code, 400)

    def test_mode_specific_page_and_legacy_frame_resolution(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn("rectified left grayscale", html)
        self.assertIn("static/grayscale.js", html)
        with self.client.get("/static/grayscale.js") as response:
            self.assertEqual(response.status_code, 200)
        gui.CFG.clear()
        gui.CFG.update(capture=str(self.sequence), scene="RGB test", depth=True,
                       seed={"n": 0, "min_vis": 10}, states={"pre": {"session": "walk_rgb"}})
        self.assertEqual(gui._frame_path("pre", "images/cam0/9.jpg"),
                         self.sequence / "sessions/walk_rgb/raw_data/images/cam0/9.jpg")
        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn('src="/static/grayscale.js"', html)
        self.assertIn("RGB test · rgb", html)


if __name__ == "__main__":
    unittest.main()
