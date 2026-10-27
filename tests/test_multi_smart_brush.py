import unittest
from dataclasses import replace

import numpy as np
from scipy import ndimage as ndi

from multi_smart_brush import BrushConstants, holes, inside_distance, refine_blob, select_blob


class MultiSmartBrushTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.zeros((512, 512), bool)
        self.mask[60:200, 60:200] = True
        self.mask[90:93, 90:93] = False  # closing would otherwise swallow this pinhole
        self.mask[250:400, 280:430] = True
        self.mask[300:305, 310:315] = False
        self.image = np.zeros(self.mask.shape, np.uint8)

    def run_brush(self, sam, point=(80, 80), shift=False, mask=None, **kw):
        return refine_blob(self.image, self.mask if mask is None else mask, point, shift, sam, **kw)

    def test_fill_pinholes_even_when_sam_is_rejected_other_blob_unchanged(self):
        r = self.run_brush(lambda *args: np.zeros_like(self.mask))
        self.assertTrue(r['sam_rejected'])
        self.assertEqual(r['holes_filled'], 1)
        self.assertTrue(r['mask'][90:93, 90:93].all())
        self.assertFalse(r['removed'].any())
        outside = np.ones_like(self.mask); outside[60:200, 60:200] = False
        np.testing.assert_array_equal(r['mask'][outside], self.mask[outside])

    def test_closing_selects_fragments_but_never_paints_bridges(self):
        m = np.zeros_like(self.mask); m[60:180, 60:130] = True; m[60:180, 133:200] = True
        _, b = select_blob(m, (80, 80))
        np.testing.assert_array_equal(b, m)
        r = self.run_brush(lambda *args: np.zeros_like(m), mask=m)
        np.testing.assert_array_equal(r['mask'], m)

    def test_click_in_hole_near_edge_and_too_far(self):
        m = self.mask.copy(); m[100:160, 100:160] = False
        component, _ = select_blob(m, (130, 130))
        self.assertTrue(component[80, 80])
        _, b = select_blob(m, (45, 80)); self.assertTrue(b[80, 80])
        def never(*args): self.fail('SAM must not run for nothing near click')
        r = self.run_brush(never, point=(44, 80), mask=m)
        self.assertTrue(r['nothing_near_click'])

    def test_trim_only_two_pixels_and_filled_holes_survive(self):
        def sam(image, points, labels, box, prior):
            s = ndi.binary_erosion(prior, iterations=8)
            s[90:93, 90:93] = False
            return s
        r = self.run_brush(sam)
        self.assertFalse(r['sam_rejected'])
        self.assertTrue(r['removed'].any())
        _, original_blob = select_blob(self.mask, (80, 80))
        self.assertTrue((inside_distance(original_blob)[r['removed']] <= 2).all())
        self.assertTrue(r['mask'][90:93, 90:93].all())
        shifted = self.run_brush(sam, shift=True)
        self.assertFalse(shifted['removed'].any())
        self.assertTrue(shifted['mask'][self.mask].all())

    def test_outward_growth_is_box_limited_other_blob_and_its_holes_protected(self):
        m = self.mask.copy()
        # A separate thin neighbor inside the prompt box, with its own hole.
        m[65:195, 207:212] = True; m[120, 209] = False
        def sam(image, points, labels, box, prior):
            return ndi.binary_dilation(prior, iterations=12)
        r = self.run_brush(sam, mask=m)
        self.assertFalse(r['sam_rejected'])
        self.assertTrue(r['added'].any())
        self.assertFalse(r['mask'][120, 209])
        _, b = select_blob(m, (80, 80))
        self.assertTrue(r['mask'][m & ~b].all())
        ys, xs = np.where(r['added'])
        self.assertTrue(((xs >= 52) & (xs < 208) & (ys >= 52) & (ys < 208)).all())

    def test_large_holes_are_only_filled_where_sam_covers_them(self):
        m = self.mask.copy(); m[100:150, 100:150] = False
        def sam(image, points, labels, box, prior):
            out = prior.copy(); out[100:125, 100:150] = True; return out
        r = self.run_brush(sam, mask=m)
        self.assertTrue(r['mask'][100:125, 100:150].all())
        self.assertFalse(r['mask'][125:150, 100:150].any())
        self.assertEqual(r['holes_filled'], 1)  # only the separate 3x3 hole

    def test_sam_prompts_only_selected_blob_and_interior_positive_points(self):
        calls = []
        def sam(image, points, labels, box, prior):
            calls.append(1)
            self.assertEqual(image.shape, (512, 512, 3))
            np.testing.assert_array_equal(image[:, :, 0], image[:, :, 2])
            self.assertLessEqual(len(points), 5)
            np.testing.assert_array_equal(points[0], [80, 80])
            self.assertTrue((labels == 1).all())
            self.assertEqual(box, [52, 52, 208, 208])
            self.assertFalse(prior[250:, :].any())
            for x, y in points[1:].astype(int):
                self.assertGreater(inside_distance(prior)[y, x], 2)
            return prior
        self.run_brush(sam)
        self.assertEqual(len(calls), 1)

    def test_growth_ratio_rejection_separate_from_iou(self):
        def sam(image, points, labels, box, prior):
            out = prior.copy(); out[200:250, 60:200] = True; return out
        r = self.run_brush(sam, constants=replace(BrushConstants(), sam_max_grow=1.1))
        self.assertTrue(r['sam_rejected'])
        self.assertFalse(r['mask'][220, 100])

    def test_full_image_and_exclusion_are_noops_without_model_call(self):
        def never(*args): self.fail('Excluded frame must not invoke SAM')
        for mask, excluded in [(np.ones_like(self.mask), False), (self.mask, True)]:
            r = self.run_brush(never, mask=mask, excluded=excluded)
            np.testing.assert_array_equal(r['mask'], mask)
            self.assertTrue(r['excluded'])
            self.assertFalse(r['added'].any() or r['removed'].any())

    def test_delta_exactly_matches_before_and_after(self):
        r = self.run_brush(lambda image, points, labels, box, prior: ndi.binary_dilation(prior, iterations=2))
        np.testing.assert_array_equal(r['added'], r['mask'] & ~self.mask)
        np.testing.assert_array_equal(r['removed'], self.mask & ~r['mask'])
        np.testing.assert_array_equal((self.mask | r['added']) & ~r['removed'], r['mask'])


if __name__ == '__main__':
    unittest.main()
