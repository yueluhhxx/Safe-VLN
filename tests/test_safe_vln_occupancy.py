"""Pure-function tests for Safe-VLN occupancy-aware waypoint prediction."""

import importlib.util
from pathlib import Path
import unittest

import torch


# Import the pure module directly so these tests do not import ETPNav's
# Habitat-dependent package initializers or require a simulator process.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vlnce_baselines"
    / "waypoint_pred"
    / "occupancy.py"
)
SPEC = importlib.util.spec_from_file_location("safe_vln_occupancy", str(MODULE_PATH))
occupancy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(occupancy)


def normalized_depth_patches(fill=1.0):
    return [torch.full((1, 1, 360, 1), fill) for _ in range(4)]


class SafeVLNOccupancyTest(unittest.TestCase):
    def test_single_obstacle_masks_surface_and_unknown(self):
        patches = normalized_depth_patches()
        # Twelve 0.25 degree rays form heading bin zero.  A 1 m return is
        # surface range bin four; nearer bins are free and farther bins unknown.
        # Patch zero starts at -45 degrees; its optical axis (forward) is at
        # raw sample 180, or radial heading H/8=15 before the -15-bin roll.
        patches[0][:, :, 180:192, :] = 1.0 / 3.0

        mask = occupancy.build_safe_occupancy_mask(patches)

        self.assertEqual(tuple(mask.shape), (1, 120, 12))
        torch.testing.assert_close(mask[0, 0, :4], torch.zeros(4))
        torch.testing.assert_close(mask[0, 0, 4:], -torch.ones(8))
        torch.testing.assert_close(mask[0, 1:], torch.zeros(119, 12))

    def test_all_free_scan_produces_zero_mask(self):
        # A normalized return of one is exactly max range.  As in Sim2Sim's
        # extra [max_range, inf] histogram bin, every represented cell is free.
        mask = occupancy.build_safe_occupancy_mask(normalized_depth_patches())
        torch.testing.assert_close(mask, torch.zeros(1, 120, 12))

    def test_fusion_clamps_normalizes_and_falls_back(self):
        probabilities = torch.tensor(
            [[[-1.0, float("nan")], [2.0, 1.0]]], dtype=torch.float32
        )
        mask = torch.tensor(
            [[[0.0, 0.0], [-1.0, 0.0]]], dtype=torch.float32
        )
        fused = occupancy.fuse_occupancy_mask(
            probabilities, mask, delta=0.5
        )

        self.assertTrue(bool(torch.all(torch.isfinite(fused))))
        self.assertTrue(bool(torch.all(fused >= 0.0)))
        torch.testing.assert_close(fused.sum(), torch.tensor(1.0))
        torch.testing.assert_close(
            fused, torch.tensor([[[0.0, 0.0], [1.0 / 3.0, 2.0 / 3.0]]])
        )

        # If every cell is removed by the mask, retain the valid base map.
        fallback = occupancy.fuse_occupancy_mask(
            torch.zeros(1, 2, 2), -torch.ones(1, 2, 2), delta=1.0
        )
        torch.testing.assert_close(fallback, torch.full((1, 2, 2), 0.25))

        # The requested run uses 1e-5 (overriding the paper's 1e-4).
        default_delta = occupancy.fuse_occupancy_mask(
            torch.tensor([[[0.6, 0.4]]]),
            torch.tensor([[[-1.0, 0.0]]]),
        )
        expected = torch.tensor(
            [[
                [(0.6 - 1.0e-5) / (1.0 - 1.0e-5),
                 0.4 / (1.0 - 1.0e-5)]
            ]]
        )
        torch.testing.assert_close(default_delta, expected)

    def test_forward_clockwise_patch_and_heading_alignment(self):
        patches = normalized_depth_patches()
        obstacle_bins = [2, 4, 6, 8]
        for patch_index, range_bin in enumerate(obstacle_bins):
            distance = range_bin * 0.25
            patches[patch_index][:, :, 180:192, :] = distance / 3.0

        mask = occupancy.build_safe_occupancy_mask(patches)

        # Patch optical axes [front, right, back, left] map to clockwise
        # heading bins [0, 30, 60, 90].  This explicitly catches omission or
        # sign reversal of the raw H/8 -> waypoint-zero correction.
        for heading_bin, range_bin in zip([0, 30, 60, 90], obstacle_bins):
            torch.testing.assert_close(
                mask[0, heading_bin, :range_bin], torch.zeros(range_bin)
            )
            self.assertEqual(mask[0, heading_bin, range_bin].item(), -1.0)
            previous_heading = (heading_bin - 1) % 120
            torch.testing.assert_close(
                mask[0, previous_heading], torch.zeros(12)
            )

        with self.assertRaises(ValueError):
            occupancy.validate_occupancy_geometry(
                4, 360, 120, 12, 3.0, heading_offset_bins=120
            )
        with self.assertRaises(ValueError):
            occupancy.validate_occupancy_geometry(
                4, 360, 120, 12, 3.0, heading_offset_bins=0
            )

    def test_regional_probability_sampling_preserves_original_semantics(self):
        # Original ETPNav applies softmax to selected regional logits.  With
        # occupancy disabled, conditioning the global softmax on that region
        # must be identical; this is the path now shared with masked sampling.
        logits = torch.tensor(
            [[-2.0, 0.5, 1.5, -0.25], [3.0, -1.0, 0.0, 2.0]]
        )
        region = torch.tensor([0, 2, 3])
        original = torch.softmax(logits[:, region], dim=1)
        global_probabilities = torch.softmax(logits, dim=1)
        conditioned = occupancy.normalize_probability_rows(
            global_probabilities[:, region]
        )
        torch.testing.assert_close(conditioned, original)

    def test_all_heading_range_pairs_round_trip_regional_sampling(self):
        headings = torch.arange(120).view(120, 1).expand(120, 12)
        ranges = torch.arange(12).view(1, 12).expand(120, 12)
        for offset in (0, 5, 15, -5):
            rolled_headings = (headings + offset) % 120
            regions = rolled_headings // 10
            regional_samples = (rolled_headings % 10) * 12 + ranges

            decoded_headings, decoded_ranges = (
                occupancy.regional_waypoint_to_global_indices(
                    regions, regional_samples, heading_offset=offset
                )
            )

            torch.testing.assert_close(decoded_headings, headings)
            torch.testing.assert_close(decoded_ranges, ranges)

    def test_regional_inverse_matches_actual_heatmap_roll_all_cells(self):
        heatmap = torch.arange(120 * 12).reshape(120, 12)
        # This is the production map layout: clockwise headings are rolled
        # by five, then reshaped into twelve 10-heading panorama regions.
        regional = torch.roll(heatmap, shifts=5, dims=0).reshape(12, 120)
        regions = torch.arange(12).view(12, 1)
        samples = torch.arange(120).view(1, 120)
        headings, ranges = occupancy.regional_waypoint_to_global_indices(
            regions, samples
        )
        torch.testing.assert_close(heatmap[headings, ranges], regional)
        torch.testing.assert_close(
            headings[0, ::12], torch.tensor([115, 116, 117, 118, 119, 0, 1, 2, 3, 4])
        )

    def test_decoded_heading_keeps_correct_clockwise_and_visual_region(self):
        regions = torch.arange(12).view(12, 1).expand(12, 120)
        samples = torch.arange(120).view(1, 120).expand(12, 120)
        headings, _ = occupancy.regional_waypoint_to_global_indices(regions, samples)

        # Selection uses clockwise image indices; RGB/depth features are
        # later restored to counter-clockwise order in the production policy.
        clockwise_region = ((headings + 5) // 10) % 12
        counter_clockwise_visual_region = (12 - (headings + 5) // 10) % 12
        torch.testing.assert_close(clockwise_region, regions)
        torch.testing.assert_close(counter_clockwise_visual_region, (12 - regions) % 12)

    def test_all_regions_conditioning_matches_logits_without_occupancy(self):
        generator = torch.Generator().manual_seed(121)
        logits = torch.randn((3, 120, 12), generator=generator, dtype=torch.float64)
        global_probabilities = torch.softmax(logits.reshape(3, -1), dim=1).reshape_as(logits)
        regional_logits = torch.roll(logits, shifts=5, dims=1).reshape(-1, 120)
        expected = torch.softmax(regional_logits, dim=1)

        # Both disabling the coefficient and an entirely free scan must retain
        # the original ETPNav conditional distribution for every region.
        for delta, mask in (
            (0.0, -torch.ones_like(logits)),
            (1.0e-5, torch.zeros_like(logits)),
        ):
            fused = occupancy.fuse_occupancy_mask(global_probabilities, mask, delta=delta)
            regional_probabilities = torch.roll(fused, shifts=5, dims=1).reshape(-1, 120)
            conditioned = occupancy.normalize_probability_rows(regional_probabilities)
            torch.testing.assert_close(conditioned, expected)

    def test_fusion_preserves_probability_dtype_and_gradient_path(self):
        # The scan-to-mask conversion deliberately uses detached FP32, but
        # fusion retains the heatmap dtype and does not detach that heatmap.
        # No new cast/no_grad is introduced by the heading-index correction.
        for dtype in (torch.float32, torch.float64):
            probabilities = torch.tensor([[[0.6, 0.4]]], dtype=dtype, requires_grad=True)
            mask = torch.tensor([[[-1.0, 0.0]]], dtype=torch.float32)
            fused = occupancy.fuse_occupancy_mask(probabilities, mask)
            self.assertEqual(fused.dtype, dtype)
            fused[0, 0, 0].backward()
            self.assertIsNotNone(probabilities.grad)
            self.assertEqual(probabilities.grad.dtype, dtype)
            self.assertTrue(bool(torch.all(torch.isfinite(probabilities.grad))))
            self.assertGreater(probabilities.grad.abs().sum().item(), 0.0)

    def test_regional_inverse_validates_geometry(self):
        for kwargs in (
            {"heading_bins": 0}, {"range_bins": 0},
            {"region_width": 0}, {"region_width": 7},
        ):
            with self.assertRaises(ValueError):
                occupancy.regional_waypoint_to_global_indices(0, 0, **kwargs)


if __name__ == "__main__":
    unittest.main()
