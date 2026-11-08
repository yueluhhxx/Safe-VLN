"""CPU-only tests for batched candidate and cached teacher queries."""

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vlnce_baselines"
    / "common"
    / "teacher_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "safe_vln_teacher_cache", str(MODULE_PATH)
)
teacher_cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teacher_cache
SPEC.loader.exec_module(teacher_cache)


class CandidateBatchingTest(unittest.TestCase):
    def test_pose_reuse_is_limited_to_r2r_train_or_video(self):
        should_reuse = teacher_cache.should_use_candidate_payload_pose

        self.assertTrue(should_reuse("r2r", "train", []))
        self.assertTrue(should_reuse("r2r", "eval", ["disk"]))
        self.assertFalse(should_reuse("r2r", "eval", []))
        self.assertFalse(should_reuse("r2r", "infer", []))
        self.assertFalse(should_reuse("rxr", "train", []))

    def test_batch_preserves_scalar_order_and_arguments(self):
        calls = []

        def scalar_position(*, angle, forward):
            calls.append((angle, forward))
            return (angle, 0.0, forward)

        positions = teacher_cache.batch_candidate_real_positions(
            scalar_position,
            angles=[0.3, 1.2, 2.4],
            forwards=[0.25, 1.0, 2.75],
        )

        self.assertEqual(
            calls,
            [(0.3, 0.25), (1.2, 1.0), (2.4, 2.75)],
        )
        self.assertEqual(
            positions,
            [(0.3, 0.0, 0.25), (1.2, 0.0, 1.0), (2.4, 0.0, 2.75)],
        )

    def test_batch_rejects_truncated_candidate_payload(self):
        with self.assertRaises(ValueError):
            teacher_cache.batch_candidate_real_positions(
                lambda **kwargs: kwargs,
                angles=[0.0, 1.0],
                forwards=[0.25],
            )

    def test_teacher_payload_combines_all_environment_results(self):
        calls = []

        def scalar_position(*, angle, forward):
            calls.append(("position", angle, forward))
            return (angle, 0.0, forward)

        def goal_distances(positions):
            calls.append(("goals", tuple(positions)))
            return [position[0] + position[2] for position in positions]

        def current_goal_distance():
            calls.append(("current",))
            return 8.5

        payload = teacher_cache.build_candidate_teacher_payload(
            scalar_position,
            goal_distances,
            current_goal_distance,
            angles=[0.5, 1.5],
            forwards=[0.25, 0.75],
        )

        self.assertEqual(payload, {
            "positions": [(0.5, 0.0, 0.25), (1.5, 0.0, 0.75)],
            "goal_distances": [0.75, 2.25],
            "current_goal_distance": 8.5,
        })
        self.assertEqual(calls, [
            ("current",),
            ("position", 0.5, 0.25),
            ("position", 1.5, 0.75),
            (
                "goals",
                ((0.5, 0.0, 0.25), (1.5, 0.0, 0.75)),
            ),
        ])

    def test_teacher_payload_returns_pose_only_after_exact_restoration(self):
        initial_pose = ((4.0, 0.5, -2.0), (0.0, 0.2, 0.0, 0.98))
        state = {"pose": initial_pose}
        calls = []

        def pose():
            calls.append(("pose", state["pose"]))
            return state["pose"]

        def scalar_position(*, angle, forward):
            saved_pose = state["pose"]
            state["pose"] = (
                (angle, saved_pose[0][1], forward),
                (0.0, angle, 0.0, 1.0),
            )
            projected_position = state["pose"][0]
            state["pose"] = saved_pose
            calls.append(("position", angle, forward))
            return projected_position

        payload = teacher_cache.build_candidate_teacher_payload(
            scalar_position,
            lambda positions: [1.0 for _ in positions],
            lambda: 3.0,
            angles=[0.25, 0.75],
            forwards=[1.0, 2.0],
            get_pose=pose,
        )

        self.assertEqual(payload["pose"], initial_pose)
        self.assertEqual(state["pose"], initial_pose)
        self.assertEqual(calls[0][0], "pose")
        self.assertEqual(calls[-1][0], "pose")

    def test_teacher_payload_rejects_inexact_pose_restoration(self):
        poses = iter([
            ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
            ((1.0, 2.0, 3.0 + 1.0e-12), (0.0, 0.0, 0.0, 1.0)),
        ])

        with self.assertRaisesRegex(RuntimeError, "exactly restore"):
            teacher_cache.build_candidate_teacher_payload(
                lambda **kwargs: (0.0, 0.0, 0.0),
                lambda positions: [0.0 for _ in positions],
                lambda: 0.0,
                angles=[0.0],
                forwards=[0.25],
                get_pose=lambda: next(poses),
            )


class EpisodeGeodesicCacheTest(unittest.TestCase):
    def test_missing_batches_are_split_per_environment_and_deduplicated(self):
        caches = [
            teacher_cache.EpisodeGeodesicCache(),
            teacher_cache.EpisodeGeodesicCache(),
        ]
        cached = (1.0, 0.0, 2.0)
        caches[0].update([cached], [7.0])

        positions = [
            [cached, (2.0, 0.0, 3.0), (2.0, 0.0, 3.0)],
            [(8.0, 0.0, 9.0), (9.0, 0.0, 10.0)],
        ]
        missing = teacher_cache.prepare_missing_position_batches(
            caches, positions
        )

        self.assertEqual(missing, [
            [(2.0, 0.0, 3.0)],
            [(8.0, 0.0, 9.0), (9.0, 0.0, 10.0)],
        ])
        teacher_cache.update_geodesic_cache_batches(
            caches,
            missing,
            [[5.0], [4.0, 3.0]],
        )
        self.assertEqual(
            teacher_cache.prepare_missing_position_batches(caches, positions),
            [[], []],
        )

    def test_cached_minima_match_legacy_per_ghost_queries(self):
        # Two ghosts share p1.  The legacy pathfinder code queried each group
        # every step; the optimized code queries the three unique positions
        # once and then performs the same per-ghost minimum.
        p0 = (0.0, 0.0, 0.0)
        p1 = (1.0, 0.0, 0.0)
        p2 = (2.0, 0.0, 0.0)
        ghost_groups = [[p0, p1], [p1, p2]]

        def geodesic(position):
            return {p0: 6.0, p1: 2.0, p2: 4.0}[tuple(position)]

        legacy = [
            min(geodesic(position) for position in group)
            for group in ghost_groups
        ]

        cache = teacher_cache.EpisodeGeodesicCache()
        flattened = [position for group in ghost_groups for position in group]
        missing = cache.missing_positions(flattened)
        self.assertEqual(missing, [p0, p1, p2])
        cache.update(missing, [geodesic(position) for position in missing])
        optimized = [cache.minimum(group) for group in ghost_groups]

        self.assertEqual(optimized, legacy)
        self.assertEqual(cache.missing_positions(flattened), [])

    def test_exact_tuple_does_not_round_nearby_positions(self):
        cache = teacher_cache.EpisodeGeodesicCache()
        first = (1.0, 2.0, 3.0)
        nearby = (1.0, 2.0, 3.0 + 1.0e-12)
        cache.update([first], [5.0])

        self.assertEqual(cache.missing_positions([nearby]), [nearby])

    def test_result_length_mismatch_is_rejected(self):
        cache = teacher_cache.EpisodeGeodesicCache()
        with self.assertRaises(ValueError):
            cache.update([(0.0, 0.0, 0.0)], [])


if __name__ == "__main__":
    unittest.main()
