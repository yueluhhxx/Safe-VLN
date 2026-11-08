"""Exercise actual Tryout methods with a heading-aware, 15-degree CPU fake.

The shared AST loader extracts the production controller instead of copying its
algorithm.  Unlike the older position-only fake, every simulated turn here
updates the heading used by subsequent relative-angle calculations.
"""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from tests import test_sliding_tryout_controller as controller_tests


Actions = controller_tests._Actions


def signed_degrees(radians):
    return (math.degrees(float(radians)) + 180.0) % 360.0 - 180.0


class _SweepChoice:
    def __init__(self, left_first):
        self.left_first = left_first
        self.calls = 0

    def choice(self, values):
        self.calls += 1
        return values[0 if self.left_first else 1]


class _HeadingSim(controller_tests._FakeSim):
    def __init__(self, events, heading_degrees=0.0):
        super().__init__(events)
        self.rotation = math.radians(heading_degrees)
        self.forward_headings = []
        for action in (Actions.TURN_LEFT, Actions.TURN_RIGHT):
            self.agent.agent_config.action_space[action].actuation.amount = 15

    def forward(self):
        self.forward_headings.append(signed_degrees(self.rotation))
        super().forward()


def _make_env(events, left_first=True, heading_degrees=0.0):
    # Each extraction has independent globals so sweep selection cannot leak
    # into the shared tests or another test's controller instance.
    environment_class = controller_tests._extract_environment_class()
    sweep_choice = _SweepChoice(left_first)
    environment_class.single_step_control.__globals__["random"] = sweep_choice
    env = object.__new__(environment_class)
    sim = _HeadingSim(events, heading_degrees)
    env.video_option = False
    env.action_log = []
    env.teleports = []

    def no_stop(action):
        raise AssertionError("Tryout exhaustion must not call env.step/STOP")

    env._env = SimpleNamespace(sim=sim, step=no_stop)

    def wrap_act(action, vis_info):
        before = sim.rotation
        if action == Actions.MOVE_FORWARD:
            sim.forward()
        elif action in (Actions.TURN_LEFT, Actions.TURN_RIGHT):
            amount = sim.agent.agent_config.action_space[action].actuation.amount
            sim.rotation += math.radians(
                amount if action == Actions.TURN_LEFT else -amount
            )
            sim.previous_step_collided = False
        else:
            raise AssertionError("unexpected low-level action: {!r}".format(action))
        env.action_log.append((action, before, sim.rotation))
        return {}

    def no_teleport(position):
        env.teleports.append(tuple(np.asarray(position).tolist()))
        raise AssertionError("control-mode Tryout must not teleport")

    env.wrap_act = wrap_act
    env.teleport = no_teleport
    env.get_observation_at = lambda position, rotation: {
        "position": tuple(np.asarray(position).tolist()),
        "heading": float(rotation),
    }
    env.get_reward = lambda observations: 0.0
    env.get_done = lambda observations: False
    env.get_info = lambda observations: {}
    return env, sim, sweep_choice


def _blocked(count):
    return [((0.0, 0.0, 0.0), True)] * count


class TryoutHeadingAndFallbackTest(unittest.TestCase):
    def assertHeadings(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for observed, requested in zip(actual, expected):
            # Compare angles modulo one revolution, including +/-180.
            error = (observed - requested + 180.0) % 360.0 - 180.0
            self.assertAlmostEqual(error, 0.0, places=5)

    def test_both_sweeps_use_original_six_absolute_probe_headings(self):
        target_heading = 45.0
        target = np.array([-math.sqrt(2.0), 0.0, -math.sqrt(2.0)])
        for left_first in (True, False):
            with self.subTest(left_first=left_first):
                env, sim, sweep = _make_env(
                    _blocked(7), left_first=left_first, heading_degrees=-15.0
                )
                result = env.single_step_control(
                    target, True, None, reach_tolerance=0.1
                )
                offsets = [90, 60, 30, -30, -60, -90]
                if not left_first:
                    offsets.reverse()
                self.assertHeadings(
                    sim.forward_headings,
                    [target_heading] + [target_heading + x for x in offsets],
                )
                self.assertEqual(result["tryout_probe_count"], 6)
                self.assertEqual(result["trap_reason"], "probes_exhausted")
                self.assertEqual(result["outcome"], "trapped")
                self.assertEqual(sweep.calls, 1)
                self.assertEqual(sim.forward_calls, 7)
                self.assertFalse(sim.events)

    def test_all_failed_probes_preserve_final_heading_and_position(self):
        for left_first, final_heading in ((True, -90.0), (False, 90.0)):
            with self.subTest(left_first=left_first):
                env, sim, _ = _make_env(_blocked(7), left_first=left_first)
                result = env.single_step_control(
                    np.array([0.0, 0.0, -2.0]), True, None, reach_tolerance=0.1
                )
                self.assertTrue(result["trapped"])
                self.assertHeadings([signed_degrees(sim.rotation)], [final_heading])
                np.testing.assert_array_equal(sim.position, np.zeros(3))
                # No cleanup turn or extra action after the last failed probe.
                self.assertEqual(env.action_log[-1][0], Actions.MOVE_FORWARD)
                self.assertEqual(env.teleports, [])

    def test_moving_probe_restores_target_heading_before_blocked_tail(self):
        for left_first in (True, False):
            with self.subTest(left_first=left_first):
                probe_x = -0.25 if left_first else 0.25
                events = _blocked(1) + [((probe_x, 0.0, 0.0), True)] + _blocked(1)
                env, sim, sweep = _make_env(events, left_first=left_first)
                result = env.single_step_control(
                    np.array([0.0, 0.0, -2.0]), True, None, reach_tolerance=0.1
                )
                self.assertHeadings(
                    sim.forward_headings, [0.0, 90.0 if left_first else -90.0, 0.0]
                )
                self.assertHeadings([signed_degrees(sim.rotation)], [0.0])
                self.assertEqual(result["trap_reason"], "tail_blocked")
                self.assertEqual(result["tryout_probe_count"], 1)
                self.assertEqual(sim.forward_calls, 3)
                self.assertEqual(sweep.calls, 1)
                self.assertFalse(sim.events)
                # The other five probes must not run after committing a detour.
                self.assertEqual(env.action_log[-1][0], Actions.MOVE_FORWARD)

    def test_real_sliding_motion_does_not_trigger_turn_probes(self):
        env, sim, sweep = _make_env([((0.0, 0.0, -0.25), True)] * 4)
        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None, reach_tolerance=0.1
        )
        self.assertEqual(result["outcome"], "reached")
        self.assertFalse(result["trapped"])
        self.assertFalse(result["tryout_started"])
        self.assertEqual(result["collision_count"], 4)
        self.assertEqual(sweep.calls, 0)
        self.assertEqual(sim.forward_calls, 4)
        self.assertHeadings(sim.forward_headings, [0.0] * 4)

    def test_next_target_turn_is_recomputed_from_actual_failed_heading(self):
        env, sim, sweep = _make_env(_blocked(14))
        first = env.single_step_control(
            np.array([0.0, 0.0, -2.0]), True, None, reach_tolerance=0.1
        )
        self.assertTrue(first["trapped"])
        self.assertHeadings([signed_degrees(sim.rotation)], [-90.0])
        first_action_count = len(env.action_log)
        second = env.single_step_control(
            np.array([-2.0, 0.0, 0.0]), True, None, reach_tolerance=0.1
        )
        self.assertTrue(second["trapped"])
        second_actions = env.action_log[first_action_count:]
        first_forward = next(
            index for index, item in enumerate(second_actions)
            if item[0] == Actions.MOVE_FORWARD
        )
        # The remaining -90-degree heading requires a 180-degree turn to the
        # +90-degree second target: twelve real 15-degree TURN_LEFT actions.
        self.assertEqual(first_forward, 12)
        self.assertEqual(
            [item[0] for item in second_actions[:first_forward]],
            [Actions.TURN_LEFT] * 12,
        )
        self.assertHeadings(sim.forward_headings[7:], [90, 180, 150, 120, 60, 30, 0])
        self.assertEqual(sweep.calls, 2)
        self.assertEqual(sim.forward_calls, 14)
        self.assertFalse(sim.events)

    def test_step_both_targets_tryout_then_return_real_observation(self):
        for left_first in (True, False):
            with self.subTest(left_first=left_first):
                env, sim, sweep = _make_env(_blocked(14), left_first=left_first)
                attempts = []
                for ghost, position in (
                    ("g0", [0.0, 0.0, -2.0]),
                    ("g1", [-2.0, 0.0, 0.0]),
                ):
                    attempts.append({
                        "ghost_vp": ghost,
                        "ghost_pos": np.asarray(position),
                        "front_vp": "0",
                        "front_pos": np.zeros(3),
                        "back_path": [],
                        "tryout": True,
                    })
                observation, _, done, info = env.step({
                    "act": 4,
                    "cur_vp": "0",
                    "attempts": attempts,
                    "reach_tolerance": 0.1,
                }, None)
                result = info["reselection"]
                self.assertFalse(done)
                self.assertFalse(result["success"])
                self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
                self.assertEqual(result["trapped_ghosts"], ["g0", "g1"])
                self.assertEqual(result["attempts"], 2)
                self.assertEqual(len(result["details"]), 2)
                for detail in result["details"]:
                    self.assertTrue(detail["target_attempted"])
                    self.assertEqual(detail["move_result"]["tryout_probe_count"], 6)
                    self.assertEqual(detail["move_result"]["trap_reason"], "probes_exhausted")
                self.assertEqual(sweep.calls, 2)
                self.assertEqual(sim.forward_calls, 14)
                self.assertFalse(sim.events)
                offsets = [90, 60, 30, -30, -60, -90]
                if not left_first:
                    offsets.reverse()
                self.assertHeadings(sim.forward_headings[:7], [0] + offsets)
                self.assertHeadings(
                    sim.forward_headings[7:], [90] + [90 + x for x in offsets]
                )
                self.assertHeadings(
                    [signed_degrees(sim.rotation)], [90 + offsets[-1]]
                )
                self.assertEqual(env.action_log[-1][0], Actions.MOVE_FORWARD)
                self.assertEqual(env.teleports, [])
                self.assertEqual(observation["position"], tuple(sim.position.tolist()))
                self.assertEqual(observation["heading"], sim.rotation)

    def test_direct_secondary_starts_at_offset_pose_and_double_failure_stays_there(self):
        for left_first in (True, False):
            with self.subTest(left_first=left_first):
                probe_x = -0.25 if left_first else 0.25
                # Primary: blocked direct, moving probe, blocked tail.  The
                # next target must start at that displaced pose, not at cur_vp.
                events = _blocked(1) + [((probe_x, 0.0, 0.0), True)] + _blocked(8)
                env, sim, sweep = _make_env(events, left_first=left_first)
                attempts = [{
                    "ghost_vp": "g0",
                    "ghost_pos": np.array([0.0, 0.0, -2.0]),
                    "front_vp": "0",
                    "front_pos": np.zeros(3),
                    "back_path": [],
                    "tryout": True,
                }, {
                    "ghost_vp": "g1",
                    "ghost_pos": np.array([probe_x, 0.0, 1.0]),
                    # Direct mode must not visit or claim this planned front.
                    "front_vp": "remote",
                    "front_pos": np.array([20.0, 0.0, 20.0]),
                    "back_path": [("remote", [20.0, 0.0, 20.0])],
                    "routes_from": {
                        "0": [("0", [0.0, 0.0, 0.0]),
                              ("remote", [20.0, 0.0, 20.0])],
                    },
                    "tryout": True,
                }]
                observation, _, done, info = env.step({
                    "act": 4,
                    "cur_vp": "0",
                    "attempts": attempts,
                    "reselection_route_mode": "direct",
                    "reach_tolerance": 0.1,
                }, None)
                result = info["reselection"]
                self.assertFalse(done)
                self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
                self.assertEqual(result["trapped_ghosts"], ["g0", "g1"])
                self.assertEqual(result["details"][0]["move_result"]["trap_reason"], "tail_blocked")
                self.assertEqual(result["details"][1]["move_result"]["tryout_probe_count"], 6)
                self.assertIsNone(result["details"][1]["execution_path"])
                self.assertEqual(result["details"][1]["execution_anchor_vp"], "0")
                self.assertEqual(result["failure_anchor_vp"], "0")
                self.assertEqual(sweep.calls, 2)
                self.assertEqual(sim.forward_calls, 10)
                self.assertFalse(sim.events)
                offsets = [90, 60, 30, -30, -60, -90]
                if not left_first:
                    offsets.reverse()
                self.assertHeadings(
                    sim.forward_headings[3:], [180] + [180 + x for x in offsets]
                )
                self.assertHeadings(
                    [signed_degrees(sim.rotation)], [180 + offsets[-1]]
                )
                np.testing.assert_array_equal(sim.position, [probe_x, 0.0, 0.0])
                self.assertEqual(observation["position"], tuple(sim.position.tolist()))
                self.assertEqual(observation["heading"], sim.rotation)
                self.assertEqual(env.action_log[-1][0], Actions.MOVE_FORWARD)
                self.assertEqual(env.teleports, [])

    def test_failed_topological_route_nodes_do_not_suppress_secondary_tryout(self):
        # Primary, two route nodes, and secondary each run a direct step plus
        # six failed probes.  The already reached anchor needs no actions.
        env, sim, sweep = _make_env(_blocked(28))
        route = [("0", [0.0, 0.0, 0.0]),
                 ("1", [-1.0, 0.0, 0.0]),
                 ("2", [-2.0, 0.0, 0.0])]
        attempts = [{
            "ghost_vp": "g0",
            "ghost_pos": np.array([0.0, 0.0, -2.0]),
            "front_vp": "0",
            "front_pos": np.zeros(3),
            "back_path": [],
            "tryout": True,
        }, {
            "ghost_vp": "g1",
            "ghost_pos": np.array([0.0, 0.0, 2.0]),
            "front_vp": "2",
            "front_pos": np.array([-2.0, 0.0, 0.0]),
            "back_path": route[1:],
            "routes_from": {"0": route},
            "tryout": True,
        }]
        observation, _, done, info = env.step({
            "act": 4,
            "cur_vp": "0",
            "attempts": attempts,
            "reselection_route_mode": "topological",
            "reach_tolerance": 0.1,
        }, None)
        result = info["reselection"]
        self.assertFalse(done)
        self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
        self.assertEqual(result["trapped_ghosts"], ["g0", "g1"])
        self.assertEqual(result["attempts"], 2)
        secondary = result["details"][1]
        self.assertEqual(secondary["execution_path"], route)
        self.assertFalse(secondary["back_result"]["reached"])
        self.assertEqual(secondary["back_result"]["last_reached_vp"], "0")
        self.assertTrue(secondary["target_attempted"])
        self.assertEqual(secondary["move_result"]["tryout_probe_count"], 6)
        self.assertEqual(secondary["execution_anchor_vp"], "0")
        self.assertEqual(result["failure_anchor_vp"], "0")
        self.assertIsNone(result["selected_front_vp"])
        self.assertEqual(sweep.calls, 4)
        self.assertEqual(sim.forward_calls, 28)
        self.assertFalse(sim.events)
        self.assertHeadings(
            [sim.forward_headings[index] for index in (0, 7, 14, 21)],
            [0.0, 90.0, 90.0, 180.0],
        )
        self.assertHeadings([signed_degrees(sim.rotation)], [90.0])
        self.assertEqual(observation["position"], tuple(sim.position.tolist()))
        self.assertEqual(observation["heading"], sim.rotation)
        self.assertEqual(env.action_log[-1][0], Actions.MOVE_FORWARD)
        self.assertEqual(env.teleports, [])


if __name__ == "__main__":
    unittest.main()
