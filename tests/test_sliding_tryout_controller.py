"""CPU integration tests for sliding-aware ETPNav low-level control.

The production environment module imports Habitat and GPU-adjacent packages at
module import time.  These tests instead compile the actual controller methods
from its AST and execute them against a small fake simulator.  This keeps the
tests fast while ensuring they exercise the implementation rather than a
duplicated helper.
"""

import ast
from collections import deque
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH = (
    REPO_ROOT / "vlnce_baselines" / "common" / "environments.py"
)
RESELECTION_PATH = (
    REPO_ROOT / "vlnce_baselines" / "common" / "reselection.py"
)


class _Actions:
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class _DeterministicRandom:
    @staticmethod
    def choice(values):
        return values[0]


def _load_reselection_controller():
    spec = importlib.util.spec_from_file_location(
        "sliding_tryout_reselection", str(RESELECTION_PATH)
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ReselectionController


def _extract_environment_class():
    tree = ast.parse(ENVIRONMENT_PATH.read_text())
    calculate_position = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "calculate_vp_rel_pos"
    )
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLNCEDaggerEnv"
    )
    wanted = {"turn", "single_step_control", "multi_step_control", "step"}
    methods = [
        copy.deepcopy(node)
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    extracted_class = ast.ClassDef(
        name="ExtractedVLNCEDaggerEnv",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[copy.deepcopy(calculate_position), extracted_class],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "math": math,
        "random": _DeterministicRandom,
        "HabitatSimActions": _Actions,
        "heading_from_quaternion": lambda rotation: float(rotation),
        "ReselectionController": _load_reselection_controller(),
    }
    exec(compile(module, str(ENVIRONMENT_PATH), "exec"), namespace)
    return namespace["ExtractedVLNCEDaggerEnv"]


ExtractedEnv = _extract_environment_class()


class _FakeSim:
    def __init__(self, events=(), position=(0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float32)
        self.rotation = 0.0
        self.events = deque(events)
        self.previous_step_collided = False
        self.forward_calls = 0
        forward_spec = SimpleNamespace(actuation=SimpleNamespace(amount=0.25))
        turn_spec = SimpleNamespace(actuation=SimpleNamespace(amount=30))
        action_space = {
            _Actions.MOVE_FORWARD: forward_spec,
            _Actions.TURN_LEFT: turn_spec,
            _Actions.TURN_RIGHT: turn_spec,
        }
        self.agent = SimpleNamespace(
            agent_config=SimpleNamespace(action_space=action_space)
        )

    def get_agent(self, agent_id):
        self.assert_default_agent(agent_id)
        return self.agent

    @staticmethod
    def assert_default_agent(agent_id):
        if agent_id != 0:
            raise AssertionError("the controller must use the default agent")

    def get_agent_state(self):
        return SimpleNamespace(
            position=self.position.copy(), rotation=self.rotation
        )

    def set_agent_state(self, position, rotation):
        self.position = np.asarray(position, dtype=np.float32).copy()
        self.rotation = 0.0
        self.previous_step_collided = False

    def forward(self):
        if not self.events:
            raise AssertionError("unexpected MOVE_FORWARD action")
        delta, collided = self.events.popleft()
        self.position = self.position + np.asarray(delta, dtype=np.float32)
        self.previous_step_collided = bool(collided)
        self.forward_calls += 1


def _controller_env(events=(), position=(0.0, 0.0, 0.0)):
    instance = object.__new__(ExtractedEnv)
    sim = _FakeSim(events, position)
    instance._env = SimpleNamespace(sim=sim)
    instance.video_option = False
    instance.turn_actions = []

    def wrap_act(action, vis_info):
        if action == _Actions.MOVE_FORWARD:
            sim.forward()
        else:
            instance.turn_actions.append(action)
            sim.previous_step_collided = False
        return {}

    instance.wrap_act = wrap_act
    return instance, sim


def _geometric_controller_env(
    motion_mode, events=(), position=(0.0, 0.0, 0.0), heading=0.0
):
    """A heading-aware free-space sim after any scripted collision events."""
    instance, sim = _controller_env(events, position)
    instance.controller_motion_mode = motion_mode
    sim.rotation = float(heading)
    for turn_action in (_Actions.TURN_LEFT, _Actions.TURN_RIGHT):
        sim.agent.agent_config.action_space[turn_action].actuation.amount = 15

    def wrap_act(action, vis_info):
        if action == _Actions.MOVE_FORWARD:
            if sim.events:
                sim.forward()
            else:
                direction = np.array(
                    [-math.sin(sim.rotation), 0.0, -math.cos(sim.rotation)]
                )
                sim.position += (0.25 * direction).astype(np.float32)
                sim.previous_step_collided = False
                sim.forward_calls += 1
        else:
            instance.turn_actions.append(action)
            turn = math.radians(15)
            sim.rotation = (
                sim.rotation + (turn if action == _Actions.TURN_LEFT else -turn)
            ) % (2 * math.pi)
            sim.previous_step_collided = False
        return {}

    instance.wrap_act = wrap_act
    return instance, sim


def _result(outcome, remaining=1.0):
    reached = outcome == "reached"
    trapped = outcome == "trapped"
    return {
        "outcome": outcome,
        "reached": reached,
        "trapped": trapped,
        "target_attempted": True,
        "remaining_distance": 0.0 if reached else remaining,
        "remaining_distance_xz": 0.0 if reached else remaining,
        "remaining_distance_3d": 0.0 if reached else remaining,
        "vertical_error": 0.0,
        "travelled_distance": 0.0,
        "net_progress": 0.0,
        "collision_count": int(trapped),
        "tryout_started": trapped,
        "tryout_probe_count": 6 if trapped else 0,
        "tryout_exhausted": trapped,
        "trap_reason": "probes_exhausted" if trapped else None,
    }


def _attempt(ghost, front, ghost_x, back_path=None):
    return {
        "ghost_vp": ghost,
        "ghost_pos": np.array([ghost_x, 0.0, 0.0], dtype=np.float32),
        "front_vp": front,
        "front_pos": np.array([ghost_x - 0.25, 0.0, 0.0], dtype=np.float32),
        "back_path": back_path,
        "tryout": True,
    }


class SlidingAwareSingleStepTest(unittest.TestCase):
    def test_arrival_uses_xz_and_ignores_synthetic_target_height(self):
        env, sim = _controller_env(position=(2.0, 0.0, -3.0))

        result = env.single_step_control(
            np.array([2.0, 4.0, -3.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "reached")
        self.assertTrue(result["reached"])
        self.assertEqual(result["remaining_distance_xz"], 0.0)
        self.assertEqual(result["vertical_error"], 4.0)
        self.assertEqual(sim.forward_calls, 0)

    def test_visited_pose_arrival_uses_3d_across_floors(self):
        env, sim = _controller_env(position=(2.0, 0.0, -3.0))

        result = env.single_step_control(
            np.array([2.0, 4.0, -3.0]),
            False,
            None,
            reach_tolerance=0.1,
            horizontal_only=False,
        )

        self.assertEqual(result["outcome"], "inconclusive")
        self.assertFalse(result["reached"])
        self.assertEqual(result["remaining_distance_xz"], 0.0)
        self.assertEqual(result["remaining_distance_3d"], 4.0)
        self.assertEqual(result["remaining_distance"], 4.0)
        self.assertEqual(sim.forward_calls, 0)

    def test_sliding_collision_with_motion_is_not_trapped(self):
        events = [((0.0, 0.0, -0.25), True)] * 4
        env, sim = _controller_env(events)

        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "reached")
        self.assertFalse(result["trapped"])
        self.assertFalse(result["tryout_started"])
        self.assertEqual(result["collision_count"], 4)
        self.assertEqual(sim.forward_calls, 4)

    def test_six_stationary_probes_confirm_trapped(self):
        # Stuck detection is motion-based.  A simulator/plugin that reports no
        # collision bit but also produces no translation must still be trapped.
        events = [((0.0, 0.0, 0.0), False)] * 7
        env, sim = _controller_env(events)

        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "trapped")
        self.assertTrue(result["tryout_exhausted"])
        self.assertEqual(result["trap_reason"], "probes_exhausted")
        self.assertEqual(result["tryout_probe_count"], 6)
        self.assertEqual(sim.forward_calls, 7)

    def test_prior_progress_then_static_direct_and_probes_is_trapped(self):
        # Earlier motion does not disprove that the agent is now genuinely
        # trapped.  A later stationary direct step plus all six stationary
        # probes is the explicit failure evidence that permits re-selection.
        events = [
            ((0.0, 0.0, -0.25), False),
            ((0.0, 0.0, 0.0), False),
        ] + [((0.0, 0.0, 0.0), False)] * 6
        env, sim = _controller_env(events)

        result = env.single_step_control(
            np.array([0.0, 0.0, -2.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "trapped")
        self.assertTrue(result["trapped"])
        self.assertGreater(result["net_progress"], 0.24)
        self.assertEqual(result["trap_reason"], "probes_exhausted")
        self.assertEqual(result["tryout_probe_count"], 6)
        self.assertEqual(sim.forward_calls, 8)

    def test_moving_probe_with_stationary_tail_is_trapped(self):
        events = [
            ((0.0, 0.0, 0.0), True),   # direct path is blocked
            ((0.2, 0.0, 0.0), True),   # sliding probe has real motion
            ((0.0, 0.0, 0.0), True),   # but its committed tail is blocked
        ]
        env, sim = _controller_env(events)

        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "trapped")
        self.assertEqual(result["trap_reason"], "tail_blocked")
        self.assertEqual(result["tryout_probe_count"], 1)
        self.assertEqual(sim.forward_calls, 3)

    def test_short_of_target_with_progress_does_not_retry(self):
        events = [((0.0, 0.0, -0.1), True)] * 4
        env, sim = _controller_env(events)

        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "progressed")
        self.assertFalse(result["trapped"])
        self.assertFalse(result["tryout_started"])
        self.assertGreater(result["net_progress"], 0.39)
        self.assertEqual(sim.forward_calls, 4)

    def test_tryout_disabled_never_probes_or_reports_trapped(self):
        env, sim = _controller_env([((0.0, 0.0, 0.0), True)])

        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), False, None, reach_tolerance=0.1
        )

        self.assertEqual(result["outcome"], "inconclusive")
        self.assertFalse(result["trapped"])
        self.assertFalse(result["tryout_started"])
        self.assertEqual(result["tryout_probe_count"], 0)
        self.assertEqual(sim.forward_calls, 1)

    def test_multi_step_keeps_etpnav_best_effort_path_after_nonarrival(self):
        env, _ = _controller_env()
        scripted = deque([
            _result("reached"), _result("progressed", 0.8),
            _result("trapped"),
        ])
        calls = []

        def single_step(
            position,
            tryout,
            vis_info,
            reach_tolerance,
            horizontal_only=True,
        ):
            calls.append(tuple(position))
            self.assertFalse(horizontal_only)
            return scripted.popleft()

        env.single_step_control = single_step
        path = [
            ("n1", (1.0, 0.0, 0.0)),
            ("n2", (2.0, 0.0, 0.0)),
            ("n3", (3.0, 0.0, 0.0)),
        ]

        result = env.multi_step_control(path, True, None, 0.5)

        self.assertEqual(result["outcome"], "trapped")
        self.assertFalse(result["reached"])
        self.assertEqual(result["last_reached_vp"], "n1")
        self.assertEqual(len(calls), 3)

    def test_back_path_does_not_reach_same_xz_on_another_floor(self):
        env, sim = _controller_env(position=(2.0, 0.0, -3.0))

        result = env.multi_step_control(
            [("upper", (2.0, 4.0, -3.0))],
            False,
            None,
            reach_tolerance=0.1,
        )

        self.assertEqual(result["outcome"], "inconclusive")
        self.assertFalse(result["reached"])
        self.assertIsNone(result["last_reached_vp"])
        self.assertEqual(result["remaining_distance_xz"], 0.0)
        self.assertEqual(result["remaining_distance_3d"], 4.0)
        self.assertEqual(sim.forward_calls, 0)


class DistanceBudgetMotionTest(unittest.TestCase):
    def test_default_preserves_legacy_half_metre_early_stop(self):
        for distance, expected in ((0.5, 0), (1.0, 2), (2.0, 6)):
            with self.subTest(distance=distance):
                env, sim = _geometric_controller_env('legacy_tolerance')
                del env.controller_motion_mode
                result = env.single_step_control(
                    np.array([0.0, 0.0, -distance]), True, None
                )
                self.assertTrue(result['reached'])
                self.assertEqual(sim.forward_calls, expected)

    def test_direct_steps_match_etpnav_budget_at_multiple_headings(self):
        # Cardinal targets have exactly representable distances, so this tests
        # the original floor(distance / 0.25) budget without floating-point
        # rounding of the input target changing the expected step count.
        directions = ((0, 0, -1), (-1, 0, 0), (0, 0, 1), (1, 0, 0))
        for distance, expected in ((0.5, 2), (1.0, 4), (2.0, 8)):
            for initial_heading in (0, 30, 90, 180, 270):
                for direction in directions:
                    with self.subTest(
                        distance=distance, heading=initial_heading,
                        direction=direction,
                    ):
                        env, sim = _geometric_controller_env(
                            'etpnav_distance',
                            heading=math.radians(initial_heading),
                        )
                        result = env.single_step_control(
                            np.asarray(direction, dtype=np.float64) * distance,
                            True, None,
                        )
                        self.assertEqual(sim.forward_calls, expected)
                        self.assertLess(result['remaining_distance_xz'], 1e-6)
                        self.assertTrue(result['reached'])
                        self.assertFalse(result['tryout_started'])

    def test_arrival_tolerance_no_longer_changes_forward_budget(self):
        for tolerance in (0.01, 0.5, 1.5):
            with self.subTest(tolerance=tolerance):
                env, sim = _geometric_controller_env('etpnav_distance')
                result = env.single_step_control(
                    np.array([0.0, 0.0, -1.0]), True, None,
                    reach_tolerance=tolerance,
                )
                self.assertEqual(sim.forward_calls, 4)
                self.assertTrue(result['reached'])

    def test_fractional_step_remainder_is_not_rounded_up_or_probed(self):
        for distance, expected in ((0.1, 0), (0.49, 1), (0.74, 2)):
            with self.subTest(distance=distance):
                env, sim = _geometric_controller_env('etpnav_distance')
                result = env.single_step_control(
                    np.array([0.0, 0.0, -distance]), True, None,
                    reach_tolerance=0.01,
                )
                self.assertEqual(sim.forward_calls, expected)
                self.assertFalse(result['tryout_started'])
                self.assertFalse(result['reached'])
                self.assertLessEqual(-float(sim.position[2]), distance)

    def test_exact_target_requires_neither_turn_nor_forward(self):
        env, sim = _geometric_controller_env(
            'etpnav_distance', position=(2.0, 0.0, -3.0), heading=1.3
        )
        result = env.single_step_control(
            np.array([2.0, 4.0, -3.0]), True, None
        )
        self.assertTrue(result['reached'])
        self.assertEqual(sim.forward_calls, 0)
        self.assertEqual(env.turn_actions, [])

    def test_same_xz_upper_floor_does_not_trigger_zero_budget_tryout(self):
        env, sim = _geometric_controller_env('etpnav_distance')
        result = env.single_step_control(
            np.array([0.0, 4.0, 0.0]), True, None, horizontal_only=False
        )
        self.assertFalse(result['reached'])
        self.assertFalse(result['tryout_started'])
        self.assertEqual(sim.forward_calls, 0)

    def test_sliding_collision_does_not_shorten_fixed_distance_budget(self):
        env, sim = _geometric_controller_env(
            'etpnav_distance', events=[((0.0, 0.0, -0.25), True)] * 4
        )
        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None
        )
        self.assertEqual(sim.forward_calls, 4)
        self.assertEqual(result['collision_count'], 4)
        self.assertFalse(result['tryout_started'])
        self.assertFalse(result['trapped'])

    def test_tryout_tail_keeps_official_remaining_step_budget(self):
        # A perpendicular moving probe leaves all four planned forward steps.
        # V2 stops its tail after three because the 0.5 m arrival ball is hit;
        # distance mode preserves the fourth step, ending 0.25 m off-axis.
        for mode, expected in (('legacy_tolerance', 5), ('etpnav_distance', 6)):
            with self.subTest(mode=mode):
                env, sim = _geometric_controller_env(
                    mode,
                    events=[((0.0, 0.0, 0.0), True),
                            ((-0.25, 0.0, 0.0), True)],
                )
                result = env.single_step_control(
                    np.array([0.0, 0.0, -1.0]), True, None
                )
                self.assertEqual(sim.forward_calls, expected)
                self.assertEqual(result['tryout_probe_count'], 1)
                self.assertTrue(result['reached'])
                self.assertFalse(result['trapped'])

    def test_exact_arrival_during_tryout_tail_stops_before_overshoot(self):
        env, sim = _geometric_controller_env(
            'etpnav_distance',
            events=[((0.0, 0.0, 0.0), True), ((0.0, 0.0, -0.25), True)],
        )
        result = env.single_step_control(
            np.array([0.0, 0.0, -0.5]), True, None
        )
        self.assertEqual(sim.forward_calls, 3)
        self.assertTrue(result['reached'])
        self.assertLess(result['remaining_distance_xz'], 1e-6)

    def test_stationary_tryout_tail_remains_a_confirmed_trap(self):
        env, sim = _geometric_controller_env(
            'etpnav_distance',
            events=[((0.0, 0.0, 0.0), True), ((-0.25, 0.0, 0.0), True),
                    ((0.0, 0.0, 0.0), False)],
        )
        result = env.single_step_control(
            np.array([0.0, 0.0, -1.0]), True, None
        )
        self.assertEqual(sim.forward_calls, 3)
        self.assertTrue(result['trapped'])
        self.assertEqual(result['trap_reason'], 'tail_blocked')

    def test_both_real_controllers_try_six_probes_then_return_observation(self):
        env, sim = _geometric_controller_env(
            'etpnav_distance', events=[((0.0, 0.0, 0.0), False)] * 14
        )
        env.get_observation_at = lambda position, rotation: {
            'position': tuple(np.asarray(position).tolist())
        }
        env.get_reward = lambda observations: 0.0
        env.get_done = lambda observations: False
        env.get_info = lambda observations: {}
        attempts = [
            {'ghost_vp': 'g0', 'ghost_pos': np.array([0.0, 0.0, -1.0]),
             'front_vp': '0', 'front_pos': np.zeros(3), 'back_path': [],
             'tryout': True},
            {'ghost_vp': 'g1', 'ghost_pos': np.array([1.0, 0.0, 0.0]),
             'front_vp': '0', 'front_pos': np.zeros(3), 'back_path': [],
             'routes_from': {'0': []}, 'tryout': True},
        ]
        obs, _, done, info = env.step(
            {'act': 4, 'cur_vp': '0', 'attempts': attempts}, None
        )
        result = info['reselection']
        self.assertEqual(sim.forward_calls, 14)
        self.assertEqual(result['attempted_ghosts'], ['g0', 'g1'])
        self.assertEqual(result['trapped_ghosts'], ['g0', 'g1'])
        self.assertEqual(result['fallback'], 'etpnav_next_observation')
        self.assertEqual(
            [item['move_result']['tryout_probe_count']
             for item in result['details']], [6, 6]
        )
        self.assertEqual(obs['position'], (0.0, 0.0, 0.0))
        self.assertFalse(done)
        # The second target is recomputed from the live heading left by the
        # first six-probe sweep, not from the original observation's heading.
        self.assertAlmostEqual(sim.rotation, math.pi)

    def test_unknown_motion_mode_fails_closed(self):
        env, sim = _geometric_controller_env('typo')
        with self.assertRaisesRegex(ValueError, 'controller_motion_mode'):
            env.single_step_control(np.array([0.0, 0.0, -1.0]), True, None)
        self.assertEqual(sim.forward_calls, 0)


class ReselectionStepIntegrationTest(unittest.TestCase):
    def make_env(self, target_results=(), route_results=()):
        env = object.__new__(ExtractedEnv)
        sim = _FakeSim()
        env._env = SimpleNamespace(sim=sim)
        env.video_option = False
        env.target_results = deque(target_results)
        env.route_results = deque(route_results)
        env.control_targets = []
        env.control_horizontal_only = []
        env.control_tryout = []
        env.route_calls = []
        env.teleports = []

        def single_step(
            position,
            tryout,
            vis_info,
            reach_tolerance,
            horizontal_only=True,
        ):
            env.control_targets.append(tuple(np.asarray(position).tolist()))
            env.control_horizontal_only.append(horizontal_only)
            env.control_tryout.append(tryout)
            if not env.target_results:
                raise AssertionError("unexpected single_step_control call")
            return env.target_results.popleft()

        def multi_step(path, tryout, vis_info, reach_tolerance=0.5):
            env.route_calls.append(path)
            if not env.route_results:
                raise AssertionError("unexpected multi_step_control call")
            return env.route_results.popleft()

        def teleport(position):
            env.teleports.append(tuple(np.asarray(position).tolist()))
            sim.set_agent_state(position, 0.0)

        env.single_step_control = single_step
        env.multi_step_control = multi_step
        env.teleport = teleport
        env.get_observation_at = lambda position, rotation: {
            "position": tuple(np.asarray(position).tolist())
        }
        env.get_reward = lambda observations: 0.0
        env.get_done = lambda observations: False
        env.get_info = lambda observations: {}
        return env

    @staticmethod
    def run_step(env, attempts, reach_tolerance=0.5):
        action = {
            "act": 4,
            "cur_vp": "0",
            "attempts": attempts,
            "reach_tolerance": reach_tolerance,
        }
        return env.step(action, None)

    def test_progressed_primary_does_not_retry_or_blacklist(self):
        env = self.make_env([_result("progressed", 0.3)])
        attempts = [_attempt("g0", "0", 1.0), _attempt("g1", "0", 2.0)]

        _, _, _, info = self.run_step(env, attempts)
        result = info["reselection"]

        self.assertEqual(result["outcome"], "progressed")
        self.assertEqual(result["attempted_ghosts"], ["g0"])
        self.assertEqual(result["failed_ghosts"], [])
        self.assertEqual(result["trapped_ghosts"], [])
        self.assertEqual(len(env.control_targets), 1)
        self.assertEqual(env.control_horizontal_only, [True])

    def test_inconclusive_primary_does_not_retry(self):
        # This is the high-level behavior produced when Tryout is disabled and
        # the low-level action neither arrives nor confirms a trap.
        env = self.make_env([_result("inconclusive", 0.8)])
        attempts = [_attempt("g0", "0", 1.0), _attempt("g1", "0", 2.0)]

        _, _, _, info = self.run_step(env, attempts)
        result = info["reselection"]

        self.assertEqual(result["outcome"], "inconclusive")
        self.assertEqual(result["attempted_ghosts"], ["g0"])
        self.assertEqual(result["failed_ghosts"], [])
        self.assertEqual(result["unattempted_ghosts"], ["g1"])
        self.assertEqual(len(env.control_targets), 1)
        self.assertEqual(env.control_horizontal_only, [True])

    def test_only_trapped_primary_advances_to_reached_secondary(self):
        env = self.make_env([_result("trapped"), _result("reached")])
        attempts = [_attempt("g0", "0", 1.0), _attempt("g1", "0", 2.0)]

        _, _, _, info = self.run_step(env, attempts)
        result = info["reselection"]

        self.assertTrue(result["success"])
        self.assertEqual(result["selected_ghost"], "g1")
        self.assertEqual(result["failed_ghosts"], ["g0"])
        self.assertEqual(result["trapped_ghosts"], ["g0"])
        self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
        self.assertEqual(result["selected_front_vp"], "0")
        self.assertEqual(len(env.control_targets), 2)
        self.assertEqual(env.control_horizontal_only, [True, True])

    def test_failed_retry_route_still_attempts_secondary_with_tryout(self):
        route_result = _result("progressed", 0.8)
        route_result['last_reached_vp'] = None
        env = self.make_env(
            [_result("trapped"), _result("reached")], [route_result]
        )
        attempts = [
            _attempt("g0", "0", 1.0),
            _attempt("g1", "1", 2.0, back_path=[("1", (1.5, 0.0, 0.0))]),
        ]

        _, _, _, info = self.run_step(env, attempts)
        result = info["reselection"]

        self.assertTrue(result["success"])
        self.assertEqual(result["failed_ghosts"], ["g0"])
        self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
        self.assertEqual(result["unattempted_ghosts"], [])
        self.assertIsNone(result["abort_reason"])
        self.assertTrue(result["details"][1]["target_attempted"])
        self.assertIsNone(result["details"][1]["return_result"])
        # The planned front "1" was NOT reached, even though its target was.
        self.assertEqual(result["selected_front_vp"], "0")
        self.assertEqual(len(env.control_targets), 2)
        self.assertEqual(env.control_horizontal_only, [True, True])
        self.assertEqual(env.control_tryout, [True, True])
        self.assertEqual(len(env.route_calls), 1)

    def test_retry_routes_from_last_physically_reached_anchor(self):
        primary_route = _result("reached")
        primary_route['last_reached_vp'] = '3'
        secondary_route = _result("inconclusive", 3.0)
        secondary_route['last_reached_vp'] = None
        env = self.make_env(
            [_result("trapped"), _result("trapped")],
            [primary_route, secondary_route],
        )
        attempts = [
            _attempt("g0", "3", 1.0, back_path=[('3', (0.5, 0, 0))]),
            _attempt(
                "g1",
                "1",
                2.0,
                back_path=[("1", (1.5, 0.0, 0.0))],
            ),
        ]
        actual_path = [('3', (0.5, 0, 0)), ('1', (1.5, 0, 0))]
        attempts[1]['routes_from'] = {'3': actual_path}

        _, _, _, info = self.run_step(env, attempts, reach_tolerance=0.1)
        result = info["reselection"]
        self.assertEqual(env.route_calls[1], actual_path)
        self.assertEqual(result["failed_ghosts"], ["g0", "g1"])
        self.assertEqual(result["attempted_ghosts"], ["g0", "g1"])
        self.assertEqual(result["unattempted_ghosts"], [])
        self.assertIsNone(result["abort_reason"])
        self.assertEqual(result["failure_anchor_vp"], '3')
        self.assertEqual(result["fallback"], 'etpnav_next_observation')

    def test_failed_back_route_does_not_skip_target_tryout(self):
        route_result = {
            "outcome": "trapped",
            "reached": False,
            "trapped": True,
            "last_reached_vp": None,
            "remaining_distance": 1.0,
            "travelled_distance": 0.0,
            "collision_count": 1,
            "tryout_exhausted": True,
            "trap_reason": "probes_exhausted",
        }
        env = self.make_env([_result('progressed')], [route_result])
        attempts = [
            _attempt("g0", "1", 1.0, back_path=[("1", (0.5, 0.0, 0.0))])
        ]

        _, _, _, info = self.run_step(env, attempts)
        result = info["reselection"]

        self.assertEqual(result["failed_ghosts"], [])
        self.assertEqual(result["attempted_ghosts"], ['g0'])
        self.assertEqual(result["unattempted_ghosts"], [])
        self.assertIsNone(result["abort_reason"])
        self.assertTrue(result["details"][0]["target_attempted"])
        self.assertEqual(env.control_tryout, [True])
        self.assertEqual(len(env.route_calls), 1)

    def test_explicit_direct_secondary_uses_live_pose_without_return_or_teleport(self):
        env = self.make_env([_result('trapped'), _result('reached')])
        attempts = [_attempt('g0', '0', 1.0), _attempt('g1', 'far', 2.0)]
        action = {'act': 4, 'cur_vp': '0', 'attempts': attempts,
                  'reselection_route_mode': 'direct'}
        _, _, _, info = env.step(action, None)
        result = info['reselection']
        self.assertEqual(result['selected_ghost'], 'g1')
        self.assertEqual(result['selected_front_vp'], '0')
        self.assertEqual(len(env.teleports), 1)  # primary training teleport only
        self.assertEqual(env.route_calls, [])
        self.assertEqual(env.control_tryout, [True, True])


if __name__ == "__main__":
    unittest.main()
