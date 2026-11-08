"""Pure logic and controller-state tests for Safe-VLN re-selection."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vlnce_baselines"
    / "common"
    / "reselection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "safe_vln_reselection", str(MODULE_PATH)
)
reselection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reselection
SPEC.loader.exec_module(reselection)


class SafeVLNTeacherTest(unittest.TestCase):
    def test_teacher_is_stable_by_total_cost_then_action_index(self):
        # Inputs are intentionally out of order.  g1/g2 tie and action index
        # makes g1 the stable optimal label; g2 is the sub-optimal label.
        primary, secondary = reselection.stable_teacher_top2([
            (5, "g3", 8.0),
            (4, "g2", 3.0),
            (3, "g1", 3.0),
        ])
        self.assertEqual((primary, secondary), (3, 4))

    def test_stop_and_single_candidate_ignore_secondary(self):
        self.assertEqual(
            reselection.stable_teacher_top2([(2, "g0", 1.0)], stop=True),
            (0, -100),
        )
        self.assertEqual(
            reselection.stable_teacher_top2([(2, "g0", 1.0)]),
            (2, -100),
        )
        self.assertEqual(reselection.stable_teacher_top2([]), (-100, -100))


class SafeVLNReselectionTest(unittest.TestCase):
    @staticmethod
    def fake_graph():
        graph = SimpleNamespace(node_pos={'0': (), '3': ()},
                                ghost_pos={'g0': (), 'g1': (), 'g2': ()},
                                deleted=[])
        def delete(ghost):
            graph.deleted.append(ghost)
            del graph.ghost_pos[ghost]
        graph.delete_ghost = delete
        return graph

    def test_etpnav_consumes_both_failed_attempts_but_not_unused_candidate(self):
        graph = self.fake_graph()
        result = {'success': False, 'failure_anchor_vp': '3',
                  'attempted_ghosts': ['g0', 'g1'], 'trapped_ghosts': ['g0', 'g1']}
        anchor = reselection.commit_etpnav_attempts(graph, '0', result)
        self.assertEqual(anchor, '3')
        self.assertEqual(graph.deleted, ['g0', 'g1'])
        self.assertEqual(set(graph.ghost_pos), {'g2'})

    def test_etpnav_consumes_progressed_primary_only(self):
        graph = self.fake_graph()
        result = {'success': False, 'failure_anchor_vp': '0',
                  'attempted_ghosts': ['g0'], 'outcome': 'progressed'}
        reselection.commit_etpnav_attempts(graph, '0', result)
        self.assertEqual(graph.deleted, ['g0'])

    def test_consume_disabled_and_invalid_anchor_do_not_mutate_graph(self):
        graph = self.fake_graph()
        result = {'success': True, 'selected_front_vp': '3',
                  'attempted_ghosts': ['g0']}
        self.assertEqual(reselection.commit_etpnav_attempts(
            graph, '0', result, consume_ghost=False), '3')
        self.assertEqual(graph.deleted, [])
        result['selected_front_vp'] = 'unvisited'
        with self.assertRaises(ValueError):
            reselection.commit_etpnav_attempts(graph, '0', result)
        self.assertEqual(graph.deleted, [])

    def test_diagnostics_count_real_secondary_tryout_and_double_failure(self):
        details = [
            {'target_attempted': True, 'outcome': 'trapped',
             'move_result': {'tryout_started': True},
             'back_result': {'reached': True, 'outcome': 'reached'}},
            {'target_attempted': True, 'outcome': 'trapped',
             'move_result': {'tryout_started': True},
             'back_result': {'reached': False, 'outcome': 'progressed'}},
        ]
        stats = reselection.reselection_diagnostics({
            'details': details, 'fallback': 'etpnav_next_observation',
            'unattempted_ghosts': [],
        })
        self.assertEqual(stats['control_secondary_attempts'], 1)
        self.assertEqual(stats['control_secondary_tryouts'], 1)
        self.assertEqual(stats['control_double_trap_fallbacks'], 1)
        self.assertEqual(stats['control_route_nonarrivals'], 1)
        self.assertEqual(stats['control_skipped_secondary_after_trap'], 0)

    def test_trainer_tryout_gate_is_independent_of_sliding(self):
        # Compile the actual rollout assignment to catch regressions where
        # enabling sliding silently disables the turn probes again.
        trainer_path = MODULE_PATH.parents[1] / "ss_trainer_ETP.py"
        tree = ast.parse(trainer_path.read_text())
        assignments = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "use_tryout"
                    for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        module = ast.Module(body=assignments, type_ignores=[])
        code = compile(ast.fix_missing_locations(module), str(trainer_path), "exec")
        for sliding in (False, True):
            for tryout in (False, True):
                with self.subTest(sliding=sliding, tryout=tryout):
                    config = SimpleNamespace(
                        IL=SimpleNamespace(tryout=tryout),
                        TASK_CONFIG=SimpleNamespace(SIMULATOR=SimpleNamespace(
                            HABITAT_SIM_V0=SimpleNamespace(ALLOW_SLIDING=sliding)
                        )),
                    )
                    namespace = {"self": SimpleNamespace(config=config)}
                    exec(code, namespace)
                    self.assertEqual(namespace["use_tryout"], tryout)

    def test_rank_keeps_selected_then_uses_planner_logits(self):
        vp_ids = [None, "0", "g0", "g1", "g2", "g3"]
        logits = [0.2, float("-inf"), 0.1, 0.9, 0.9, float("nan")]

        ranked = reselection.rank_reselection_candidates(
            selected_action=2,
            gmap_vp_ids=vp_ids,
            planner_logits=logits,
            max_retries=2,
        )
        # g1/g2 tie is stable by graph action index; NaN is never retried.
        self.assertEqual(ranked, [2, 3, 4])

        ranked_without_failed = reselection.rank_reselection_candidates(
            selected_action=2,
            gmap_vp_ids=vp_ids,
            planner_logits=logits,
            blacklisted_ghosts={"g1"},
            max_retries=2,
        )
        self.assertEqual(ranked_without_failed, [2, 4])

    def test_stop_visited_and_blacklisted_actions_have_no_attempt(self):
        ids = [None, "0", "g0"]
        logits = [3.0, 2.0, 1.0]
        self.assertEqual(
            reselection.rank_reselection_candidates(0, ids, logits), []
        )
        self.assertEqual(
            reselection.rank_reselection_candidates(1, ids, logits), []
        )
        self.assertEqual(
            reselection.rank_reselection_candidates(
                2, ids, logits, blacklisted_ghosts={"g0"}
            ),
            [],
        )

    def test_controller_blacklists_trapped_target_and_stops_on_success(self):
        controller = reselection.ReselectionController(["g0", "g1", "g2"])
        self.assertEqual(controller.next_ghost, "g0")
        controller.record_target_outcome("g0", "trapped")
        self.assertEqual(controller.next_ghost, "g1")
        controller.record_target_outcome("g1", "reached")

        self.assertTrue(controller.done)
        self.assertEqual(controller.failed, ["g0"])
        self.assertEqual(controller.succeeded, "g1")
        self.assertEqual(controller.summary(), {
            "success": True,
            "selected_ghost": "g1",
            "attempted_ghosts": ["g0", "g1"],
            "failed_ghosts": ["g0"],
            "attempts": 2,
            "trapped_ghosts": ["g0"],
            "unattempted_ghosts": ["g2"],
            "outcome": "reached",
            "abort_reason": None,
        })

    def test_controller_reports_all_attempts_failed(self):
        controller = reselection.ReselectionController(["g0", "g1"])
        controller.record_target_outcome("g0", "trapped")
        controller.record_target_outcome("g1", "trapped")

        self.assertTrue(controller.done)
        self.assertIsNone(controller.succeeded)
        self.assertEqual(controller.failed, ["g0", "g1"])
        self.assertFalse(controller.summary()["success"])

        with self.assertRaises(RuntimeError):
            controller.record_target_outcome("g1", "trapped")

    def test_partial_progress_and_uncertainty_do_not_retry_or_blacklist(self):
        for outcome in ("progressed", "inconclusive"):
            with self.subTest(outcome=outcome):
                controller = reselection.ReselectionController(["g0", "g1"])
                controller.record_target_outcome("g0", outcome)
                self.assertTrue(controller.done)
                summary = controller.summary()
                self.assertEqual(summary["attempted_ghosts"], ["g0"])
                self.assertEqual(summary["trapped_ghosts"], [])
                self.assertEqual(summary["unattempted_ghosts"], ["g1"])
                self.assertFalse(summary["success"])
                self.assertEqual(summary["outcome"], outcome)

    def test_failed_back_path_does_not_count_as_target_attempt(self):
        controller = reselection.ReselectionController(["g0", "g1"])
        controller.abort("back_path_failed")
        self.assertTrue(controller.done)
        summary = controller.summary()
        self.assertEqual(summary["attempted_ghosts"], [])
        self.assertEqual(summary["failed_ghosts"], [])
        self.assertEqual(summary["unattempted_ghosts"], ["g0", "g1"])
        self.assertEqual(summary["outcome"], "route_failed")
        self.assertEqual(summary["abort_reason"], "back_path_failed")

    def test_failed_return_does_not_blacklist_unattempted_alternative(self):
        controller = reselection.ReselectionController(["g0", "g1"])
        controller.record_target_outcome("g0", "trapped")
        controller.abort("return_failed")
        summary = controller.summary()
        self.assertEqual(summary["attempted_ghosts"], ["g0"])
        self.assertEqual(summary["trapped_ghosts"], ["g0"])
        self.assertEqual(summary["unattempted_ghosts"], ["g1"])
        self.assertEqual(summary["abort_reason"], "return_failed")

    def test_invalid_results_do_not_mutate_state(self):
        controller = reselection.ReselectionController(["g0", "g0", "g1"])
        self.assertEqual(controller.candidates, ("g0", "g1"))
        before = controller.summary()
        with self.assertRaises(ValueError):
            controller.record_target_outcome("g1", "trapped")
        with self.assertRaises(ValueError):
            controller.record_target_outcome("g0", "failed")
        self.assertEqual(controller.summary(), before)

    def test_empty_sequence_is_finished(self):
        controller = reselection.ReselectionController([])
        self.assertTrue(controller.done)
        self.assertEqual(controller.summary()["attempts"], 0)
        with self.assertRaises(RuntimeError):
            controller.abort()

    def test_graph_anchor_never_uses_a_failed_front(self):
        self.assertEqual(
            reselection.resolve_graph_anchor(
                current_vp="7",
                selected_front_vp=None,
                succeeded=False,
                failure_anchor_vp="3",
            ),
            "3",
        )
        # Older environments did not report a confirmed failure anchor.
        self.assertEqual(
            reselection.resolve_graph_anchor(
                current_vp="7",
                selected_front_vp=None,
                succeeded=False,
            ),
            "7",
        )
        self.assertEqual(
            reselection.resolve_graph_anchor(
                current_vp="7",
                selected_front_vp="3",
                succeeded=True,
            ),
            "3",
        )
        with self.assertRaises(ValueError):
            reselection.resolve_graph_anchor("7", None, succeeded=True)


if __name__ == "__main__":
    unittest.main()
