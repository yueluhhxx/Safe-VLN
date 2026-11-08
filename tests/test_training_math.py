"""CPU checks for loss equivalence, exact schedule, and teacher semantics."""

import ast
from collections import defaultdict
from contextlib import nullcontext
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_helper(name):
    path = ROOT / "vlnce_baselines" / "common" / (name + ".py")
    spec = importlib.util.spec_from_file_location("test_" + name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training_math = load_helper("training_math")
teacher_cache = load_helper("teacher_cache")
reselection = load_helper("reselection")


class SamplingScheduleTest(unittest.TestCase):
    def test_decay_is_at_optimizer_iteration_boundary(self):
        for iteration, exponent in [(0, 1), (2999, 1), (3000, 2), (8999, 3),
                                    (9000, 4), (14999, 5), (15000, 6)]:
            self.assertEqual(
                training_math.scheduled_sampling_probability(.75, 3000, iteration),
                .75 ** exponent,
            )

    def test_checkpoint_interval_does_not_delay_decay(self):
        ratios = {}
        for block_start in range(0, 20000, 2000):
            interval_ratio = training_math.scheduled_sampling_probability(.75, 3000, block_start)
            for offset in range(2000):
                iteration = block_start + offset
                ratios[iteration] = training_math.iteration_sampling_probability(
                    .75, 3000, iteration, "per_iteration", interval_ratio
                )
        for iteration in [3000, 9000, 15000]:
            self.assertEqual(ratios[iteration], .75 ** (iteration // 3000 + 1))
            self.assertNotEqual(ratios[iteration], ratios[iteration - 1])

    def test_resume_and_legacy_schedule_are_explicit(self):
        old = training_math.scheduled_sampling_probability(.75, 3000, 14000)
        self.assertEqual(training_math.iteration_sampling_probability(
            .75, 3000, 15000, "legacy_interval", old), old)
        self.assertEqual(training_math.iteration_sampling_probability(
            .75, 3000, 15000, "per_iteration", old), .75 ** 6)
        with self.assertRaises(ValueError):
            training_math.iteration_sampling_probability(.75, 3000, 0, "unknown", .75)

    def test_invalid_schedule_inputs_fail_closed(self):
        for base, decay, iteration in [(1.1, 3000, 0), (-.1, 3000, 0), (.75, 0, 0), (.75, 3000, -1)]:
            with self.assertRaises(ValueError):
                training_math.scheduled_sampling_probability(base, decay, iteration)

    def test_actual_trainer_updates_counter_across_managed_intervals(self):
        source = (ROOT / "vlnce_baselines" / "ss_trainer_ETP.py").read_text()
        trainer_class = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "RLTrainer")
        method = next(node for node in trainer_class.body if isinstance(node, ast.FunctionDef) and node.name == "_train_interval")
        scope = {"defaultdict": defaultdict, "deepcopy": copy.deepcopy, "autocast": nullcontext,
                 "iteration_sampling_probability": training_math.iteration_sampling_probability}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "train_interval", "exec"), scope)
        encoder = SimpleNamespace(eval=lambda: None)
        policy = SimpleNamespace(train=lambda: None, net=SimpleNamespace(
            module=SimpleNamespace(rgb_encoder=encoder, depth_encoder=encoder)))
        trainer = SimpleNamespace(policy=policy, waypoint_predictor=encoder, world_size=8, local_rank=1,
            config=SimpleNamespace(IL=SimpleNamespace(sample_ratio=.75, decay_interval=3000,
                                                      sampling_schedule="per_iteration")),
            optimizer=SimpleNamespace(zero_grad=lambda: None),
            scaler=SimpleNamespace(scale=lambda loss: SimpleNamespace(backward=lambda: None),
                                   step=lambda optimizer: None, update=lambda: None),
            _global_train_iteration=2998)
        ratios = []
        trainer.rollout = lambda mode, weight, ratio: ratios.append((trainer._global_train_iteration, ratio))
        scope["_train_interval"](trainer, 4, 1., .75)
        scope["_train_interval"](trainer, 2, 1., .75)
        self.assertEqual(ratios, [(2998, .75), (2999, .75), (3000, .75 ** 2),
                                  (3001, .75 ** 2), (3002, .75 ** 2), (3003, .75 ** 2)])
        self.assertEqual(trainer._global_train_iteration, 3004)


class ReferenceGroupLossTest(unittest.TestCase):
    counts = [3, 5, 11, 17, 2, 8, 19, 23]

    def setUp(self):
        torch.manual_seed(7)
        self.base = nn.Linear(5, 4).double()
        self.inputs = [torch.randn(n, 5, dtype=torch.float64) for n in self.counts]
        self.labels = [torch.randint(0, 4, (n,)) for n in self.counts]
        self.secondary = [torch.where(torch.arange(n) % 3 == 0, torch.full((n,), -100), (y + 1) % 4)
                          for n, y in zip(self.counts, self.labels)]

    def run_update(self, objective):
        model = copy.deepcopy(self.base)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        # Nonzero momentum state avoids first-step Adam masking loss-scale bugs.
        sum(p.square().sum() for p in model.parameters()).backward()
        optimizer.step()
        optimizer.zero_grad()
        losses = [.8 * nn.functional.cross_entropy(model(x), y, reduction="sum")
                  + .2 * nn.functional.cross_entropy(model(x), secondary, reduction="sum", ignore_index=-100)
                  for x, y, secondary in zip(self.inputs, self.labels, self.secondary)]
        if objective == "reference":
            loss = (sum(losses[:4]) / sum(self.counts[:4]) + sum(losses[4:]) / sum(self.counts[4:])) / 2
        elif objective == "virtual":
            normalized = []
            for rank in range(8):
                def reduce_counts(counts, rank=rank):
                    self.assertEqual(counts.dtype, torch.int64)
                    expected = [0, 0]
                    expected[rank // 4] = self.counts[rank]
                    self.assertEqual(counts.tolist(), expected)
                    counts[:] = torch.tensor([sum(self.counts[:4]), sum(self.counts[4:])])
                normalized.append(training_math.normalize_rollout_loss(
                    losses[rank], self.counts[rank], 8, rank, 2, all_reduce=reduce_counts))
            loss = sum(normalized) / 8  # DDP gradient averaging.
        else:
            loss = sum(loss / count for loss, count in zip(losses, self.counts)) / 8
        loss.backward()
        gradient = torch.cat([p.grad.flatten() for p in model.parameters()])
        optimizer.step()
        parameters = torch.cat([p.flatten() for p in model.parameters()])
        return gradient, parameters

    def test_virtual_eight_ranks_match_two_reference_gradients_and_adamw(self):
        expected_gradient, expected_parameters = self.run_update("reference")
        gradient, parameters = self.run_update("virtual")
        torch.testing.assert_close(gradient, expected_gradient, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(parameters, expected_parameters, atol=1e-14, rtol=1e-14)

    def test_naive_local_mean_is_not_reference_objective(self):
        expected_gradient, expected_parameters = self.run_update("reference")
        gradient, parameters = self.run_update("legacy")
        self.assertGreater((gradient - expected_gradient).abs().max().item(), .01)
        self.assertGreater((parameters - expected_parameters).abs().max().item(), 1e-8)

    def test_legacy_and_one_rank_groups_need_no_collective(self):
        for groups in (0, 8):
            reducer = mock.Mock(side_effect=AssertionError("unneeded collective"))
            loss = training_math.normalize_rollout_loss(torch.tensor(12.), 3, 8, 5, groups, reducer)
            self.assertEqual(loss.item(), 4.)
            reducer.assert_not_called()

    def test_invalid_layout_fails_before_collective(self):
        for world, rank, groups in [(8, 0, 3), (1, 0, 2), (8, 8, 2), (8, 0, -1), (0, 0, 0)]:
            with self.assertRaises(ValueError):
                training_math.reference_group_layout(world, rank, groups)
        with self.assertRaises(ValueError):
            training_math.normalize_rollout_loss(torch.tensor(1.), 0, 8, 0, 2)

    def test_low_precision_sum_is_normalized_in_float32(self):
        def reduce_counts(counts):
            counts[:] = torch.tensor([100000, 100000])
        loss = training_math.normalize_rollout_loss(torch.tensor(100., dtype=torch.float16), 1, 8, 0, 2, reduce_counts)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertAlmostEqual(loss.item(), .004)


class TeacherPositionModeTest(unittest.TestCase):
    @staticmethod
    def teacher_method():
        source = (ROOT / "vlnce_baselines" / "ss_trainer_ETP.py").read_text()
        trainer = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "RLTrainer")
        method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_teacher_action_new")
        scope = {"torch": torch, "choose_teacher_positions": training_math.choose_teacher_positions,
                 "prepare_missing_position_batches": teacher_cache.prepare_missing_position_batches,
                 "update_geodesic_cache_batches": teacher_cache.update_geodesic_cache_batches,
                 "stable_teacher_top2": reselection.stable_teacher_top2}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "teacher_method", "exec"), scope)
        return scope["_teacher_action_new"]

    def run_teacher(self, mode, stop=False, prefill=True):
        positions = {"g0": [(8., 0., 0.), (1., 0., 0.)], "g1": [(3., 0., 0.)]}
        graph = SimpleNamespace(has_real_pos=True, ghost_real_pos=positions, ghost_aug_pos={},
                                ghost_is_available=lambda ghost: True)
        cache = teacher_cache.EpisodeGeodesicCache()
        if prefill:
            cache.update(positions["g0"] + positions["g1"], [8., 1., 3.])
        queries = []
        def call(methods, arguments):
            queries.extend(arguments[0]["positions"])
            return [[p[0] for p in arguments[0]["positions"]]]
        trainer = SimpleNamespace(config=SimpleNamespace(IL=SimpleNamespace(
            teacher_position_mode=mode, teacher_stop_distance=1.5, expert_policy="spl")),
            envs=SimpleNamespace(num_envs=1, call=call), gmaps=[graph],
            _teacher_goal_distance_caches=[cache], _teacher_current_goal_distances=[1. if stop else 10.],
            device=torch.device("cpu"))
        with mock.patch.object(training_math.random, "choice", side_effect=lambda values: values[0]) as choice:
            primary, secondary = self.teacher_method()(trainer, [[None, "v0", "g0", "g1"]], [False])
        return primary.tolist(), secondary.tolist(), choice.call_count, queries

    def test_representative_not_optimistic_minimum_is_used_for_labels(self):
        self.assertEqual(self.run_teacher("random_representative")[:3], ([3], [2], 2))
        self.assertEqual(self.run_teacher("minimum")[:3], ([2], [3], 0))

    def test_exact_cache_queries_only_selected_positions(self):
        primary, secondary, choices, queries = self.run_teacher("random_representative", prefill=False)
        self.assertEqual((primary, secondary, choices), ([3], [2], 2))
        self.assertEqual(queries, [(8., 0., 0.), (3., 0., 0.)])

    def test_stop_preserves_ignore_secondary_and_consumes_no_rng(self):
        self.assertEqual(self.run_teacher("random_representative", stop=True), ([0], [-100], 0, []))

    def test_position_mode_rejects_unknown_and_empty_inputs(self):
        with self.assertRaises(ValueError):
            training_math.choose_teacher_positions([], "minimum")
        with self.assertRaises(ValueError):
            training_math.choose_teacher_positions([(0, 0, 0)], "unknown")


if __name__ == "__main__":
    unittest.main()
