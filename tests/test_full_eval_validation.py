import gzip
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_full_eval import (
    EvaluationValidationError,
    compare_metrics,
    validate_full_evaluation,
)


class FullEvaluationValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir()
        self.dataset = self.root / "val_unseen.json.gz"
        self.checkpoint_index = 7
        self.world_size = 2

    def tearDown(self):
        self.temporary.cleanup()

    def write_dataset(self, episode_ids=(1, 2, 3, 4)):
        with gzip.open(self.dataset, "wt", encoding="utf-8") as output:
            json.dump(
                {"episodes": [{"episode_id": item} for item in episode_ids]},
                output,
            )

    @staticmethod
    def metrics(success, spl, distance):
        return {
            "success": success,
            "spl": spl,
            "ndtw": (success + spl) / 2.0,
            "sdtw": success * spl,
            "distance_to_goal": distance,
        }

    def rank_path(self, rank):
        return self.results_dir / (
            "stats_ep_ckpt_{}_val_unseen_r{}_w{}.json".format(
                self.checkpoint_index, rank, self.world_size
            )
        )

    def write_rank(self, rank, episodes):
        self.rank_path(rank).write_text(json.dumps(episodes), encoding="utf-8")

    def write_aggregate(self, metrics):
        path = self.results_dir / "stats_ckpt_7_val_unseen.json"
        path.write_text(json.dumps(metrics), encoding="utf-8")
        return path

    def valid_artifacts(self):
        self.write_dataset()
        first = {
            "1": self.metrics(1.0, 0.8, 1.0),
            "3": self.metrics(0.0, 0.0, 5.0),
        }
        second = {
            "2": self.metrics(1.0, 0.6, 2.0),
            "4": self.metrics(0.0, 0.0, 4.0),
        }
        self.write_rank(0, first)
        self.write_rank(1, second)
        all_metrics = list(first.values()) + list(second.values())
        aggregate = {
            name: sum(item[name] for item in all_metrics) / len(all_metrics)
            for name in all_metrics[0]
        }
        aggregate_path = self.write_aggregate(aggregate)
        return aggregate, aggregate_path

    def validate(self):
        return validate_full_evaluation(
            self.results_dir,
            self.dataset,
            checkpoint_index=self.checkpoint_index,
            world_size=self.world_size,
            expected_episode_count=4,
        )

    def test_complete_evaluation_passes_and_recomputes_means(self):
        aggregate, aggregate_path = self.valid_artifacts()

        report = self.validate()

        self.assertTrue(report["coverage_validated"])
        self.assertEqual(report["episode_count"], 4)
        self.assertEqual(report["rank_episode_counts"], {"0": 2, "1": 2})
        self.assertEqual(report["metrics"], aggregate)
        self.assertEqual(report["recomputed_metrics"], aggregate)
        self.assertEqual(report["aggregate_path"], str(aggregate_path.resolve()))

    def test_duplicate_episode_across_ranks_is_rejected(self):
        self.write_dataset()
        self.write_rank(0, {"1": self.metrics(1.0, 0.8, 1.0)})
        self.write_rank(1, {"1": self.metrics(0.0, 0.0, 5.0)})

        with self.assertRaisesRegex(EvaluationValidationError, "duplicate episode_id"):
            self.validate()

    def test_duplicate_episode_in_dataset_is_rejected(self):
        self.write_dataset((1, 2, 3, 3))

        with self.assertRaisesRegex(EvaluationValidationError, "dataset.*duplicate"):
            self.validate()

    def test_missing_episode_is_rejected(self):
        self.write_dataset()
        self.write_rank(
            0,
            {
                "1": self.metrics(1.0, 0.8, 1.0),
                "2": self.metrics(1.0, 0.6, 2.0),
            },
        )
        self.write_rank(1, {"3": self.metrics(0.0, 0.0, 5.0)})

        with self.assertRaisesRegex(EvaluationValidationError, "coverage mismatch"):
            self.validate()

    def test_aggregate_mean_mismatch_is_rejected(self):
        aggregate, _ = self.valid_artifacts()
        aggregate["success"] += 0.1
        self.write_aggregate(aggregate)

        with self.assertRaisesRegex(EvaluationValidationError, "mean mismatch"):
            self.validate()

    def test_missing_required_metric_is_rejected(self):
        self.write_dataset()
        first = self.metrics(1.0, 0.8, 1.0)
        second = self.metrics(0.0, 0.0, 5.0)
        first.pop("success")
        second.pop("success")
        self.write_rank(0, {"1": first, "3": second})
        self.write_rank(1, {"2": first, "4": second})

        with self.assertRaisesRegex(EvaluationValidationError, "required metrics"):
            self.validate()

    def test_rate_metric_outside_probability_range_is_rejected(self):
        self.write_dataset()
        invalid = self.metrics(1.01, 0.8, 1.0)
        self.write_rank(0, {"1": invalid, "3": self.metrics(0.0, 0.0, 5.0)})
        self.write_rank(
            1,
            {
                "2": self.metrics(1.0, 0.6, 2.0),
                "4": self.metrics(0.0, 0.0, 4.0),
            },
        )

        with self.assertRaisesRegex(EvaluationValidationError, r"in \[0, 1\]"):
            self.validate()

    def test_nonfinite_episode_metric_is_rejected(self):
        self.write_dataset()
        invalid = self.metrics(1.0, 0.8, 1.0)
        invalid["ndtw"] = float("nan")
        self.write_rank(0, {"1": invalid, "3": self.metrics(0.0, 0.0, 5.0)})
        self.write_rank(
            1,
            {
                "2": self.metrics(1.0, 0.6, 2.0),
                "4": self.metrics(0.0, 0.0, 4.0),
            },
        )

        with self.assertRaisesRegex(EvaluationValidationError, "must be finite"):
            self.validate()

    def test_nonfinite_aggregate_metric_is_rejected(self):
        aggregate, _ = self.valid_artifacts()
        aggregate["spl"] = float("inf")
        self.write_aggregate(aggregate)

        with self.assertRaisesRegex(EvaluationValidationError, "must be finite"):
            self.validate()

    def test_compare_metrics_reports_only_rate_percentage_points(self):
        comparison = compare_metrics(
            {
                "success": 0.55,
                "spl": 0.40,
                "ndtw": 0.60,
                "sdtw": 0.35,
                "distance_to_goal": 4.0,
            },
            {
                "success": 0.50,
                "spl": 0.42,
                "ndtw": 0.58,
                "sdtw": 0.30,
                "distance_to_goal": 5.0,
            },
        )

        self.assertEqual(set(comparison), {"success", "spl", "ndtw", "sdtw"})
        self.assertAlmostEqual(
            comparison["success"]["delta_percentage_points"], 5.0
        )
        self.assertAlmostEqual(
            comparison["spl"]["delta_percentage_points"], -2.0
        )
        self.assertNotIn("distance_to_goal", comparison)

    def test_legacy_collision_ratio_above_one_is_valid_not_clipped(self):
        aggregate, _ = self.valid_artifacts()
        values = [1.5, 0.0, 2.0, 0.5]
        offset = 0
        for rank in range(self.world_size):
            records = json.loads(self.rank_path(rank).read_text())
            for metrics in records.values():
                metrics['collisions'] = values[offset]
                offset += 1
            self.write_rank(rank, records)
        aggregate['collisions'] = sum(values) / len(values)
        self.write_aggregate(aggregate)
        report = self.validate()
        self.assertEqual(report['recomputed_metrics']['collisions'], 1.0)
        self.assertNotIn('collisions', compare_metrics(
            {'success': 0.5, 'collisions': 1.5},
            {'success': 0.5, 'collisions': 2.0}))

    def test_negative_legacy_collision_ratio_is_rejected(self):
        self.valid_artifacts()
        records = json.loads(self.rank_path(0).read_text())
        records['1']['collisions'] = -0.1
        self.write_rank(0, records)
        with self.assertRaisesRegex(EvaluationValidationError, 'nonnegative'):
            self.validate()


if __name__ == "__main__":
    unittest.main()
