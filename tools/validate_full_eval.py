#!/usr/bin/env python3
"""Validate complete distributed R2R-CE evaluation results.

This module intentionally uses only the Python standard library.  It can be
run after all evaluator ranks have exited without importing Habitat, PyTorch,
or any GPU-facing package.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set


DEFAULT_EXPECTED_EPISODE_COUNT = 1839
DEFAULT_CHECKPOINT_INDEX = 7
DEFAULT_WORLD_SIZE = 8

# Habitat's aggregate is commonly computed in float32 while this validator
# recomputes with math.fsum (double precision).  These tolerances accept that
# representation difference but still reject meaningful metric mismatches.
AGGREGATE_REL_TOLERANCE = 1.0e-6
AGGREGATE_ABS_TOLERANCE = 1.0e-6

# These are dimensionless rates/scores stored as probabilities.  Distances,
# path lengths, step counts, and ghost counts must never be reported as
# percentage-point changes.
RATE_METRICS = frozenset(
    {
        "success",
        "oracle_success",
        "spl",
        "ndtw",
        "sdtw",
    }
)
REQUIRED_METRICS = frozenset({"success", "spl", "ndtw", "sdtw"})


class EvaluationValidationError(ValueError):
    """Raised when evaluation artifacts are incomplete or inconsistent."""


def _reject_duplicate_json_keys(pairs: Iterable[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationValidationError(
                "duplicate JSON key {!r}; duplicate episode keys must not be "
                "silently overwritten".format(key)
            )
        result[key] = value
    return result


def _read_json(path: Path) -> Any:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else path.open
    try:
        if path.suffix == ".gz":
            stream = opener(path, "rt", encoding="utf-8")
        else:
            stream = opener("r", encoding="utf-8")
        with stream:
            return json.load(stream, object_pairs_hook=_reject_duplicate_json_keys)
    except EvaluationValidationError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationValidationError(
            "cannot read valid JSON from {}: {}".format(path, error)
        ) from error


def _episode_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise EvaluationValidationError(
            "{} has unsupported episode_id {!r}; expected an integer or string".format(
                context, value
            )
        )
    normalized = str(value)
    if not normalized:
        raise EvaluationValidationError("{} has an empty episode_id".format(context))
    return normalized


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationValidationError(
            "{} must be a finite number, got {!r}".format(context, value)
        )
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationValidationError(
            "{} must be finite, got {!r}".format(context, value)
        )
    return number


def _dataset_episode_ids(dataset: Path, expected_episode_count: int) -> Set[str]:
    document = _read_json(Path(dataset))
    if not isinstance(document, dict) or not isinstance(document.get("episodes"), list):
        raise EvaluationValidationError(
            "dataset {} must contain a top-level episodes list".format(dataset)
        )

    episode_ids: Set[str] = set()
    for index, episode in enumerate(document["episodes"]):
        if not isinstance(episode, dict) or "episode_id" not in episode:
            raise EvaluationValidationError(
                "dataset episode {} must be an object with episode_id".format(index)
            )
        episode_id = _episode_id(
            episode["episode_id"], context="dataset episode {}".format(index)
        )
        if episode_id in episode_ids:
            raise EvaluationValidationError(
                "dataset contains duplicate episode_id {!r}".format(episode_id)
            )
        episode_ids.add(episode_id)

    if len(episode_ids) != expected_episode_count:
        raise EvaluationValidationError(
            "dataset has {} unique episodes; expected {}".format(
                len(episode_ids), expected_episode_count
            )
        )
    return episode_ids


def _preview(values: Iterable[str], limit: int = 10) -> str:
    ordered = sorted(values)
    suffix = "" if len(ordered) <= limit else ", ..."
    return "[{}{}]".format(", ".join(repr(item) for item in ordered[:limit]), suffix)


def _coerce_metric_mapping(raw: Any, *, context: str) -> Dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise EvaluationValidationError(
            "{} must be a non-empty metric object".format(context)
        )
    metrics: Dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or not name:
            raise EvaluationValidationError(
                "{} contains an invalid metric name {!r}".format(context, name)
            )
        number = _finite_number(
            value, context="{} metric {!r}".format(context, name)
        )
        if name in RATE_METRICS and not 0.0 <= number <= 1.0:
            raise EvaluationValidationError(
                "{} rate metric {!r} must be in [0, 1], got {!r}".format(
                    context, name, number
                )
            )
        # ETPNav divides collision count by the number of distinct consecutive
        # path positions, NOT by attempted forward actions.  Stationary blocked
        # moves increase the numerator without adding a position, so this
        # legacy ratio is finite/nonnegative but legitimately unbounded above.
        # Do not clamp it or silently drop an otherwise valid episode.
        if name == "collisions" and number < 0.0:
            raise EvaluationValidationError(
                "{} collisions ratio must be nonnegative".format(context)
            )
        metrics[name] = number
    return metrics


def validate_full_evaluation(
    results_dir: Path,
    dataset: Path,
    checkpoint_index: int = DEFAULT_CHECKPOINT_INDEX,
    world_size: int = DEFAULT_WORLD_SIZE,
    expected_episode_count: int = DEFAULT_EXPECTED_EPISODE_COUNT,
) -> Dict[str, Any]:
    """Validate rank coverage and aggregate means for a complete evaluation.

    The expected files are ``stats_ep_ckpt_{checkpoint}_val_unseen_r{rank}_w{N}.json``
    for every rank and ``stats_ckpt_{checkpoint}_val_unseen.json`` for the
    evaluator's aggregate.  Any duplicate, missing, or unexpected episode is
    rejected before aggregate values are accepted.
    """

    results_dir = Path(results_dir)
    dataset = Path(dataset)
    if (
        isinstance(checkpoint_index, bool)
        or not isinstance(checkpoint_index, int)
        or checkpoint_index < 0
    ):
        raise EvaluationValidationError("checkpoint_index must be non-negative")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise EvaluationValidationError("world_size must be positive")
    if (
        isinstance(expected_episode_count, bool)
        or not isinstance(expected_episode_count, int)
        or expected_episode_count <= 0
    ):
        raise EvaluationValidationError("expected_episode_count must be positive")

    dataset_ids = _dataset_episode_ids(dataset, expected_episode_count)
    observed_sources: Dict[str, int] = {}
    rank_episode_counts: Dict[str, int] = {}
    metric_values: Dict[str, list[float]] = {}
    expected_metric_names: Optional[Set[str]] = None

    for rank in range(world_size):
        rank_path = results_dir / (
            "stats_ep_ckpt_{}_val_unseen_r{}_w{}.json".format(
                checkpoint_index, rank, world_size
            )
        )
        rank_document = _read_json(rank_path)
        if not isinstance(rank_document, dict):
            raise EvaluationValidationError(
                "rank {} result {} must be an episode object".format(rank, rank_path)
            )
        rank_episode_counts[str(rank)] = len(rank_document)

        for raw_episode_id, raw_metrics in rank_document.items():
            episode_id = _episode_id(
                raw_episode_id,
                context="rank {} result {}".format(rank, rank_path),
            )
            if episode_id in observed_sources:
                raise EvaluationValidationError(
                    "duplicate episode_id {!r} appears in ranks {} and {}".format(
                        episode_id, observed_sources[episode_id], rank
                    )
                )
            observed_sources[episode_id] = rank

            metrics = _coerce_metric_mapping(
                raw_metrics,
                context="rank {} episode {}".format(rank, episode_id),
            )
            metric_names = set(metrics)
            if expected_metric_names is None:
                expected_metric_names = metric_names
                missing_required = REQUIRED_METRICS - expected_metric_names
                if missing_required:
                    raise EvaluationValidationError(
                        "rank results are missing required metrics: {}".format(
                            sorted(missing_required)
                        )
                    )
                metric_values = {name: [] for name in sorted(metric_names)}
            elif metric_names != expected_metric_names:
                missing_metrics = expected_metric_names - metric_names
                extra_metrics = metric_names - expected_metric_names
                raise EvaluationValidationError(
                    (
                        "rank {} episode {} metric schema mismatch: "
                        "missing={} extra={}"
                    ).format(
                        rank,
                        episode_id,
                        sorted(missing_metrics),
                        sorted(extra_metrics),
                    )
                )
            for name, value in metrics.items():
                metric_values[name].append(value)

    observed_ids = set(observed_sources)
    missing_ids = dataset_ids - observed_ids
    unexpected_ids = observed_ids - dataset_ids
    if missing_ids or unexpected_ids:
        raise EvaluationValidationError(
            "episode coverage mismatch: missing={} unexpected={}".format(
                _preview(missing_ids), _preview(unexpected_ids)
            )
        )
    if len(observed_ids) != expected_episode_count:
        # The set comparison above normally implies this.  Keep the explicit
        # guard so the contract remains clear if ID normalization changes.
        raise EvaluationValidationError(
            "results contain {} unique episodes; expected {}".format(
                len(observed_ids), expected_episode_count
            )
        )
    if expected_metric_names is None:
        raise EvaluationValidationError("rank results contain no episode metrics")

    recomputed_metrics = {
        name: math.fsum(metric_values[name]) / expected_episode_count
        for name in sorted(metric_values)
    }
    aggregate_path = (
        results_dir / "stats_ckpt_{}_val_unseen.json".format(checkpoint_index)
    )
    aggregate_metrics = _coerce_metric_mapping(
        _read_json(aggregate_path), context="aggregate {}".format(aggregate_path)
    )
    aggregate_names = set(aggregate_metrics)
    if aggregate_names != expected_metric_names:
        raise EvaluationValidationError(
            "aggregate metric schema mismatch: missing={} extra={}".format(
                sorted(expected_metric_names - aggregate_names),
                sorted(aggregate_names - expected_metric_names),
            )
        )

    mismatches = []
    for name in sorted(expected_metric_names):
        aggregate_value = aggregate_metrics[name]
        recomputed_value = recomputed_metrics[name]
        if not math.isclose(
            aggregate_value,
            recomputed_value,
            rel_tol=AGGREGATE_REL_TOLERANCE,
            abs_tol=AGGREGATE_ABS_TOLERANCE,
        ):
            mismatches.append(
                "{}: aggregate={!r}, recomputed={!r}".format(
                    name, aggregate_value, recomputed_value
                )
            )
    if mismatches:
        raise EvaluationValidationError(
            "aggregate mean mismatch: {}".format("; ".join(mismatches))
        )

    return {
        "metrics": {
            name: aggregate_metrics[name] for name in sorted(aggregate_metrics)
        },
        "recomputed_metrics": recomputed_metrics,
        "episode_count": len(observed_ids),
        "coverage_validated": True,
        "rank_episode_counts": rank_episode_counts,
        "aggregate_path": str(aggregate_path.resolve()),
    }


def compare_metrics(
    new_metrics: Mapping[str, Any], old_metrics: Mapping[str, Any]
) -> Dict[str, Dict[str, float]]:
    """Return percentage-point deltas for shared probability metrics only."""

    if not isinstance(new_metrics, Mapping) or not isinstance(old_metrics, Mapping):
        raise EvaluationValidationError("new_metrics and old_metrics must be mappings")

    comparison: Dict[str, Dict[str, float]] = {}
    for name in sorted(RATE_METRICS & set(new_metrics) & set(old_metrics)):
        new_value = _finite_number(
            new_metrics[name], context="new metric {!r}".format(name)
        )
        old_value = _finite_number(
            old_metrics[name], context="old metric {!r}".format(name)
        )
        if not (0.0 <= new_value <= 1.0 and 0.0 <= old_value <= 1.0):
            raise EvaluationValidationError(
                "rate metric {!r} must use probability units in [0, 1]".format(name)
            )
        comparison[name] = {
            "new": new_value,
            "old": old_value,
            "delta_percentage_points": (new_value - old_value) * 100.0,
        }
    if not comparison:
        raise EvaluationValidationError(
            "no shared rate metrics are available to compare"
        )
    return comparison


def _metrics_from_evaluation_complete(path: Path) -> Mapping[str, Any]:
    document = _read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("metrics"), dict):
        raise EvaluationValidationError(
            "{} must contain a top-level metrics object".format(path)
        )
    return document["metrics"]


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--checkpoint-index", type=int, default=DEFAULT_CHECKPOINT_INDEX
    )
    parser.add_argument("--world-size", type=int, default=DEFAULT_WORLD_SIZE)
    parser.add_argument(
        "--expected-episode-count",
        type=int,
        default=DEFAULT_EXPECTED_EPISODE_COUNT,
    )
    parser.add_argument(
        "--old-evaluation-complete",
        type=Path,
        help="optional previous evaluation_complete.json for percentage-point deltas",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(argv)
    try:
        report = validate_full_evaluation(
            arguments.results_dir,
            arguments.dataset,
            checkpoint_index=arguments.checkpoint_index,
            world_size=arguments.world_size,
            expected_episode_count=arguments.expected_episode_count,
        )
        if arguments.old_evaluation_complete is not None:
            old_metrics = _metrics_from_evaluation_complete(
                arguments.old_evaluation_complete
            )
            report["comparison"] = compare_metrics(report["metrics"], old_metrics)
            report["old_evaluation_complete"] = str(
                arguments.old_evaluation_complete.resolve()
            )
    except EvaluationValidationError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
