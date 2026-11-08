"""Simulator-independent training arithmetic and explicit teacher choices.

Virtual reference groups preserve ETPNav's mean of two rank-local means for
fixed samples while distributing their 16 environments over more GPUs. They
do not claim identical dropout, episode ordering, or floating-point kernels.
"""

import operator
import random

import torch
import torch.distributed as distributed


def scheduled_sampling_probability(base_ratio, decay_interval, iteration):
    """ETPNav's probability for a zero-based optimizer-iteration index."""

    iteration = operator.index(iteration)
    decay_interval = operator.index(decay_interval)
    if iteration < 0 or decay_interval <= 0:
        raise ValueError("iteration must be nonnegative and decay_interval positive")
    if not 0.0 <= base_ratio <= 1.0:
        raise ValueError("base sampling probability must be in [0, 1]")
    return base_ratio ** (iteration // decay_interval + 1)


def iteration_sampling_probability(
    base_ratio, decay_interval, iteration, schedule, interval_ratio
):
    """Keep the old interval schedule available as an explicit ablation."""

    if schedule == "per_iteration":
        return scheduled_sampling_probability(base_ratio, decay_interval, iteration)
    if schedule == "legacy_interval":
        return interval_ratio
    raise ValueError("unknown sampling schedule: {}".format(schedule))


def choose_teacher_positions(positions, mode, choice=None):
    """Select once per ghost; the exact cache must not replace this choice.

    ``random_representative`` uses the released ETPNav random-choice rule.
    ``minimum`` retains V2's best-of-all-observations label for comparison.
    """

    if len(positions) == 0:
        raise ValueError("a teacher ghost must have at least one position")
    if mode == "minimum":
        return list(positions)
    if mode == "random_representative":
        return [(random.choice if choice is None else choice)(positions)]
    raise ValueError("unknown teacher position mode: {}".format(mode))


def reference_group_layout(world_size, rank, reference_groups):
    """Return this rank's reference group and its number of member ranks."""

    world_size = operator.index(world_size)
    rank = operator.index(rank)
    reference_groups = operator.index(reference_groups)
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed world size or rank")
    if reference_groups < 0:
        raise ValueError("reference_groups must be nonnegative")
    if reference_groups == 0:
        return rank, 1
    if reference_groups > world_size or world_size % reference_groups:
        raise ValueError("reference_groups must evenly divide the world size")
    ranks_per_group = world_size // reference_groups
    return rank // ranks_per_group, ranks_per_group


def normalize_rollout_loss(
    loss_sum, total_actions, world_size, rank, reference_groups=0, all_reduce=None
):
    """Normalize before AMP backward, without differentiating action counts.

    For W real ranks and G reference groups, each rank contributes
    ``(W/G) * local_loss_sum / group_action_count``. DDP's division by W then
    gives the mean of G action-normalized group losses, exactly the reference
    objective for fixed samples. G=0 retains the legacy rank-local objective.

    Only one small integer collective is needed, after every rollout has
    finished; there are no extra per-navigation-step collectives. The injected
    reducer supports CPU tests and has the usual in-place all_reduce contract.
    """

    total_actions = operator.index(total_actions)
    if total_actions <= 0:
        raise ValueError("training rollout must contain at least one action")
    group, ranks_per_group = reference_group_layout(world_size, rank, reference_groups)
    if reference_groups == 0 or ranks_per_group == 1:
        return loss_sum / total_actions

    if all_reduce is None:
        if not distributed.is_initialized():
            raise RuntimeError("reference-group normalization requires initialized DDP")
        if distributed.get_world_size() != world_size or distributed.get_rank() != rank:
            raise RuntimeError("reference-group layout disagrees with DDP")
        all_reduce = distributed.all_reduce
    counts = torch.zeros(reference_groups, dtype=torch.int64, device=loss_sum.device)
    counts[group] = total_actions
    all_reduce(counts)
    # Autocast cross_entropy already returns float32. Keep low-precision caller
    # inputs safe too, and preserve float64 for strict CPU equivalence checks.
    numerator = loss_sum.float() if loss_sum.dtype in (torch.float16, torch.bfloat16) else loss_sum
    return numerator * ranks_per_group / counts[group].to(dtype=numerator.dtype)
