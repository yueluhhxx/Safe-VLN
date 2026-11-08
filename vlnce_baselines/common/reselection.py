"""Pure re-selection helpers used by the Safe-VLN controller.

This module deliberately has no Habitat, NumPy, or Torch dependency.  Keeping
the small state machine independent makes the collision/retry behaviour easy
to test without starting a simulator.
"""

import math
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple


IGNORE_INDEX = -100


def stable_teacher_top2(
    candidate_costs: Iterable[Tuple[int, str, float]],
    stop: bool = False,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[int, int]:
    """Return deterministic primary and secondary teacher action indices.

    ``candidate_costs`` contains ``(action_index, ghost_id, total_cost)``.
    Ties are resolved by the action index and then the ghost id, so dict order,
    process rank, and simulator call timing cannot change the labels.
    """

    if stop:
        return 0, ignore_index

    ranked = []
    for action_index, ghost_id, cost in candidate_costs:
        cost = float(cost)
        if math.isnan(cost):
            continue
        ranked.append((cost, int(action_index), str(ghost_id)))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))

    if not ranked:
        return ignore_index, ignore_index
    primary = ranked[0][1]
    secondary = ranked[1][1] if len(ranked) > 1 else ignore_index
    return primary, secondary


def rank_reselection_candidates(
    selected_action: int,
    gmap_vp_ids: Sequence[Optional[str]],
    planner_logits: Sequence[float],
    blacklisted_ghosts: Iterable[str] = (),
    max_retries: int = 1,
) -> List[int]:
    """Rank the selected ghost followed by bounded planner alternatives.

    The first action is kept even when it did not come from argmax (for
    example, under teacher forcing).  Retry candidates are unvisited ghost
    nodes ordered by descending planner logit with a stable action-index tie
    break.  Stop, visited graph nodes, non-finite logits, and blacklisted
    ghosts are never used as retry targets.
    """

    max_retries = max(0, int(max_retries))
    selected_action = int(selected_action)
    if not 0 <= selected_action < len(gmap_vp_ids):
        return []

    selected_vp = gmap_vp_ids[selected_action]
    if selected_vp is None or not str(selected_vp).startswith("g"):
        return []

    blocked: Set[str] = set(blacklisted_ghosts)
    if selected_vp in blocked:
        return []

    alternatives = []
    for action_index, (vp_id, logit) in enumerate(
        zip(gmap_vp_ids, planner_logits)
    ):
        if action_index == selected_action or vp_id is None:
            continue
        vp_id = str(vp_id)
        score = float(logit)
        if (
            not vp_id.startswith("g")
            or vp_id in blocked
            or not math.isfinite(score)
        ):
            continue
        alternatives.append((-score, action_index, vp_id))
    alternatives.sort(key=lambda item: (item[0], item[1], item[2]))

    return [selected_action] + [
        item[1] for item in alternatives[:max_retries]
    ]


def resolve_graph_anchor(
    current_vp: str,
    selected_front_vp: Optional[str],
    succeeded: bool,
    failure_anchor_vp: Optional[str] = None,
) -> str:
    """Resolve the node from which the next observed position is connected.

    A successful high-to-low action uses its confirmed front node, matching
    ETPNav.  If every attempt fails, connect from the last graph node that the
    controller actually reached.  The decision node is a compatibility
    fallback for older environments that do not report this anchor.
    """

    if succeeded:
        if selected_front_vp is None:
            raise ValueError("a successful attempt requires a front node")
        return str(selected_front_vp)
    if failure_anchor_vp is not None:
        return str(failure_anchor_vp)
    return str(current_vp)


def commit_etpnav_attempts(graph, current_vp, result, consume_ghost=True):
    """Consume executed ghosts, including failed ones, as in released ETPNav.

    A failed low-level action returns a real observation to the next planning
    step; it is not a terminal STOP.  Unlike the old persistent blacklist, a
    consumed ghost may be observed again from a new pose.  Never consume an
    unattempted alternative, and never invent an edge from an unreached front.
    Validate all graph references before mutating the graph.
    """
    anchor = resolve_graph_anchor(
        current_vp, result.get("selected_front_vp"),
        bool(result.get("success")), result.get("failure_anchor_vp"),
    )
    if anchor not in graph.node_pos:
        raise ValueError("controller reported an unknown graph anchor: " + anchor)
    attempted = list(dict.fromkeys(result.get("attempted_ghosts", ())))
    if consume_ghost:
        for ghost in attempted:
            if ghost in graph.ghost_pos:
                graph.delete_ghost(ghost)
    return anchor


def reselection_diagnostics(result):
    """Count actual target Tryouts separately from route-control failures."""
    details = result.get("details", ())
    targets = [item for item in details if item.get("target_attempted")]
    primary = targets[:1]
    secondary = targets[1:]
    counters = {}
    for label, items in (("primary", primary), ("secondary", secondary)):
        counters["control_" + label + "_attempts"] = len(items)
        counters["control_" + label + "_tryouts"] = sum(
            bool(item.get("move_result", {}).get("tryout_started"))
            for item in items
        )
        counters["control_" + label + "_trapped"] = sum(
            item.get("outcome") == "trapped" for item in items
        )
    counters["control_double_trap_fallbacks"] = int(
        len(targets) == 2
        and all(item.get("outcome") == "trapped" for item in targets)
        and result.get("fallback") == "etpnav_next_observation"
    )
    counters["control_skipped_secondary_after_trap"] = int(
        len(targets) == 1 and targets[0].get("outcome") == "trapped"
        and bool(result.get("unattempted_ghosts"))
    )
    counters["control_route_nonarrivals"] = sum(
        not item.get("back_result", {}).get("reached", False)
        and item.get("back_result", {}).get("outcome") != "not_required"
        for item in details
    )
    return counters


class ReselectionController:
    """Retry only after Tryout confirms that the attempted ghost is trapped.

    Not reaching a target is insufficient evidence: a sliding movement may
    still be making progress, or the route to the target's front may fail
    before the target itself is attempted.
    """

    def __init__(self, candidates: Sequence[str]) -> None:
        # De-duplicate while preserving the planner order.
        self.candidates = tuple(dict.fromkeys(str(vp) for vp in candidates))
        self._cursor = 0
        self.attempted = []  # type: List[str]
        self.failed = []  # type: List[str]
        self.succeeded = None  # type: Optional[str]
        self.outcome = None  # type: Optional[str]
        self.abort_reason = None  # type: Optional[str]
        self._halted = False

    @property
    def next_ghost(self) -> Optional[str]:
        if (
            self._halted
            or self.succeeded is not None
            or self._cursor >= len(self.candidates)
        ):
            return None
        return self.candidates[self._cursor]

    @property
    def done(self) -> bool:
        return self.next_ghost is None

    def record_target_outcome(self, ghost_id: str, outcome: str) -> None:
        """Record an executed target segment, never an unexecuted route.

        Only ``trapped`` advances to the planner's next-ranked alternative.
        ``progressed`` and ``inconclusive`` end this action.  This state
        machine does not mutate the graph; commit_etpnav_attempts applies
        the configured consume-ghost policy to all executed targets.
        """
        expected = self.next_ghost
        if expected is None:
            raise RuntimeError("re-selection controller is already finished")
        if str(ghost_id) != expected:
            raise ValueError(
                "attempt result is out of order: "
                f"expected {expected!r}, got {ghost_id!r}"
            )
        if outcome not in ("reached", "trapped", "progressed", "inconclusive"):
            raise ValueError("unknown target outcome: {!r}".format(outcome))

        self.attempted.append(expected)
        self._cursor += 1
        self.outcome = outcome
        if outcome == "reached":
            self.succeeded = expected
        elif outcome == "trapped":
            self.failed.append(expected)
        else:
            self._halted = True

    def abort(self, reason: str = "route_failed") -> None:
        """Stop before a target attempt without blaming its ghost node."""
        if self.done:
            raise RuntimeError("re-selection controller is already finished")
        self._halted = True
        self.outcome = "route_failed"
        self.abort_reason = str(reason)

    def summary(self) -> Mapping[str, object]:
        return {
            "success": self.succeeded is not None,
            "selected_ghost": self.succeeded,
            "attempted_ghosts": list(self.attempted),
            "failed_ghosts": list(self.failed),
            "attempts": len(self.attempted),
            # Keep failed_ghosts as a compatibility alias, but its meaning is
            # now strictly Tryout-confirmed trapped targets.
            "trapped_ghosts": list(self.failed),
            "unattempted_ghosts": list(self.candidates[self._cursor:]),
            "outcome": self.outcome,
            "abort_reason": self.abort_reason,
        }
