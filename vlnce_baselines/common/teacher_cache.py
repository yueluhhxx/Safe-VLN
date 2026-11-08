"""Small, simulator-independent helpers for batched ETPNav teacher queries.

The R2R goal and navigation mesh are fixed for the lifetime of one episode,
so a geodesic distance for an exact 3-D position can be reused throughout a
rollout.  Keeping the cache logic here makes the performance optimization
testable without Habitat or a GPU.
"""


def exact_position_tuple(position):
    """Return an exact, hashable key for one 3-D simulator position."""

    try:
        key = tuple(float(component) for component in position)
    except TypeError as error:
        raise ValueError("position must be an iterable with three values") from error
    if len(key) != 3:
        raise ValueError("position must contain exactly three values")
    return key


def exact_pose_tuple(pose):
    """Return an exact, hashable key for a simulator position/rotation pair."""

    try:
        position, orientation = pose
    except (TypeError, ValueError) as error:
        raise ValueError("pose must contain position and orientation") from error
    position_key = exact_position_tuple(position)
    try:
        orientation_key = tuple(float(component) for component in orientation)
    except TypeError as error:
        raise ValueError("orientation must be an iterable with four values") from error
    if len(orientation_key) != 4:
        raise ValueError("orientation must contain exactly four values")
    return position_key, orientation_key


def should_use_candidate_payload_pose(task_type, mode, video_enabled):
    """Whether the R2R rollout can reuse the candidate worker reply for pose."""

    return task_type == "r2r" and (mode == "train" or bool(video_enabled))


def batch_candidate_real_positions(get_position, angles, forwards):
    """Evaluate scalar candidate projection in order through one RPC call.

    ``get_position`` is deliberately the existing scalar implementation.  The
    helper only removes parent/worker round trips; it does not change any
    simulator action, collision, or state-restoration behavior.
    """

    if len(angles) != len(forwards):
        raise ValueError("candidate angles and forwards must have equal length")
    return [
        get_position(angle=angle, forward=forward)
        for angle, forward in zip(angles, forwards)
    ]


def build_candidate_teacher_payload(
    get_position,
    get_goal_distances,
    get_current_goal_distance,
    angles,
    forwards,
    get_pose=None,
):
    """Build one environment reply for candidate projection and teaching.

    Candidate simulation, candidate-to-goal geodesics, and the current
    position-to-goal geodesic all execute inside one Habitat worker command.
    """

    initial_pose = get_pose() if get_pose is not None else None
    initial_pose_key = (
        exact_pose_tuple(initial_pose) if initial_pose is not None else None
    )
    current_goal_distance = get_current_goal_distance()
    positions = batch_candidate_real_positions(
        get_position, angles, forwards
    )
    goal_distances = get_goal_distances(positions)
    if len(goal_distances) != len(positions):
        raise ValueError(
            "candidate positions and goal distances must have equal length"
        )
    payload = {
        "positions": positions,
        "goal_distances": goal_distances,
        "current_goal_distance": current_goal_distance,
    }
    if get_pose is not None:
        # Candidate projection temporarily moves the Habitat agent.  The pose
        # may replace the trainer's former standalone RPC only after every
        # scalar projection has restored exactly the state seen on entry.
        restored_pose = get_pose()
        if exact_pose_tuple(restored_pose) != initial_pose_key:
            raise RuntimeError(
                "candidate projection did not exactly restore the agent pose"
            )
        payload["pose"] = restored_pose
    return payload


class EpisodeGeodesicCache:
    """Cache point-to-goal geodesics for one active environment episode."""

    def __init__(self):
        self._costs = {}

    def __len__(self):
        return len(self._costs)

    def missing_positions(self, positions):
        """Return uncached positions once each, preserving first-seen order."""

        missing = []
        pending = set()
        for position in positions:
            key = exact_position_tuple(position)
            if key not in self._costs and key not in pending:
                missing.append(position)
                pending.add(key)
        return missing

    def update(self, positions, costs):
        """Store an ordered result batch, rejecting a truncated reply."""

        if len(positions) != len(costs):
            raise ValueError("geodesic positions and costs must have equal length")
        for position, cost in zip(positions, costs):
            self._costs[exact_position_tuple(position)] = float(cost)

    def cost(self, position):
        return self._costs[exact_position_tuple(position)]

    def minimum(self, positions):
        return min(self.cost(position) for position in positions)


def prepare_missing_position_batches(caches, positions_by_environment):
    """Split uncached positions into one ordered RPC payload per environment."""

    if len(caches) != len(positions_by_environment):
        raise ValueError("cache and environment batches must have equal length")
    return [
        cache.missing_positions(positions)
        for cache, positions in zip(caches, positions_by_environment)
    ]


def update_geodesic_cache_batches(
    caches, positions_by_environment, costs_by_environment
):
    """Apply parallel RPC replies to their corresponding episode caches."""

    if not (
        len(caches)
        == len(positions_by_environment)
        == len(costs_by_environment)
    ):
        raise ValueError("cache, position, and cost batches must have equal length")
    for cache, positions, costs in zip(
        caches, positions_by_environment, costs_by_environment
    ):
        cache.update(positions, costs)
