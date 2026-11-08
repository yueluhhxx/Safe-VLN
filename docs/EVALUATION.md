# Evaluation configuration

The published results use R2R-CE **val-unseen**, 1839 episodes, seed 0,
eight distributed ranks and two environments per rank. The checkpoint table
contains all 41 iterations from 12000 to 20000, at an interval of 200.
Checkpoint selection uses SR first and SPL as the tie-breaker.

The saved settings are:

```yaml
MODEL.SAFE_VLN.DELTA: 1.0e-5
MODEL.SAFE_VLN.SENSOR_HEIGHT: 0.24
TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING: true
IL.tryout: true
IL.reselection: true
IL.reselection_max_retries: 1
IL.back_algo: control
IL.controller_motion_mode: etpnav_distance
IL.ghost_route_mode: nearest_front
IL.max_traj_len: 15
EVAL.fast_eval: false
EVAL.EPISODE_COUNT: -1
EVAL.USE_CKPT_CONFIG: false
VIDEO_OPTION: []
```

Evaluation uses physical low-level control, not teleportation. Training uses
the original ETPNav teleport-based rollout convention. At evaluation time,
actions are chosen from model scores; reference paths and goal distances are
used for metrics, not for choosing STOP or the next waypoint.

The controller uses simulator pose and heading. Four low-mounted depth strips
provide the planar occupancy observations. These sensing assumptions are part
of the configuration.

As in the released ETPNav evaluator, SR is computed from final geodesic distance
`<= 3 m` after episode termination. There is no automatic goal-distance STOP.
The primary rollout budget is 15 high-level decisions; low-level motion inside
one decision is not a strict 500-action protocol. Changing the stopping or
action-budget convention defines a different evaluation configuration.

The chosen checkpoint and controller were selected on val-unseen. These
validation results are not an independent test-set score. The launcher checks
all episode IDs and recomputes metrics from the eight per-rank JSON files.
It does not drop failed episodes or report oracle success as SR.
