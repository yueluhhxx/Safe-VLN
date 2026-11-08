from typing import Any, Dict, Optional, Tuple, List, Union
import math
import random
import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_extensions.utils import generate_video, heading_from_quaternion, navigator_video_frame, planner_video_frame
from vlnce_baselines.common.reselection import ReselectionController
from vlnce_baselines.common.teacher_cache import build_candidate_teacher_payload
from scipy.spatial.transform import Rotation as R
import cv2
import os


def quat_from_heading(heading, elevation=0):
    array_h = np.array([0, heading, 0])
    array_e = np.array([0, elevation, 0])
    rotvec_h = R.from_rotvec(array_h)
    rotvec_e = R.from_rotvec(array_e)
    quat = (rotvec_h * rotvec_e).as_quat()
    return quat

def calculate_vp_rel_pos(p1, p2, base_heading=0, base_elevation=0):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    xz_dist = max(np.sqrt(dx**2 + dz**2), 1e-8)
    # xyz_dist = max(np.sqrt(dx**2 + dy**2 + dz**2), 1e-8)

    heading = np.arcsin(-dx / xz_dist)  # (-pi/2, pi/2)
    if p2[2] > p1[2]:
        heading = np.pi - heading
    heading -= base_heading
    # to (0, 2pi)
    while heading < 0:
        heading += 2*np.pi
    heading = heading % (2*np.pi)

    return heading, xz_dist

@baseline_registry.register_env(name="VLNCEDaggerEnv")
class VLNCEDaggerEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)
        self.prev_episode_id = "something different"

        self.video_option = config.VIDEO_OPTION
        self.video_dir = config.VIDEO_DIR
        self.video_frames = []
        self.plan_frames = []
        self.controller_motion_mode = getattr(
            config.IL, 'controller_motion_mode', 'legacy_tolerance'
        )

    def get_reward_range(self) -> Tuple[float, float]:
        # We don't use a reward for DAgger, but the baseline_registry requires
        # we inherit from habitat.RLEnv.
        return (0.0, 0.0)

    def get_reward(self, observations: Observations) -> float:
        return 0.0

    def get_done(self, observations: Observations) -> bool:
        return self._env.episode_over

    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()

    def get_metrics(self):
        return self.habitat_env.get_metrics()

    def get_geodesic_dist(self, 
        node_a: List[float], node_b: List[float]):
        return self._env.sim.geodesic_distance(node_a, node_b)

    def check_navigability(self, node: List[float]):
        return self._env.sim.is_navigable(node)

    def get_agent_info(self):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
    
    def get_pos_ori(self):
        agent_state = self._env.sim.get_agent_state()
        pos = agent_state.position
        ori = np.array([*(agent_state.rotation.imag), agent_state.rotation.real])
        return (pos, ori)

    def get_observation_at(self,
        source_position: List[float],
        source_rotation: List[Union[int, np.float64]],
        keep_agent_at_new_pose: bool = False):
        
        obs = self._env.sim.get_observations_at(source_position, source_rotation, keep_agent_at_new_pose)
        obs.update(self._env.task.sensor_suite.get_observations(
            observations=obs, episode=self._env.current_episode, task=self._env.task
        ))
        return obs

    def current_dist_to_goal(self):
        init_state = self._env.sim.get_agent_state()
        init_distance = self._env.sim.geodesic_distance(
            init_state.position, self._env.current_episode.goals[0].position,
        )
        return init_distance
    
    def point_dist_to_goal(self, pos):
        dist = self._env.sim.geodesic_distance(
            pos, self._env.current_episode.goals[0].position,
        )
        return dist

    def point_dists_to_goal(self, positions):
        """Batch geodesic query used by deterministic merged-ghost teachers."""
        goal = self._env.current_episode.goals[0].position
        return [
            self._env.sim.geodesic_distance(pos, goal) for pos in positions
        ]
    
    def get_cand_real_pos(self, forward, angle):
        '''get cand real_pos by executing action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_pose = post_state.position

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_pose

    def get_cand_real_positions(
        self, angles, forwards, include_pose=False
    ):
        """Project candidates and their teacher distances in one worker call.

        The scalar method still restores the simulator after every candidate,
        so batching changes only IPC scheduling and not candidate or collision
        semantics.
        """

        return build_candidate_teacher_payload(
            self.get_cand_real_pos,
            self.point_dists_to_goal,
            self.current_dist_to_goal,
            angles,
            forwards,
            get_pose=self.get_pos_ori if include_pose else None,
        )

    def current_dist_to_refpath(self, path):
        sim = self._env.sim
        init_state = sim.get_agent_state()
        current_pos = init_state.position
        circle_dists = []
        for pos in path:
            circle_dists.append(
                self._env.sim.geodesic_distance(current_pos, pos)
            )
        # circle_dists = np.linalg.norm(np.array(path)-current_pos, axis=1).tolist()
        return circle_dists

    def ghost_dist_to_ref(self, ghost_vp_pos, ref_path):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        # ghost dis to subgoal
        ghost_dists_to_subgoal = []
        for ghost_vp, ghost_pos in ghost_vp_pos:
            dist = self._env.sim.geodesic_distance(ghost_pos, sub_goal_pos)
            ghost_dists_to_subgoal.append(dist)

        oracle_ghost_vp = ghost_vp_pos[np.argmin(ghost_dists_to_subgoal)][0]
        self.prev_episode_id = episode_id
            
        return oracle_ghost_vp

    def get_cand_idx(self, ref_path, angles, distances, candidate_length):
        episode_id = self._env.current_episode.episode_id
        if episode_id != self.prev_episode_id:
            self.progress = 0
            self.prev_sub_goal_pos = [0.0, 0.0, 0.0]
        progress = self.progress
        # ref_path = self.envs.current_episodes()[j].reference_path
        circle_dists = self.current_dist_to_refpath(ref_path)
        circle_bool = np.array(circle_dists) <= 3.0
        cand_dists_to_goal = []
        if circle_bool.sum() == 0: # no gt point within 3.0m
            sub_goal_pos = self.prev_sub_goal_pos
        else:
            cand_idxes = np.where(circle_bool * (np.arange(0,len(ref_path))>=progress))[0]
            if len(cand_idxes) == 0:
                sub_goal_pos = ref_path[progress] #prev_sub_goal_pos[perm_index]
            else:
                compare = np.array(list(range(cand_idxes[0],cand_idxes[0]+len(cand_idxes)))) == cand_idxes
                if np.all(compare):
                    sub_goal_idx = cand_idxes[-1]
                else:
                    sub_goal_idx = np.where(compare==False)[0][0]-1
                sub_goal_pos = ref_path[sub_goal_idx]
                self.progress = sub_goal_idx
            
            self.prev_sub_goal_pos = sub_goal_pos

        for k in range(len(angles)):
            angle_k = angles[k]
            forward_k = distances[k]
            dist_k = self.cand_dist_to_subgoal(angle_k, forward_k, sub_goal_pos)
            # distance to subgoal
            cand_dists_to_goal.append(dist_k)

        # distance to final goal
        curr_dist_to_goal = self.current_dist_to_goal()
        # if within target range (which def as 3.0)
        if curr_dist_to_goal < 1.5:
            oracle_cand_idx = candidate_length - 1
        else:
            oracle_cand_idx = np.argmin(cand_dists_to_goal)

        self.prev_episode_id = episode_id
        # if curr_dist_to_goal == np.inf:
            
        return oracle_cand_idx #, sub_goal_pos

    def cand_dist_to_goal(self, angle: float, forward: float):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
        post_state = sim.get_agent_state()
        post_distance = self._env.sim.geodesic_distance(
            post_state.position, self._env.current_episode.goals[0].position,
        )

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def cand_dist_to_subgoal(self, 
        angle: float, forward: float,
        sub_goal: Any):
        r'''get resulting distance to goal by executing 
        a candidate action'''

        sim = self._env.sim
        init_state = sim.get_agent_state()

        forward_action = HabitatSimActions.MOVE_FORWARD
        init_forward = sim.get_agent(0).agent_config.action_space[
            forward_action].actuation.amount

        theta = np.arctan2(init_state.rotation.imag[1], 
            init_state.rotation.real) + angle / 2
        rotation = np.quaternion(np.cos(theta), 0, np.sin(theta), 0)
        sim.set_agent_state(init_state.position, rotation)

        ksteps = int(forward//init_forward)
        prev_pos = init_state.position
        dis = 0.
        for k in range(ksteps):
            sim.step_without_obs(forward_action)
            pos = sim.get_agent_state().position
            dis += np.linalg.norm(prev_pos - pos)
            prev_pos = pos
        post_state = sim.get_agent_state()

        post_distance = self._env.sim.geodesic_distance(
            post_state.position, sub_goal,
        ) + dis

        # reset agent state
        sim.set_agent_state(init_state.position, init_state.rotation)
        
        return post_distance
    
    def reset(self):
        observations = self._env.reset()
        if self.video_option:
            info = self.get_info(observations)
            self.video_frames = [
                navigator_video_frame(
                    observations, 
                    info,
                )
            ]
        return observations

    # def wrap_act(self, act, ang, dis, cand_wp, action_wp, oracle_wp, start_p, start_h):
    def wrap_act(self, act, vis_info):
        ''' wrap action, get obs if video_option '''
        observations = None
        if self.video_option:
            observations = self._env.step(act)
            info = self.get_info(observations)
            self.video_frames.append(
                navigator_video_frame(
                    observations,
                    info,
                    vis_info,
                )
            )
        else:
            self._env.sim.step_without_obs(act)
            self._env._task.measurements.update_measures(
                episode=self._env.current_episode, action=act, task=self._env.task 
            )
        return observations

    def turn(self, ang, vis_info):    
        ''' angle: 0 ~ 360 degree '''
        act_l = HabitatSimActions.TURN_LEFT
        act_r = HabitatSimActions.TURN_RIGHT
        uni_l = self._env.sim.get_agent(0).agent_config.action_space[act_l].actuation.amount
        ang_degree = math.degrees(ang)
        ang_degree = round(ang_degree / uni_l) * uni_l
        observations = None

        if 180 < ang_degree <= 360:
            ang_degree -= 360
        if ang_degree >=0:
            turns = [act_l] * ( ang_degree // uni_l)
        else:
            turns = [act_r] * (-ang_degree // uni_l)

        for turn in turns:
            observations = self.wrap_act(turn, vis_info)
        return observations

    def teleport(self, pos):
        self._env.sim.set_agent_state(pos, quat_from_heading(0))

    def single_step_control(
        self,
        pos,
        tryout,
        vis_info,
        reach_tolerance=0.5,
        horizontal_only=True,
    ):
        """Move towards one target and classify the low-level outcome.

        Habitat reports a collision whenever the navigation filter shortens a
        requested move.  With sliding enabled that can still leave substantial
        tangential motion, so the collision bit alone is not evidence that the
        agent is trapped.  Motion is therefore measured in the horizontal
        plane.  Candidate ghost targets are planar projections, so their
        arrival metric is XZ by default.  Callers following a graph path or
        returning to a visited pose pass ``horizontal_only=False`` so a node
        on another floor cannot be accepted merely because its XZ matches.

        ``trapped`` is deliberately narrow: all six Tryout probes were unable
        to move, or a movable probe was followed by a stationary blocked tail.
        Merely ending short of the target after making progress is reported as
        ``progressed`` and must not trigger re-selection or blacklisting.

        ``legacy_tolerance`` preserves the V2 controller for paired evaluation.
        ``etpnav_distance`` keeps the same arrival classification but does not
        use its 0.5 m tolerance to shorten movement: direct motion uses ETPNav's
        floor(distance / forward_step) budget, stopping early only at the
        numerical target.  This setting must be pinned for train/eval/infer.
        """
        motion_mode = getattr(
            self, 'controller_motion_mode', 'legacy_tolerance'
        )
        if motion_mode not in ('legacy_tolerance', 'etpnav_distance'):
            raise ValueError('Unknown controller_motion_mode: %s' % motion_mode)
        motion_stop_distance = (
            float(reach_tolerance)
            if motion_mode == 'legacy_tolerance'
            else 1.0e-6
        )
        act_f = HabitatSimActions.MOVE_FORWARD
        uni_f = self._env.sim.get_agent(0).agent_config.action_space[act_f].actuation.amount
        agent_state = self._env.sim.get_agent_state()
        start_pos = np.array(agent_state.position, dtype=np.float32)
        target_pos = np.array(pos, dtype=np.float32)

        # Forward actions are 0.25 m in the requested setup.  These tolerances
        # reject only simulator jitter while retaining genuine sliding motion.
        motion_epsilon = 1.0e-3
        progress_epsilon = 1.0e-2

        def horizontal_distance(first, second):
            first = np.asarray(first, dtype=np.float32)
            second = np.asarray(second, dtype=np.float32)
            return float(np.linalg.norm(first[[0, 2]] - second[[0, 2]]))

        def arrival_distance(first, second):
            if horizontal_only:
                return horizontal_distance(first, second)
            first = np.asarray(first, dtype=np.float32)
            second = np.asarray(second, dtype=np.float32)
            return float(np.linalg.norm(first - second))

        start_remaining = arrival_distance(start_pos, target_pos)
        collision_count = 0
        travelled_distance = 0.0
        tryout_started = False
        tryout_probe_count = 0
        trap_reason = None

        def forward_once():
            nonlocal collision_count, travelled_distance
            before = np.array(
                self._env.sim.get_agent_state().position, dtype=np.float32
            )
            before_remaining = arrival_distance(before, target_pos)
            self.wrap_act(act_f, vis_info)
            after = np.array(
                self._env.sim.get_agent_state().position, dtype=np.float32
            )
            moved = horizontal_distance(before, after)
            after_remaining = arrival_distance(after, target_pos)
            collided = bool(self._env.sim.previous_step_collided)
            collision_count += int(collided)
            travelled_distance += moved
            return {
                'position': after,
                'motion': moved,
                'progress': before_remaining - after_remaining,
                'remaining': after_remaining,
                'collided': collided,
            }

        def finish(outcome, reason=None):
            end_pos = np.array(
                self._env.sim.get_agent_state().position, dtype=np.float32
            )
            remaining_distance_xz = horizontal_distance(end_pos, target_pos)
            remaining_distance_3d = float(np.linalg.norm(end_pos - target_pos))
            remaining_distance = (
                remaining_distance_xz
                if horizontal_only
                else remaining_distance_3d
            )
            reached = remaining_distance <= float(reach_tolerance)
            if reached:
                outcome = 'reached'
                reason = None
            net_progress = start_remaining - remaining_distance
            return {
                'outcome': outcome,
                'reached': reached,
                'trapped': outcome == 'trapped',
                'target_attempted': True,
                # Keep the legacy key as the arrival metric chosen by the
                # caller, while retaining both distances for diagnostics.
                'remaining_distance': remaining_distance,
                'remaining_distance_xz': remaining_distance_xz,
                'remaining_distance_3d': remaining_distance_3d,
                'vertical_error': float(abs(end_pos[1] - target_pos[1])),
                'travelled_distance': travelled_distance,
                'net_progress': net_progress,
                'collision_count': collision_count,
                'tryout_started': tryout_started,
                'tryout_probe_count': tryout_probe_count,
                'tryout_exhausted': outcome == 'trapped',
                'trap_reason': reason,
            }

        if start_remaining <= motion_stop_distance:
            return finish('reached')

        ang, dis = calculate_vp_rel_pos(
            agent_state.position, pos,
            heading_from_quaternion(agent_state.rotation),
        )
        self.turn(ang, vis_info)
        ksteps = int(dis // uni_f)

        completed_direct_steps = 0
        direct_blocked = False
        for _ in range(ksteps):
            step_result = forward_once()
            if step_result['remaining'] <= motion_stop_distance:
                return finish('reached')
            if step_result['motion'] <= motion_epsilon:
                direct_blocked = True
                break
            # A collision with real motion is sliding, not a blocked step.
            completed_direct_steps += 1

        current_pos = np.array(
            self._env.sim.get_agent_state().position, dtype=np.float32
        )
        current_remaining = arrival_distance(current_pos, target_pos)
        net_progress = start_remaining - current_remaining

        if not direct_blocked:
            if net_progress > progress_epsilon:
                return finish('progressed')
            if motion_mode == 'etpnav_distance' and ksteps == 0:
                # ETPNav does not probe a target closer than one forward step.
                # A tolerance used for graph arrival must not create extra
                # physical movement here (including a same-XZ upper floor).
                return finish('inconclusive')
            if not tryout:
                return finish('inconclusive')
        elif not tryout:
            if net_progress > progress_epsilon:
                return finish('progressed')
            return finish('inconclusive')

        # Direct control made no meaningful progress or met a stationary
        # obstruction.  Try the original six ETPNav probe headings.  A probe
        # that slides is allowed to continue; only zero-motion probes fail.
        tryout_started = True
        remaining_steps = max(0, ksteps - completed_direct_steps)
        try_ang = random.choice([math.radians(90), math.radians(270)])
        self.turn(try_ang, vis_info)
        if try_ang == math.radians(90):     # from left to right
            turn_seqs = [
                (0, 270),   # 90, turn_left=30, turn_right=330
                (330, 300), # 60
                (330, 330), # 30
                (300, 30),  # -30
                (330, 60),  # -60
                (330, 90),  # -90
            ]
        else:                               # from right to left
            turn_seqs = [
                (0, 90),   # -90
                (30, 60),  # -60
                (30, 30),  # -30
                (60, 330), # 30
                (30, 300), # 60
                (30, 270), # 90
            ]

        for turn_seq in turn_seqs:
            self.turn(math.radians(turn_seq[0]), vis_info)
            tryout_probe_count += 1
            probe_result = forward_once()
            if probe_result['remaining'] <= motion_stop_distance:
                return finish('reached')
            if probe_result['motion'] <= motion_epsilon:
                continue

            # A genuinely moving probe commits to this detour, matching the
            # original controller.  A later stationary tail is nevertheless a
            # confirmed Tryout failure rather than a successful detour.
            self.turn(math.radians(turn_seq[1]), vis_info)
            for _ in range(remaining_steps):
                tail_result = forward_once()
                if tail_result['remaining'] <= motion_stop_distance:
                    return finish('reached')
                if tail_result['motion'] <= motion_epsilon:
                    trap_reason = 'tail_blocked'
                    return finish('trapped', trap_reason)

            end_pos = np.array(
                self._env.sim.get_agent_state().position, dtype=np.float32
            )
            end_progress = start_remaining - arrival_distance(end_pos, target_pos)
            if end_progress > progress_epsilon:
                return finish('progressed')
            return finish('inconclusive')

        trap_reason = 'probes_exhausted'
        return finish('trapped', trap_reason)
    
    def multi_step_control(
        self, path, tryout, vis_info, reach_tolerance=0.5
    ):
        result = {
            'outcome': 'reached',
            'reached': True,
            'trapped': False,
            # The caller can safely anchor the next graph observation at the
            # last path node that was actually reached.  This matters when a
            # collision leaves the physical agent between two graph nodes.
            'last_reached_vp': None,
            'remaining_distance': 0.0,
            'remaining_distance_xz': 0.0,
            'remaining_distance_3d': 0.0,
            'travelled_distance': 0.0,
            'collision_count': 0,
            'tryout_exhausted': False,
            'trap_reason': None,
        }
        for vp, vp_pos in path: #path[::-1]:
            step_result = self.single_step_control(
                vp_pos,
                tryout,
                vis_info,
                reach_tolerance,
                horizontal_only=False,
            )
            result['outcome'] = step_result['outcome']
            result['reached'] = step_result['reached']
            result['trapped'] = step_result['trapped']
            result['remaining_distance'] = step_result['remaining_distance']
            result['remaining_distance_xz'] = step_result[
                'remaining_distance_xz'
            ]
            result['remaining_distance_3d'] = step_result[
                'remaining_distance_3d'
            ]
            result['travelled_distance'] += step_result['travelled_distance']
            result['collision_count'] += step_result['collision_count']
            result['tryout_exhausted'] = (
                result['tryout_exhausted']
                or step_result['tryout_exhausted']
            )
            result['trap_reason'] = step_result.get('trap_reason')
            if step_result['reached']:
                result['last_reached_vp'] = str(vp)
            # Released ETPNav attempts every node in its planned path even if
            # an earlier local controller did not arrive.  Keep that bounded
            # best-effort execution; a route failure must not silently suppress
            # the subsequent ghost's own Tryout.  Only confirmed arrivals may
            # update the graph anchor.
        return result

    def get_plan_frame(self, vis_info):
        agent_state = self._env.sim.get_agent_state()
        observations = self.get_observation_at(agent_state.position, agent_state.rotation)
        info = self.get_info(observations)

        frame = planner_video_frame(observations, info, vis_info)
        frame = cv2.copyMakeBorder(frame, 6,6,5,5, cv2.BORDER_CONSTANT, value=(255,255,255))
        self.plan_frames.append(frame)

    def step(self, action, vis_info, *args, **kwargs):
        act = action['act']
        reselection_result = None

        if act == 4: # high to low
            if self.video_option:
                self.get_plan_frame(vis_info)

            # New actions carry planner-ranked attempts.  Falling back to the
            # legacy one-attempt payload preserves compatibility with older
            # checkpoints and callers.
            attempts = action.get('attempts')
            if attempts is None:
                attempts = [action]
            reach_tolerance = float(action.get('reach_tolerance', 0.5))
            route_mode = action.get('reselection_route_mode', 'topological')
            if route_mode not in ('topological', 'direct'):
                raise ValueError('unknown re-selection route mode: ' + str(route_mode))
            controller = ReselectionController(
                [attempt['ghost_vp'] for attempt in attempts]
            )
            attempts_by_id = {
                attempt['ghost_vp']: attempt for attempt in attempts
            }
            attempt_details = []
            initial_state = self._env.sim.get_agent_state()
            initial_position = np.array(initial_state.position)
            # This is the last graph node whose position the physical agent is
            # known to have reached.  It is intentionally updated only after a
            # confirmed controller arrival (or an exact teleport).
            confirmed_anchor_vp = str(action['cur_vp'])

            while not controller.done:
                ghost_vp = controller.next_ghost
                attempt = attempts_by_id[ghost_vp]

                is_retry = bool(controller.attempted)
                back_result = {
                    'outcome': 'not_required',
                    'reached': False,
                    'trapped': False,
                    'last_reached_vp': None,
                }
                execution_path = None
                if not (is_retry and route_mode == 'direct'):
                    if attempt['back_path'] is None:
                        self.teleport(attempt['front_pos'])
                        confirmed_anchor_vp = str(attempt['front_vp'])
                        back_result['outcome'] = 'reached'
                        back_result['reached'] = True
                        back_result['last_reached_vp'] = confirmed_anchor_vp
                    else:
                        execution_path = attempt['back_path']
                        if is_retry:
                            routes = attempt.get('routes_from', {})
                            if confirmed_anchor_vp in routes:
                                execution_path = routes[confirmed_anchor_vp]
                            elif routes:
                                raise ValueError(
                                    'missing retry route from confirmed anchor '
                                    + confirmed_anchor_vp
                                )
                            else:
                                # Compatibility with old action payloads.  Even
                                # when this best-effort path cannot be followed,
                                # still execute the secondary target controller.
                                execution_path = [
                                    (str(action['cur_vp']), initial_position)
                                ] + list(execution_path)
                        back_result = self.multi_step_control(
                            execution_path,
                            attempt['tryout'],
                            vis_info,
                            reach_tolerance,
                        )
                        if back_result['last_reached_vp'] is not None:
                            confirmed_anchor_vp = str(
                                back_result['last_reached_vp']
                            )
                # As in ETPNav, always attempt the requested target after the
                # finite route-control sequence.  The same routine recomputes
                # heading from the live pose and enables Tryout for BOTH the
                # primary and secondary.  Only its trapped result permits a
                # secondary; partial movement alone does not.
                execution_anchor_vp = confirmed_anchor_vp
                move_result = self.single_step_control(
                    attempt['ghost_pos'],
                    attempt['tryout'],
                    vis_info,
                    reach_tolerance,
                    horizontal_only=True,
                )
                controller.record_target_outcome(ghost_vp, move_result['outcome'])

                reached = bool(move_result['reached'])
                attempt_details.append({
                    'ghost_vp': ghost_vp,
                    'front_vp': attempt['front_vp'],
                    'execution_anchor_vp': execution_anchor_vp,
                    'execution_path': execution_path,
                    'reached': reached,
                    'outcome': move_result['outcome'],
                    'target_attempted': True,
                    'return_result': None,
                    'back_result': back_result,
                    'move_result': move_result,
                })

                agent_state = self._env.sim.get_agent_state()
                observations = self.get_observation_at(
                    agent_state.position, agent_state.rotation
                )

            reselection_result = dict(controller.summary())
            reselection_result['details'] = attempt_details
            reselection_result['route_mode'] = route_mode
            # Both attempts exhausted: there is no third retry, reset, turn or
            # STOP here.  Return the final real observation to high-level
            # planning, exactly the released controller's failure behavior.
            reselection_result['fallback'] = (
                'etpnav_next_observation'
                if controller.outcome == 'trapped' else None
            )
            if controller.succeeded is not None:
                successful_attempt = next(
                    detail for detail in attempt_details
                    if detail['ghost_vp'] == controller.succeeded
                )
                reselection_result['selected_front_vp'] = (
                    successful_attempt['execution_anchor_vp']
                )
            else:
                reselection_result['selected_front_vp'] = None
            reselection_result['failure_anchor_vp'] = confirmed_anchor_vp

        elif act == 0:   # stop
            if self.video_option:
                self.get_plan_frame(vis_info)

            # 1. back to stop node
            if action['back_path'] is None:
                self.teleport(action['stop_pos'])
            else:
                self.multi_step_control(action['back_path'], action['tryout'], vis_info)

            # 2. stop
            observations = self._env.step(act)
            if self.video_option:
                info = self.get_info(observations)
                self.video_frames.append(
                    navigator_video_frame(
                        observations,
                        info,
                        vis_info,
                    )
                )
                self.get_plan_frame(vis_info)

        else:
            raise NotImplementedError                

        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = dict(self.get_info(observations))
        if reselection_result is not None:
            # Explicit high-to-low execution status consumed by the planner.
            info['reselection'] = reselection_result

        if self.video_option and done:
            # if 0 < info["spl"] <= 0.6:  #TODO backtrack
            generate_video(
                video_option=self.video_option,
                video_dir=self.video_dir,
                images=self.video_frames,
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2],
                checkpoint_idx=0,
                metrics={"SPL": round(info["spl"], 3)},
                tb_writer=None,
                fps=8,
            )
            # for pano visualization
            metrics={
                        # "sr": round(info["success"], 3), 
                        "spl": round(info["spl"], 3),
                        # "ndtw": round(info["ndtw"], 3),
                        # "sdtw": round(info["sdtw"], 3),
                    }
            metric_strs = []
            for k, v in metrics.items():
                metric_strs.append(f"{k}{v:.2f}")
            episode_id=self._env.current_episode.episode_id
            scene_id=self._env.current_episode.scene_id.split('/')[-1].split('.')[-2]
            tmp_name = f"{scene_id}-{episode_id}-" + "-".join(metric_strs)
            tmp_name = tmp_name.replace(" ", "_").replace("\n", "_") + ".png"
            tmp_fn = os.path.join(self.video_dir, tmp_name)
            tmp = np.concatenate(self.plan_frames, axis=0)
            cv2.imwrite(tmp_fn, tmp)
            self.plan_frames = []

        return observations, reward, done, info

@baseline_registry.register_env(name="VLNCEInferenceEnv")
class VLNCEInferenceEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        super().__init__(config.TASK_CONFIG, dataset)

    def get_reward_range(self):
        return (0.0, 0.0)

    def get_reward(self, observations: Observations):
        return 0.0

    def get_done(self, observations: Observations):
        return self._env.episode_over

    def get_info(self, observations: Observations):
        agent_state = self._env.sim.get_agent_state()
        heading_vector = quaternion_rotate_vector(
            agent_state.rotation.inverse(), np.array([0, 0, -1])
        )
        heading = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return {
            "position": agent_state.position.tolist(),
            "heading": heading,
            "stop": self._env.task.is_stop_called,
        }
