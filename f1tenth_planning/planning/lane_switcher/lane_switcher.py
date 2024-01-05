import yaml
from argparse import Namespace


import numpy as np
import scipy.spatial

from f1tenth_planning.control.pure_pursuit.pure_pursuit import PurePursuitPlanner


class LaneSwitcher:
    def __init__(self, conf, wb=0.33):
        self.wheelbase = wb
        self.v_scale = conf.traj_v_scale
        ### load waypoints
        self.map_path = conf.map_path
        self.map_ext = conf.map_ext
        waypoints = np.loadtxt(
            conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
        )
        self.s_max = waypoints[-1, 0]
        centerLine = np.vstack(
            (
                waypoints[:, 1],
                waypoints[:, 2],
                waypoints[:, 5],
                waypoints[:, 3],
                waypoints[:, 0],
            )
        ).T
        self.centerLine = centerLine

        self.ittc_thres = conf.ittc_thres

        self.num_lane_pts = []
        self.lane_alldata = []
        self.lane_xytheta = []
        self.lane_xyv = []
        self.lane_pos = []
        self.num_lanes = conf.lanesNum
        self.lanesFiles = conf.lanesFiles

        for i in range(self.num_lanes):
            waypoints = np.loadtxt(
                self.lanesFiles[i], delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip
            )
            # [len, 5 (x, y, v, heading, s)]
            laneData = np.vstack(
                (
                    waypoints[:, 1],
                    waypoints[:, 2],
                    waypoints[:, 5] * self.v_scale,
                    waypoints[:, 3],
                    waypoints[:, 0],
                )
            ).T
            self.num_lane_pts.append(len(laneData))
            self.lane_xyv.append(laneData[:, :3])
            self.lane_xytheta.append(
                np.vstack((laneData[:, 0], laneData[:, 1], laneData[:, 3])).T
            )
            self.lane_pos.append(np.vstack((laneData[:, 0], laneData[:, 1])).T)
            self.lane_alldata.append(laneData)
        self.s_max = self.lane_alldata[0][-1][-1]

        # pure pursuit
        pure_pursuit_conf_file = conf.tracker_config_path
        with open(pure_pursuit_conf_file) as file:
            pp_conf = yaml.load(file, Loader=yaml.FullLoader)
        pp_conf = Namespace(**pp_conf)
        pp_conf.wpt_path = conf.wpt_path
        pp_conf.map_path = conf.map_path
        if conf.tracker == "advanced_pure_pursuit":
            raise NotImplementedError
        elif conf.tracker == "pure_pursuit":
            self.tracker = PurePursuitPlanner(pp_conf)
        self.step = 0

        # Car Status Variables
        self.lane_idx = 0
        self.last_lane = 0
        self.cur_lane = 0
        self.curr_idx = None
        self.goal_idx = None
        self.curr_vel = 0.0
        self.target_point = None
        self.last_ego_s = 0
        self.last_opp_s = 0
        self.state_i = 0
        self.state_t = 0

        self.opponent = np.array([np.inf, np.inf])
        self.opponent_last = np.array([0.0, 0.0])
        self.lane_free = [True] * self.num_lanes
        self.lane_dist = [0.0] * self.num_lanes
        self.follow_v_scale = conf.follow_v_scale
        self.avoid_buffer = conf.avoid_buffer
        self.slowdown_buffer = conf.slowdown_buffer
        self.slowdown_v_scale = conf.slowdown_v_scale
        self.slowdown_counter = 0
        self.avoid_counter = 0
        self.detect_oppo = False
        self.avoid_dist = conf.avoid_dist
        self.lane_occupied_dist = conf.lane_occupied_dist

    def plan(self, pose_x, pose_y, pose_theta, opp_poses, velocity, waypoints=None):
        car_follow_flag = False
        # update lane_free

        # use lane 0 as the center line
        ego_pose = np.array([pose_x, pose_y, pose_theta])
        _, _, ego_t, ego_i = nearest_point(ego_pose[:2], self.lane_pos[0])
        _, _, self.state_t, self.state_i = nearest_point(
            ego_pose[:2], self.centerLine[:, 0:2]
        )
        _, _, opp_t, opp_i = nearest_point(opp_poses[0][:2], self.lane_pos[0])
        ego_s = self.lane_alldata[0][ego_i][-1]
        opp_s = self.lane_alldata[0][opp_i][-1]
        # if self.last_opp_s == 0 and self.last_ego_s == 0:
        #     self.last_opp_s = opp_s
        #     self.last_ego_s = ego_s
        if opp_s < ego_s and (self.s_max - ego_s + opp_s) > self.avoid_dist:
            self.lane_free = [True] * self.num_lanes
        else:
            opp_ego_dist = np.linalg.norm(
                np.vstack((pose_x, pose_y)).flatten() - opp_poses[0][:2]
            )
            # print(f'opp_ego_dist is {opp_ego_dist}')
            if opp_ego_dist < self.avoid_dist:
                for i in range(self.num_lanes):
                    # import ipdb; ipdb.set_trace()
                    d = scipy.spatial.distance.cdist(self.lane_pos[i], opp_poses[:, :2])
                    self.lane_free[i] = np.min(d) > self.lane_occupied_dist
                    self.lane_dist[i] = np.min(d)
            else:
                self.lane_free = [True] * self.num_lanes

        # check if all lanes are occupied
        if not np.any(self.lane_free):
            # print("all lanes are occupied")
            car_follow_flag = True

        # switch back to main lane if possible
        if self.lane_free[0]:
            self.avoid_counter = min(self.avoid_buffer, self.avoid_counter + 1)
        else:
            self.avoid_counter = 0
        if self.avoid_counter == self.avoid_buffer and self.lane_free[0]:
            self.last_lane = 0

        # switch if last_lane is occupied
        if not self.lane_free[self.last_lane]:
            self.cur_lane = np.argmax(self.lane_dist)
        else:
            self.cur_lane = self.last_lane

        # slow down when switch
        if self.cur_lane == self.last_lane:
            self.slowdown_counter = min(self.slowdown_buffer, self.slowdown_counter + 1)
        else:
            self.slowdown_counter = 0

        self.last_lane = self.cur_lane
        cur_lane_xyv = self.lane_xyv[self.cur_lane].copy()
        if self.slowdown_counter < self.slowdown_buffer:
            cur_lane_xyv[:, 2] *= self.slowdown_v_scale
        elif car_follow_flag:
            cur_lane_xyv[:, 2] *= self.follow_v_scale

        return cur_lane_xyv

    def cal_s(self):
        t, i = self.state_t, self.state_i
        s = self.centerLine[i, 4]
        if s < self.last_ego_s:
            s = s + self.s_max
        self.last_ego_s = s
        return s

    def cal_objectives(self, rollout_states: dict, lap_time, overtake, crash):
        objectives = [0.0] * 2
        rollout_obs = rollout_states["rollout_obs"]
        rollout_ego_s = rollout_states["rollout_ego_s"]
        rollout_opp_s = rollout_states["rollout_opp_s"]
        ego_control_error = np.sum(rollout_states["ego_control_error"])
        ego_ittc = np.array(rollout_states["ego_ittc"])
        abs_ittc = np.array(rollout_states["abs_ittc"])

        # progress
        ego_start_s, ego_end_s = rollout_ego_s[0], rollout_ego_s[-1]
        opp_start_s, opp_end_s = rollout_opp_s[0], rollout_opp_s[-1]
        ego_progress = ego_end_s - ego_start_s
        opp_progress = opp_end_s - opp_start_s
        relative_progress = opp_progress - ego_progress
        # objectives[0] = relative_progress * 5 / lap_time
        objectives[0] = opp_end_s - ego_end_s

        # safety
        # objectives[1] = 5 * (10 * ego_control_error / n - np.sum(ego_ittc) * 0.2 / n)
        # objectives[1] = len(np.nonzero(ego_ittc)) / n + len(np.nonzero(ego_ittc < self.ittc_thres)) / n

        n = len(ego_ittc)
        if n == 0:
            return [0, 0]
        # import ipdb;
        # ipdb.set_trace()

        # print(ego_control_error / n, np.sum(abs_ittc) / n)
        objectives[1] += len(np.nonzero(ego_ittc)[0]) / n
        ego_ittc = ego_ittc[ego_ittc > 0]
        objectives[1] += len(np.nonzero(ego_ittc < self.ittc_thres)[0]) / n
        ego_ittc = ego_ittc[ego_ittc < self.ittc_thres]
        objectives[1] *= self.ittc_thres
        if len(ego_ittc != 0):
            objectives[1] += self.ittc_thres - np.mean(ego_ittc)

        # objectives[1] += 10 * (ego_control_error / n - np.sum(abs_ittc) / n)
        objectives[1] += 10 * (ego_control_error / n)
        #  print(np.sum(ego_ittc))

        if overtake:
            objectives[0] *= 1.1
        if crash:
            objectives[1] += 1

        return objectives
