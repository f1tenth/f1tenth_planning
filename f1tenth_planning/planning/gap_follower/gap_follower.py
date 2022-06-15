import numpy as np
from numba import njit
from queue import PriorityQueue

from numpy import vectorize


@njit(fastmath=False, cache=True)
def pre_process_LiDAR(ranges, window_size, danger_thres, rb):
    ## ranges: (1080, ) array

    # roughly filter with mean
    proc_ranges = []
    window_size = 10
    for i in range(0, len(ranges), window_size):
        cur_mean = sum(ranges[i:i+window_size])/window_size
        # if cur_mean >= self.safe_thres:
        #     cur_mean = self.safe_thres
        for _ in range(window_size):
            proc_ranges.append(cur_mean)
    proc_ranges = np.array(proc_ranges)

    # set danger range and ranges too far to zero
    p, n = 0, len(proc_ranges)
    while p < n:
        if proc_ranges[p] <= danger_thres:
            ranges[max(0, p - rb): p+rb] = 0
            p += rb
        else:
            p += 1
    return proc_ranges

# @njit(fastmath=False, cache=True)
def find_target_point(ranges, safe_thres, max_gap_length=350, min_gap_length=50):
    """_summary_
        Find all the gaps exceed a safe thres.
        Among those qualified gaps, chose the one with a farmost point, calculate the target as the middle of the gap.
    Args:
        ranges (_type_): _description_
        safe_thres (_type_): _description_
        max_gap_length (int, optional): _description_. Defaults to 350.
    Returns:
        target: int
    """
    n = len(ranges)
    safe_p_left, safe_p_right = 0, n-1
    p = safe_p_left
    safe_range = PriorityQueue()
    while p < n-1:
        if ranges[p] >= safe_thres:
            safe_p_left = p
            p += 1
            # while p < end_i and ranges[p] >= self.safe_thres and p-safe_p_left <= 290:
            while p < n-1 and ranges[p] >= safe_thres and p-safe_p_left <= max_gap_length:
                p += 1
            safe_p_right = p-1
            if safe_p_right != safe_p_left:
                safe_range.put((-(np.max(ranges[safe_p_left:safe_p_right])), (safe_p_left, safe_p_right)))
        else:
            p += 1
    if safe_range.empty():
        print('no safe range')
        return np.argmax(ranges)
    else:
        while not safe_range.empty():
            safe_p_left, safe_p_right = safe_range.get()[1]
            if safe_p_right-safe_p_left > min_gap_length:
                target = (safe_p_left+safe_p_right)//2
                if 190 < target < 900:
                    return target


class Gap_follower:
    def __init__(self) -> None:
        # 6, 350, 20
        self.windowSize = 10  # window size for filtering the LiDAR scan
        # TODO: modeify this
        self.angle_incre = 0.00435
        self.angle_min = -2.35
        self.max_speed = 6

        self.safe_thres = 3.0
        self.danger_thres = 1.2
        self.rb = 10
        self.P = 0.6


    def plan(self, scan):
        ## scan: Lidar scan of current car
        #Find closest point to LiDAR
        closest_p_idx = np.argmin(scan)
        closest_angle = self.angle_min + closest_p_idx * self.angle_incre
        filted_scan = pre_process_LiDAR(scan, self.windowSize, self.danger_thres, self.rb)  # (1080, )
        best_p_idx = find_target_point(filted_scan, self.safe_thres)
        # print(best_p_idx)

        steering = (self.angle_min + best_p_idx * self.angle_incre) * self.P
        if abs(steering) >= 1:
            velocity = self.max_speed / 2
        else:
            velocity = self.max_speed
        # return np.array([steering, velocity]), best_p_idx
        return steering, velocity