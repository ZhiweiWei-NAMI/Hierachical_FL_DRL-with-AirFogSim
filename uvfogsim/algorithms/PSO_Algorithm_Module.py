# -*- encoding: utf-8 -*-
'''
@File    :   Weighted_Voronoi_Algorithm_Module.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''

from collections import deque
from math import floor
import numpy as np
from .Base_Algorithm_Module import Base_Algorithm_Module
import torch
import torch.nn as nn
import os
from pyswarms.single.global_best import GlobalBestPSO
import copy

class PSO_Algorithm_Module(Base_Algorithm_Module):
    def __init__(self, env):
        super().__init__()
        self.max_weight = 2.5 # 权重不可以是负数，0代表只作为中继的意愿，0~1代表计算的意愿
        self.max_weight_bs = 2.5
        self.n_agent = env.n_UAV
        self.uav_weights = np.zeros(env.n_UAV)
        self.bs_weights = np.zeros(env.n_BS)
        self.n_UAV = env.n_UAV
        self.n_actions = [4,1,5] # 每一个动作的空间,通过pso算法探索,获取get_assigned
        
    def get_assigned(self, env, v2u_distance, v2i_distance, uav_weights = None, bs_weights = None):
        if uav_weights is None: uav_weights = self.uav_weights
        if bs_weights is None: bs_weights = self.bs_weights
        # 根据无人机，车辆，基站的位置，计算weighted voronoi，返回v2u_assigned和v2i_assigned。
        v2u_assigned = np.zeros((env.n_Veh, env.n_UAV))
        v2i_assigned = np.zeros((env.n_Veh, env.n_BS))
        if v2u_distance.shape[0] != env.n_Veh:
            return v2u_assigned, v2i_assigned
        # weighted voronoi的公式
        for idx, veh in enumerate(env.vehicle_by_index):
            min_dist = None
            min_j = 0
            uav_or_bs = 0
            for j, uav in enumerate(env.UAVs):
                dist = env.v2u_distance[idx,j] - uav_weights[j]
                # dist = env.V2UChannel_with_fastfading[idx,j, 0] - uav_weights[j]
                if (min_dist is None or (min_dist > dist  and env.v2u_distance[idx, j] <= 300)):
                    min_dist = dist
                    min_j = j
                    uav_or_bs = 1
            for k, bs in enumerate(env.BSs):
                dist = env.v2i_distance[idx,k] - bs_weights[k]
                # dist = env.V2IChannel_with_fastfading[idx, k, 0] - bs_weights[k]
                if (min_dist is None or (min_dist > dist  and env.v2i_distance[idx, k] <= 500)):
                    min_dist = dist
                    min_j = k
                    uav_or_bs = 2
            if uav_or_bs == 1:
                v2u_assigned[idx, min_j] = 1
            elif uav_or_bs == 2:
                v2i_assigned[idx, min_j] = 1
        return v2u_assigned, v2i_assigned
            

    def baseline_assigned_reward(self, env):
        
        def f_per_particle(m):
            v2u_assigned = np.zeros((env.n_Veh, env.n_UAV))
            v2i_assigned = np.zeros((env.n_Veh, env.n_BS))
            for vidx, device_idx in enumerate(m):
                device_idx = int(device_idx)
                if device_idx >= env.n_UAV:
                    v2i_assigned[vidx, device_idx - env.n_UAV] = 1
                elif device_idx < env.n_UAV:
                    v2u_assigned[vidx, device_idx] = 1
            return -np.sum(env.get_rewards(v2u_assigned, v2i_assigned))
        def f(x):
            n_particles = x.shape[0]
            j = [f_per_particle(x[i]) for i in range(n_particles)]
            return np.array(j)
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        dimensions = env.n_Veh # 直接以index
        lower_bound = np.zeros(dimensions)
        upper_bound = np.ones(dimensions) * (env.n_UAV+env.n_BS)
        bounds = np.array([lower_bound, upper_bound])
        optimizer = GlobalBestPSO(n_particles=10, dimensions=dimensions, bounds = bounds, options=options)
        # Perform optimization
        cost, m = optimizer.optimize(f, iters=50, verbose=False)
        v2u_assigned = np.zeros((env.n_Veh, env.n_UAV))
        v2i_assigned = np.zeros((env.n_Veh, env.n_BS))
        for vidx, device_idx in enumerate(m):
            device_idx = int(device_idx)
            if device_idx >= env.n_UAV:
                v2i_assigned[vidx, device_idx - env.n_UAV] = 1
            elif device_idx < env.n_UAV:
                v2u_assigned[vidx, device_idx] = 1
        return -cost, v2u_assigned, v2i_assigned


    def act_mobility(self, env, vehicle_ids, vehicle_positions2):
        def f_per_particle(m):
            uav_actions, bs_actions = m[:2*(env.n_UAV)], m[2*(env.n_UAV):]
            uav_actions = np.array(uav_actions).reshape(env.n_UAV, -1)
            bs_actions = np.array(bs_actions).reshape(env.n_BS, -1)
            uav_directions, uav_weights = uav_actions[:,0], uav_actions[:,1]
            uav_directions = np.array(uav_directions * (self.n_actions[0]), dtype=np.int16) / (self.n_actions[0])
            uav_weights = np.array(uav_weights * (self.n_actions[2]), dtype=np.int16) / self.n_actions[2]
            bs_weights = np.array(bs_actions * (self.n_actions[2]), dtype=np.int16) / self.n_actions[2]
            # uav_speeds = uav_speeds * env.uav_max_speed
            uav_speeds = env.uav_max_speed * np.ones_like(uav_weights)
            uav_directions = uav_directions * 2 * np.pi
            uav_weights = (uav_weights * self.max_weight).reshape(-1)
            bs_weights = (bs_weights * self.max_weight_bs).reshape(-1)
            uav_positions = np.array(env.get_uav_positions(), dtype=np.float64)
            old_uav_pos = copy.deepcopy(uav_positions)
            uav_positions[:, 0] = uav_positions[:, 0] + np.cos(uav_directions) * uav_speeds * env.time_step
            uav_positions[:, 1] = uav_positions[:, 1] + np.cos(uav_directions) * uav_speeds * env.time_step
            for i, uav in enumerate(env.UAVs):
                uav.position = uav_positions[i]
            states, _ = env.get_states(uav_weights, bs_weights, 40, 40)
            rewards = env.get_discrete_rewards(states[0,:,:,:])
            for i, uav in enumerate(env.UAVs):
                uav.position = old_uav_pos[i]
            return -np.sum(rewards)

        def f(x):
            n_particles = x.shape[0]
            j = [f_per_particle(x[i]) for i in range(n_particles)]
            return np.array(j)

        # Initialize swarm
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
        dimensions = 2*(env.n_UAV) + env.n_BS
        lower_bound = np.zeros(dimensions)
        upper_bound = np.ones(dimensions)

        # Combine the lower and upper bounds into a single array
        bounds = np.array([lower_bound, upper_bound])
        optimizer = GlobalBestPSO(n_particles=15, dimensions=dimensions, bounds = bounds, options=options)
        
        # Perform optimization
        cost, m = optimizer.optimize(f, iters=50, verbose=False)
        
        uav_actions, bs_actions = m[:2*(env.n_UAV)], m[2*(env.n_UAV):]
        uav_actions = np.array(uav_actions).reshape(env.n_UAV, -1)
        bs_actions = np.array(bs_actions).reshape(env.n_BS, -1)
        uav_directions, uav_weights = uav_actions[:,0], uav_actions[:,1]
        uav_directions = np.array(uav_directions * (self.n_actions[0]), dtype=np.int16) / self.n_actions[0]
        uav_weights = np.array(uav_weights * (self.n_actions[2]), dtype=np.int16) / self.n_actions[2]
        bs_weights = np.array(bs_actions * (self.n_actions[2]), dtype=np.int16) / self.n_actions[2]
        # uav_speeds = uav_speeds * env.uav_max_speed
        uav_speeds = env.uav_max_speed * np.ones_like(uav_weights)
        uav_directions = uav_directions * 2 * np.pi
        self.uav_weights = uav_weights.reshape(-1) * self.max_weight
        self.bs_weights = bs_weights.reshape(-1) * self.max_weight
        return uav_directions, uav_speeds

    @staticmethod
    def calculate_distance(point1, point2):
        point1 = np.expand_dims(point1, axis=1)  # shape becomes (n, 1, d)
        point2 = np.expand_dims(point2, axis=0)  # shape becomes (1, m, d)
        distance = np.sum((point1 - point2)**2, axis=-1)
        distance = np.sqrt(distance)
        return distance
    def act_offloading(self, env):
        '''返回v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2u_tran, v2i_tran, u2u_tran, u2v_tran, u2i_tran，x2x_offload更改为全局设备的计算任务idx，也就是shape = (x, x)'''
        # 0. 把所有的车辆归属到对应的无人机上，由无人机来决定是否本地计算还是卸载计算。归属过程使用weighted voronoi方法，作为参照，BS的weight设置为self.max_weight
        v2v_offload = - np.ones((env.n_Veh, env.n_Veh)).astype(int)
        v2u_offload = - np.ones((env.n_Veh, env.n_UAV)).astype(int)
        v2i_offload = - np.ones((env.n_Veh, env.n_BS)).astype(int)
        u2u_offload = - np.ones((env.n_UAV, env.n_UAV)).astype(int)
        u2v_offload = - np.ones((env.n_UAV, env.n_Veh)).astype(int)
        u2i_offload = - np.ones((env.n_UAV, env.n_BS)).astype(int)
        v2v_tran = np.zeros((env.n_Veh, env.n_Veh)).astype(int)
        v2u_tran = np.zeros((env.n_Veh, env.n_UAV)).astype(int)
        v2i_tran = np.zeros((env.n_Veh, env.n_BS)).astype(int)
        u2u_tran = np.zeros((env.n_UAV, env.n_UAV)).astype(int)
        u2v_tran = np.zeros((env.n_UAV, env.n_Veh)).astype(int)
        u2i_tran = np.zeros((env.n_UAV, env.n_BS)).astype(int)
        return v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2u_tran, v2i_tran, u2u_tran, u2v_tran, u2i_tran
