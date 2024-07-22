# -*- encoding: utf-8 -*-
'''
@File    :   Random_Algorithm_Module.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import math
from pickletools import int4
import numpy as np
from .Base_Algorithm_Module import Base_Algorithm_Module
class Random_Algorithm_Module(Base_Algorithm_Module):
    def __init__(self, env):
        super().__init__()
        self.uav_weights = np.ones(env.n_UAV)
        self.bs_weights = np.ones(env.n_BS)
        self.max_weight = 2

    def act_mobility(self, env, vehicle_ids, vehicle_positions):
        self.uav_weights = np.random.rand(env.n_UAV).reshape(-1) * self.max_weight
        self.bs_weights = np.random.rand(env.n_BS).reshape(-1) * self.max_weight
        # return 2*math.pi * (np.random.random(size=(env.n_UAV))), env.uav_max_speed * np.random.random(size=(env.n_UAV))
        return 2*math.pi * (np.random.randint(5, size=(env.n_UAV))) / 4, env.uav_max_speed * np.ones((env.n_UAV))

    
    def get_assigned(self, env, v2u_distance, v2i_distance, uav_weights = None, bs_weights = None):
        if uav_weights is None: uav_weights = self.uav_weights
        if bs_weights is None: bs_weights = self.bs_weights
        # 根据无人机，车辆，基站的位置，计算weighted voronoi，返回v2u_assigned和v2i_assigned。
        v2u_assigned = np.zeros((env.n_Veh, env.n_UAV))
        v2i_assigned = np.zeros((env.n_Veh, env.n_BS))
        # weighted voronoi的公式
        for idx, veh in enumerate(env.vehicle_by_index):
            min_dist = None
            min_j = 0
            uav_or_bs = 0
            for j, uav in enumerate(env.UAVs):
                dist = env.V2UChannel_with_fastfading[idx,j,0] - uav_weights[j] 
                # * uav.CPU_frequency
                if min_dist is None or (min_dist > dist and dist <= 500):
                    min_dist = dist
                    min_j = j
                    uav_or_bs = 1
            for k, bs in enumerate(env.BSs):
                # dist = v2i_distance[idx, k] - bs_weights[k] 
                dist = env.V2IChannel_with_fastfading[idx, k,0] - bs_weights[k] 
                # * bs.CPU_frequency
                if min_dist is None or (min_dist > dist and dist <= 1000):
                    min_dist = dist
                    min_j = k
                    uav_or_bs = 2
            if uav_or_bs == 1:
                v2u_assigned[idx, min_j] = 1
            elif uav_or_bs == 2:
                v2i_assigned[idx, min_j] = 1
        return v2u_assigned, v2i_assigned
            
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