# -*- encoding: utf-8 -*-
'''
@File    :   Cluster_Algorithm_Module.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import math
import numpy as np
from .Base_Algorithm_Module import Base_Algorithm_Module
from sklearn.cluster import KMeans
import warnings
def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
class Cluster_Algorithm_Module(Base_Algorithm_Module):
    def __init__(self):
        super().__init__()
        warnings.filterwarnings("ignore")
        self.uav_weights = np.ones(shape=(1))
        self.max_weight = 1
    def act_RB_allocation(self, env):
        return super().act_RB_allocation(env)
    def act_mining_and_pay(self, env):
        return super().act_mining_and_pay(env)
    def act_verification(self, env):
        return super().act_verification(env)
    def act_mobility(self, env, vehicle_positions):
        '''聚类算法，根据车辆的position，寻找n_UAV个聚类中心；如果车辆数量小于n_UAV，则不动'''
        uav_directions = np.zeros(shape=(env.n_UAV))
        uav_speeds = np.zeros(shape=(env.n_UAV))
        
        if len(vehicle_positions) < env.n_UAV:
            return uav_directions, uav_speeds

        # K-means 聚类
        kmeans = KMeans(n_clusters=env.n_UAV, max_iter=10).fit(vehicle_positions)

        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        uav_positions = env.get_uav_positions()
        # 计算 UAV 的方向和速度
        # 为每个无人机分配最近的聚类中心
        for i, uav_position in enumerate(uav_positions):
            closest_center_index = np.argmin([distance(uav_position, center) for center in cluster_centers])
            target_position = cluster_centers[closest_center_index]
            # cluster_centers是ndarray，删除第closest_center_index行的元素
            cluster_centers = np.delete(cluster_centers, closest_center_index, axis=0)

            # 计算方向 (弧度)
            direction = (np.arctan2(target_position[1] - uav_position[1], target_position[0] - uav_position[0]) + 2 * np.pi) % ( 2* np.pi)
            uav_directions[i] = direction

            # 计算速度
            dist = distance(uav_position, target_position)
            uav_speeds[i] = int(min(dist / env.time_step, env.uav_max_speed)/10) * 10
        self.uav_weights = np.ones(shape=(env.n_UAV))
        return uav_directions, uav_speeds
    
    def act_offloading(self, env):
        '''返回v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2u_tran, v2i_tran, u2u_tran, u2v_tran, u2i_tran，x2x_offload更改为全局设备的计算任务idx，也就是shape = (x, x)'''
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
        vehicle_by_index = env.vehicle_by_index
        bs_weights = self.max_weight * np.ones(env.n_BS)
        # 权重公式是： - gain - weight
        v2u_gains = np.mean(env.V2UChannel_with_fastfading, axis = 2) / 100 # 信道平均增益，shape = (n_Veh, n_UAV) 
        v2i_gains = np.mean(env.V2IChannel_with_fastfading, axis = 2) / 100
        u_wei = np.repeat(self.uav_weights[np.newaxis, :], env.n_Veh, axis = 0)
        v2u_gains_with_weight = v2u_gains - u_wei
        i_wei = np.repeat(bs_weights[np.newaxis, :], env.n_Veh, axis = 0)
        v2i_gains_with_weight = v2i_gains - i_wei
        # 先把所有车辆根据v2u_gains和v2i_gains归属到无人机或者BS上
        v2u_min = np.min(v2u_gains_with_weight, axis=1, keepdims=True)
        v2i_min = np.min(v2i_gains_with_weight, axis=1, keepdims=True)
        min_choice = v2u_min < v2i_min
        v2u_assigned = (v2u_gains_with_weight == v2u_min) & min_choice # shape = (n_Veh, n_UAV)
        v2i_assigned = (v2i_gains_with_weight == v2i_min) & ~min_choice
        # 接着讨论在uav或者bs范围内，车辆怎么计算。目前是简化版本，直接卸载给uav或者bs进行计算，不考虑v2v卸载
        for idx, vi in enumerate(vehicle_by_index):
            # 1. 如果车辆能自己计算，则不进行卸载
            if vi.get_require_cpu() <= 0:
                continue
            
            # 2. 如果车辆不能自己计算，则把任务信息提交给对应的uav或者bs进行计算
            # 2.1 判断v2u_assigned[idx]是否存在True，如果存在，则把任务信息提交给对应的uav
            if np.sum(v2u_assigned[idx]) > 0 and len(vi.task_queue) > 0:
                # 2.1.1 找到对应的uav
                uav_idx = np.argmax(v2u_assigned[idx])
                v2u_offload[idx, uav_idx] = 0 # 默认把第一个任务卸载了
                # RB的选择就随机好了，从env.n_RB中选择一个
                v2u_tran[idx, uav_idx] = np.random.choice(env.n_RB)
            # 2.2 判断v2i_assigned[idx]是否存在True，如果存在，则把任务信息提交给对应的bs
            elif np.sum(v2i_assigned[idx]) > 0 and len(vi.task_queue) > 0:
                bs_idx = np.argmax(v2i_assigned[idx])
                v2i_offload[idx, bs_idx] = 0
                v2i_tran[idx, bs_idx] = np.random.choice(env.n_RB)
                
            
            # 3. 所有的uav和bs收集所有的任务信息，进行区域内优化决策。
        return v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2u_tran, v2i_tran, u2u_tran, u2v_tran, u2i_tran