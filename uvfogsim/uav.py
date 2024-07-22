# -*- encoding: utf-8 -*-
'''
@File    :   uav.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import math
import numpy as np
from .FogNodeBase import FogNodeBase
class UAV(FogNodeBase):
    def __init__(self, uid, height, start_position, start_direction, velocity, cpu, reputation_score = 100, cheat_possibility = 0, init_revenue = 0,power_capacity = 1000):
        super().__init__(uid, start_position, reputation_score, cpu, cheat_possibility)
        self.height = height
        self.direction = start_direction
        self.neighbor_vehicles = []
        self.nearest_BS = None
        self.velocity = velocity
        self.power_capacity = power_capacity # 当前剩余的电量KMh
        self.type_name = 'UAV'
        self.total_revenues = init_revenue
        self.flied_distance = 0
        # [4,5]随机
        self.task_lambda = np.random.randint(4, 6)
        self.activated = True
        
    def update_direction(self, direction):
        self.direction = direction # 2pi

    def update_velocity(self, velocity):
        self.velocity = velocity

    def update_position(self, pos3d):
        # if self.power_capacity > 0:
        org_pos = self.position.copy()
        self.position = [pos3d[0], pos3d[1]]
        self.height = pos3d[2]
        self.flied_distance += np.sqrt((self.position[0] - org_pos[0])**2 + (self.position[1] - org_pos[1])**2)
        self.power_capacity -= self.velocity**2 / 1000 + 1 # 简化能量模型
        return True
        # else:
        #     return False
    
    def get_std_observation(self, env):
        '''返回归一化后的观测值'''
        observation = np.zeros(86)
        # 1) 自身的task_lambda属性，位置（xyz），速度，当前任务属性【cpu, datasize, 剩余ddl】(8个)，
        observation[0] = self.task_lambda / 10
        observation[1] = (self.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
        observation[2] = (self.position[1] - self.position[1]) / (env.max_range_x - env.min_range_x)
        observation[3] = (self.height) / 100
        observation[4] = self.velocity
        observation[5] = (self.task_profile[0] - env.task_cpu_min) / (env.task_cpu_max - env.task_cpu_min)
        observation[6] = (self.task_profile[1] - env.task_data_min) / (env.task_data_max - env.task_data_min)
        observation[7] = (self.task_profile[2] - env.task_ddl_min) / (env.task_ddl_max - env.task_ddl_min)
        # 2) 最近5个fog车辆+RSU的属性（speed, cpu, position, 当前任务等待时延, SINR）(7*6个)
        for vidx, veh in enumerate(self.neighbor_vehicles[:10]):
            observation[8 + vidx * 7] = veh.velocity - self.velocity
            observation[9 + vidx * 7] = (veh.CPU_frequency - env.veh_cpu_min) / (env.veh_cpu_max - env.veh_cpu_min) / (observation[0] * observation[5])
            observation[10 + vidx * 7] = (veh.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
            observation[11 + vidx * 7] = (veh.position[1] - self.position[1]) / (env.max_range_y - env.min_range_y)
            observation[12 + vidx * 7] = 0
            observation[13 + vidx * 7] = veh.get_task_delay() / 0.5
            observation[14 + vidx * 7] = ((env.get_fading_between(self, veh) - 80) / 120).mean()
        # RSU是self.nearst_BS
        observation[77] = 0
        observation[78] = (self.nearest_BS.CPU_frequency - env.veh_cpu_min) / (env.veh_cpu_max - env.veh_cpu_min) / (observation[0] * observation[5])
        observation[79] = (self.nearest_BS.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
        observation[80] = (self.nearest_BS.position[1] - self.position[1]) / (env.max_range_y - env.min_range_y)
        observation[81] = 0
        observation[82] = self.nearest_BS.get_task_delay() / 0.5
        observation[83] = ((env.get_fading_between(self, self.nearest_BS) - 80) / 120).mean()
        observation[84] = (self.id) / 40 + 5
        observation[85] = min(10, len(self.neighbor_vehicles)) / 10
        return observation


    
