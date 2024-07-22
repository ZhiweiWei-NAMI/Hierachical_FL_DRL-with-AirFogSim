# -*- encoding: utf-8 -*-
'''
@File    :   vehicle.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''




import numpy as np

from .FogNodeBase import FogNodeBase

class Vehicle(FogNodeBase):
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, id, start_position, start_direction, velocity, cpu, serving = True, task_lambda = 0, init_revenue = 0, init_score = 0, cheat_possibility = 0):
        super().__init__(id, start_position, init_score, cpu, cheat_possibility)
        self.direction = start_direction
        self.velocity = velocity
        self.task_lambda = task_lambda
        self.computing_res_alloc = [] # 分配计算资源策略,每个step更新（task_queue中的任务，分配的资源x）
        self.serving = serving 
        self.assigned_to = -1
        self.total_revenues = init_revenue # 钱包，记录的是自身有多少的奖励
        self.type_name = 'Vehicle'
        self.neighbor_vehicles = []
        self.nearest_BS = None
        self.reward = 0
        self.pCPU = 0
        self.served_last_period = False
        self.task_profile = None

    def update_position(self, position):
        # assert self.is_running
        self.position = position
    
    def update_velocity(self, velocity):
        # assert self.is_running
        self.velocity = velocity

    def update_direction(self, direction):
        # assert self.is_running
        self.direction = direction # 360 degree
    
    def update_time(self, time):
        self.time = time
        self.time = round(self.time, 1)

    def get_std_observation(self, env):
        '''返回归一化后的观测值'''
        observation = np.zeros(86)
        # 1) 自身的task_lambda属性，位置（xyz），速度，当前任务属性【cpu, datasize, 剩余ddl】(8个)，
        observation[0] = self.task_lambda / 10
        observation[1] = (self.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
        observation[2] = (self.position[1] - self.position[1]) / (env.max_range_y - env.min_range_y)
        observation[3] = 0
        observation[4] = self.velocity
        observation[5] = (self.task_profile[0] - env.task_cpu_min) / (env.task_cpu_max - env.task_cpu_min)
        observation[6] = (self.task_profile[1] - env.task_data_min) / (env.task_data_max - env.task_data_min)
        observation[7] = (self.task_profile[2] - env.task_ddl_min) / (env.task_ddl_max - env.task_ddl_min)
        # 2) 最近10个fog车辆+RSU的属性（speed, cpu, position, 当前任务等待时延, SINR）(7*6个)
        for vidx, veh in enumerate(self.neighbor_vehicles[:10]):
            observation[8 + vidx * 7] = veh.velocity - self.velocity
            observation[9 + vidx * 7] = (veh.CPU_frequency - env.veh_cpu_min) / (env.veh_cpu_max - env.veh_cpu_min) / (observation[0] * observation[5])
            observation[10 + vidx * 7] = (veh.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
            observation[11 + vidx * 7] = (veh.position[1] - self.position[1]) / (env.max_range_y - env.min_range_y)
            observation[12 + vidx * 7] = 0
            observation[13 + vidx * 7] = veh.get_task_delay() / 0.5
            observation[14 + vidx * 7] = ((env.get_fading_between(self, veh) - 80) / 120).mean()
        # RSU是self.nearst_BS,63+14=77
        observation[77] = 0
        observation[78] = (self.nearest_BS.CPU_frequency - env.veh_cpu_min) / (env.veh_cpu_max - env.veh_cpu_min) / (observation[0] * observation[5])
        observation[79] = (self.nearest_BS.position[0] - self.position[0]) / (env.max_range_x - env.min_range_x)
        observation[80] = (self.nearest_BS.position[1] - self.position[1]) / (env.max_range_y - env.min_range_y)
        observation[81] = 0
        observation[82] = self.nearest_BS.get_task_delay() / 0.5
        observation[83] = ((env.get_fading_between(self, self.nearest_BS) - 80) / 120).mean()
        observation[84] = (self.id) / 200
        observation[85] = min(10, len(self.neighbor_vehicles)) / 10
        return observation