# -*- encoding: utf-8 -*-
'''
@File    :   environment.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import math
from random import shuffle
from xmlrpc.client import boolean
from torch_geometric.data import Data
import torch

from uvfogsim.blockchain import Blockchain
from .V2IChannel import V2IChannel
from .V2VChannel import V2VChannel
from .V2UChannel import V2UChannel
from .U2IChannel import U2IChannel
from .U2UChannel import U2UChannel
from .I2IChannel import I2IChannel
from .U2VChannel import U2VChannel
from sklearn.cluster import KMeans
# 车辆和无人机
from .vehicle import Vehicle
from .uav import UAV
from .task import Task
from .bs import BS
# 基础
import numpy as np
import copy
from collections import deque
class Environment:
    def __init__ (self, args, draw_it = False, n_UAV = 5, time_step = 0.1, TTI = 0.01):
        self.vehicles = {} # 通过v_id索引车辆
        self.args = args
        self.serving_vehicles = {}
        self.task_vehicles = {}
        self.UAVs = [] # 列表
        self.BSs = []
        self.uav_max_speed = 25 # m/s
        self.veh_cpu_max = 2.5 # GHz
        self.veh_cpu_min = 1.5 # GHz
        self.uav_cpu = 5
        self.BS_cpu = 10 # 假设是10个计算机组成的云端
        self.v_neighbor_Veh = args.v_neighbor_Veh # 每一个车辆最多有几个neighbor
        self.task_lambda_type = [2, 5, 10] # 任务车辆的lambda类型，假设任务的生成服从泊松分布
        self.task_lambda_poss = [0.7, 0.3, 0.0] # 每种类型的出现概率
        self.max_serving_vehicles = self.args.n_serving_veh
        self.V2V_power_dB = 23 # dBm 记录的都是最大功率
        self.V2I_power_dB = 26
        self.V2U_power_dB = 26
        self.U2U_power_dB = 26
        self.U2I_power_dB = 26
        self.U2V_power_dB = 26
        self.I2I_power_dB = 26
        self.I2V_power_dB = 26
        self.I2U_power_dB = 26
        self.half_duplex = False # 默认全双工
        self.sig2_dB = -104
        self.sig2 = 10**(self.sig2_dB/10)
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.V2U_Shadowing = []
        self.U2U_Shadowing = []
        self.U2V_Shadowing = []
        self.U2I_Shadowing = []
        self.veh_delta_distance = []
        self.uav_delta_distance = []
        self.n_RB = 20 # LTE场景，大致是100RB
        self.RB_bandwidth = 1 # 180kHz = 12子载波 * 15kHz
        self.bandwidth = self.n_RB * self.RB_bandwidth
        self.n_BS = args.n_RSU
        self.n_RSU = args.n_RSU
        self.BS2C_rate = 300 # Mbps，代表BS->cloud的传输速率
        self.n_UAV = n_UAV
        self.hei_UAV = 100 
        # 数据边界：0.00,0.00,11565.39,8619.57
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.min_range_x = 250
        self.max_range_x = 2000
        self.min_range_y = 250
        self.max_range_y = 1500
        self.BS_positions = [] # 二维数组格式输入BS的位置
        self.V2VChannel = None
        self.V2IChannel = None
        self.V2UChannel = None
        self.U2UChannel = None
        self.U2IChannel = None
        self.I2IChannel = None

        self.V2V_Interference = None
        self.V2I_Interference = None
        self.V2U_Interference = None
        self.U2U_Interference = None
        self.U2I_Interference = None
        self.I2I_Interference = None
        # 是否可以连接的信道, 这里双向都要有，因为功率不同
        self.V2V_active_links = None
        self.V2I_active_links = None
        self.V2U_active_links = None
        self.U2U_active_links = None
        self.U2V_active_links = None
        self.U2I_active_links = None
        self.I2I_active_links = None
        self.I2U_active_links = None
        self.I2V_active_links = None

        self.V2V_Rate = None
        self.V2I_Rate = None
        self.V2U_Rate = None
        self.U2U_Rate = None
        self.U2I_Rate = None
        self.U2V_Rate = None
        self.I2I_Rate = None
        self.I2U_Rate = None
        self.I2V_Rate = None

        # task
        self.task_cpu_min = 0.3 # Giga-Cycle
        self.task_cpu_max = 0.4
        self.task_data_min = 0.5 # Mb
        self.task_data_max = 1.5
        self.task_ddl_min = 0.2  # s
        self.task_ddl_max = 0.8
        self.to_offload_tasks = []
        self.offloading_tasks = []
        self.succeed_tasks = {}
        self.failed_tasks = []
        self.to_verify_tasks = []
        self.verified_tasks = []
        self.cur_time = 0 
        self.time_step = time_step # 最少不能低于0.1！！！和sumo数据是相关的
        assert self.time_step>=0.1, "time_step最少不能低于0.1"
        self.last_time = self.cur_time
        self.TTI = TTI # computation and transmission time interval, 太小的话会影响到运行效率，表示传输的间隔
        assert self.TTI <= self.time_step, 'TTI 必须小于车辆运行的time_step'
        self.max_wait_transmit_threshold = 5 * self.TTI
        self.cur_fig = None
        self.draw_it = draw_it
        self.initialized = False
        self.vid_index = {} # 用于记录每个车辆的id对应所有channel中的索引
        self.task_id_cnt = 0
        # blockchain
        self.blockchain = None
        self.to_pay_tasks = {}
    @property
    def avg_task_data(self):
        return (self.task_data_min + self.task_data_max) / 2
    
    @property
    def avg_task_cpu(self):
        return (self.task_cpu_min + self.task_cpu_max) / 2
    @property
    def avg_task_ddl(self):
        return (self.task_ddl_min + self.task_ddl_max) / 2
    # ---------------------------------------------------------------------------------------------------------------------
    # 获取相关environment的数据信息
    def get_avg_task_latency(self):
        ''' 获取平均完成计算任务的延迟
        '''
        if len(self.succeed_tasks) == 0:
            return 0
        return np.mean([task.service_delay for task in self.succeed_tasks.values()])
    def get_succeed_task_cnt(self):
        ''' 获取成功的任务的数量
        '''
        return len(self.succeed_tasks)
    def get_failed_task_cnt(self):
        ''' 获取失败的任务的数量
        '''
        return len(self.failed_tasks)
    def get_failed_task_type_reason_cnt(self):
        ''' 获取失败的任务的类型和原因的统计信息
        return reason_cnt: 一个二维字典，第一维是task_type，第二维是reason
        '''
        reason_cnt = {}
        FAILURE_REASONS = Task.FAILURE_REASONS
        TASK_TYPES = Task.TASK_TYPES
        # 通过FAILURE_REASONSh和TASK_TYPES构建字典
        for k,v in TASK_TYPES.items():
            reason_cnt[k] = {}
            for k2,v2 in FAILURE_REASONS.items():
                reason_cnt[k][k2] = 0

        for task_dict in self.failed_tasks:
            task = task_dict['task']
            reason = task_dict['reason']
            task_type = task.task_type
            reason_str = None
            task_type_str = None
            for k,v in FAILURE_REASONS.items():
                if v == reason:
                    reason_str = k
                    break
            for k,v in TASK_TYPES.items():
                if v == task_type:
                    task_type_str = k
                    break
            if reason_str is None or task_type_str is None:
                raise ValueError('reason_str or task_type_str is None')
            reason_cnt[task_type_str][reason_str] += 1
        return reason_cnt
    @staticmethod
    def between(a, b):
        return np.random.rand() * (b-a)+a
    @staticmethod
    def distance(pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        return np.sqrt(np.sum((pos1 - pos2)**2))
    @property
    def n_Veh(self):
        return len(self.vehicles.keys())
    @property
    def n_SV(self):
        return len(self.serving_vehicles.keys())
    @property
    def n_TV(self):
        return len(self.task_vehicles.keys())
    def initialize(self, bound, bs_positions):
        self.initialized = True
        self.max_serving_vehicles = self.args.n_veh // 2
        self.cur_fig = None
        self.min_x = bound[0]
        self.min_y = bound[1]
        self.max_x = bound[2]
        self.max_y = bound[3]
        self.vehicles = {}
        self.serving_vehicles = {}
        self.task_vehicles = {}
        self.task_id_cnt = 0
        self.vid_index = {}
        self.initialize_BSs(bs_positions)
        self.initialize_UAVs()
        self.initialize_Channels()
        self.initialize_Interference_and_active()
        self.renew_channels_fastfading()
        self.generate_tasks()
        self.offloading_tasks = []
        self.to_offload_tasks = []
        self.to_verify_tasks = []
        self.succeed_tasks = {}
        self.failed_tasks = []
        self.verified_tasks = []
        self.cur_time = 0 
        self.last_time = self.cur_time
        self.blockchain = None
        # Blockchain(self.cur_time, self.args.mine_time_threshold, self.args.transaction_threshold, self.args.consensus_type)
        self.to_pay_tasks = {}

    @property
    def total_task_lambda(self):
        total_task_lambda = 0
        for tv in self.task_vehicles.values():
            total_task_lambda += tv.task_lambda
        return total_task_lambda

    def initialize_Interference_and_active(self):
        self.V2I_Interference = np.zeros((self.n_Veh, self.n_BS)) + self.sig2 # 默认每个车辆归属于一个基站
        self.V2V_Interference = np.zeros((self.n_Veh, self.n_Veh)) + self.sig2
        self.V2U_Interference = np.zeros((self.n_Veh, self.n_UAV)) + self.sig2
        self.U2I_Interference = np.zeros((self.n_UAV, self.n_BS)) + self.sig2
        self.U2V_Interference = np.zeros((self.n_UAV, self.n_Veh)) + self.sig2
        self.U2U_Interference = np.zeros((self.n_UAV, self.n_UAV)) + self.sig2
        self.I2I_Interference = np.zeros((self.n_BS, self.n_BS)) + self.sig2
        self.I2V_Interference = np.zeros((self.n_BS, self.n_Veh)) + self.sig2
        self.I2U_Interference = np.zeros((self.n_BS, self.n_UAV)) + self.sig2
        # 是否可以连接的信道
        self.V2V_active_links = np.zeros((self.n_Veh, self.n_Veh), dtype='bool')
        self.V2I_active_links = np.zeros((self.n_Veh, self.n_BS), dtype='bool')
        self.V2U_active_links = np.zeros((self.n_Veh, self.n_UAV), dtype='bool')
        self.U2U_active_links = np.zeros((self.n_UAV, self.n_UAV), dtype='bool')
        self.U2V_active_links = np.zeros((self.n_UAV, self.n_Veh), dtype='bool')
        self.U2I_active_links = np.zeros((self.n_UAV, self.n_BS), dtype='bool')
        self.I2U_active_links = np.zeros((self.n_BS, self.n_UAV), dtype='bool')
        self.I2V_active_links = np.zeros((self.n_BS, self.n_Veh), dtype='bool')
        self.I2I_active_links = np.zeros((self.n_BS, self.n_BS), dtype='bool')
    def initialize_Channels(self):
        self.V2VChannel = V2VChannel(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2IChannel = V2IChannel(self.n_Veh, self.n_BS, self.n_RB, self.BS_positions)
        self.V2UChannel = V2UChannel(self.n_Veh, self.n_RB, self.n_UAV, self.hei_UAV)
        self.U2UChannel = U2UChannel(self.n_RB, self.n_UAV, self.hei_UAV)
        self.U2IChannel = U2IChannel(self.n_RB, self.n_BS, self.n_UAV, self.hei_UAV, self.BS_positions)
        self.I2IChannel = I2IChannel(self.n_RB, self.n_BS, self.BS_positions)
    
    def get_uav_positions(self):
        # 获取所有无人机的位置
        uav_positions = []
        # UAVs是dict，获取所有的key，然后存储所有的无人机的位置到uav_positions
        for uav in self.UAVs:
            uav_positions.append(uav.position)
        return np.array(uav_positions)
    def get_bs_positions(self):
        return np.array(self.BS_positions)
    def get_vehicle_positions(self):
        # 获取所有车辆的位置
        vehicle_positions = []
        # vehicles是dict，获取所有的key，然后存储所有的车辆的位置到vehicle_positions
        for vehicle in self.vehicle_by_index:
            vehicle_positions.append(vehicle.position)
        return vehicle_positions
    def get_BS_positions(self):
        return np.array(self.BS_positions)
    
    def initialize_BSs(self, BS_positions):
        self.BS_positions = BS_positions if BS_positions is not None else np.zeros((self.n_BS, 2))
        self.BSs = []
        for i in range(self.n_BS):
            bs_i = BS(i, self.BS_positions[i], self.BS_cpu, self.BS2C_rate)
            self.BSs.append(bs_i)

    def initialize_UAVs(self, vehicle_positions = None):
        self.UAVs = []
        for uid in range(self.n_UAV):
            # pos_uav = cluster_centers[uid]
            pos_uav = [np.random.rand() * 1000 + 500, np.random.rand() * 1000 + 500]
            # pos_uav = [(self.max_x+self.min_x) / 2, (self.max_y + self.min_y) / 2]
            uav_i = UAV(uid, self.hei_UAV, pos_uav, 0, 0, self.uav_cpu)
            self.UAVs.append(uav_i)
    def generate_vehicle(self, row_data):
        # serve_or_offload = np.random.rand()>0.5
        if len(self.serving_vehicles) < self.max_serving_vehicles: # 服务车辆
            veh = Vehicle(row_data['id'], [row_data['x'], row_data['y']], row_data['angle'], row_data['speed'], self.between(self.veh_cpu_min, self.veh_cpu_max), serving = True)
        else:   # 任务车辆
            poss = self.task_lambda_poss
            task_type = np.random.choice(range(len(self.task_lambda_type)), p = poss) # 三种类型
            
            veh = Vehicle(row_data['id'], [row_data['x'], row_data['y']], row_data['angle'], row_data['speed'], 0, serving = False, task_lambda = self.task_lambda_type[task_type])
            
        veh.time = self.cur_time
        return veh

    def update_vehicle(self, row_data):
        veh = self.vehicles[int(row_data['id'])]
        assert veh is not None
        veh.update_direction(row_data['angle'])
        veh.update_position([row_data['x'], row_data['y']])
        veh.update_velocity(row_data['speed'])
        veh.update_time(self.cur_time)

    def check_bound(self):
        '''检查无人机的位置有没有飞出边界'''
        flags = [False for _ in range(self.n_UAV)]
        y_up_bound = self.max_y
        y_low_bound = self.min_y
        x_up_bound = self.max_x
        x_low_bound = self.min_x
        for idx, uav in enumerate(self.UAVs):
            if uav.position[0] <= x_low_bound:
                uav.position[0] = x_low_bound
                flags[idx] = True
            elif uav.position[0] >= x_up_bound:
                uav.position[0] = x_up_bound
                flags[idx] = True
            if uav.position[1] <= y_low_bound:
                uav.position[1] = y_low_bound
                flags[idx] = True
            elif uav.position[1] >= y_up_bound:
                uav.position[1] = y_up_bound
                flags[idx] = True
        return flags

    def renew_uav_positions(self, all_uav_positions):
        '''输入每个无人机的移动方向和速度'''
        # 假设directions和speeds是list
        for uid in range(len(self.UAVs)):
            self.UAVs[uid].update_direction(0)
            self.UAVs[uid].update_velocity(15)
            self.UAVs[uid].update_position(all_uav_positions[uid])
        self.update_distances()
        
    def get_fading_between(self, node1, node2):
        # 用indexof获取对应的idx，然后使用Channel_with_fastfading检索对应的pathloss
        if isinstance(node1, Vehicle):
            idx1 = self.vehicle_by_index.index(node1)
        elif isinstance(node1, UAV):
            idx1 = self.UAVs.index(node1)
        elif isinstance(node1, BS):
            idx1 = self.BSs.index(node1)
        else:
            raise ValueError('node1 is not in the environment')
        if isinstance(node2, Vehicle):
            idx2 = self.vehicle_by_index.index(node2)
        elif isinstance(node2, UAV):
            idx2 = self.UAVs.index(node2)
        elif isinstance(node2, BS):
            idx2 = self.BSs.index(node2)
        else:
            raise ValueError('node2 is not in the environment')
        if isinstance(node1, Vehicle) and isinstance(node2, Vehicle):
            return self.V2VChannel_with_fastfading[idx1, idx2]
        elif isinstance(node1, Vehicle) and isinstance(node2, BS):
            return self.V2IChannel_with_fastfading[idx1, idx2]
        elif isinstance(node1, Vehicle) and isinstance(node2, UAV):
            return self.V2UChannel_with_fastfading[idx1, idx2]
        elif isinstance(node1, UAV) and isinstance(node2, UAV):
            return self.U2UChannel_with_fastfading[idx1, idx2]
        elif isinstance(node1, UAV) and isinstance(node2, Vehicle):
            return self.U2VChannel_with_fastfading[idx1, idx2]
        elif isinstance(node1, UAV) and isinstance(node2, BS):
            return self.U2IChannel_with_fastfading[idx1, idx2]
        raise ValueError('node1 and node2 are not in the same channel')

        

    def renew_veh_positions(self, vehicle_ids, rows):
        # 把vehicle_ids保证是int
        vehicle_ids = [int(v_id) for v_id in vehicle_ids]
        tmp_keys = list(self.vehicles.keys()).copy()
        removed_vehicle_id = []
        for v_id in tmp_keys:
            # rows是字典，判断vehicle.id是否在字典中
            if v_id in vehicle_ids:
                row_data = rows[v_id]
                vehicle = self.vehicles[v_id]
            else:
                row_data = None
                removed_vehicle_id.append(v_id)
            if row_data is not None:
                vehicle.update_direction(row_data['angle'])
                vehicle.update_position([row_data['x'], row_data['y']])
                vehicle.update_velocity(row_data['speed'])
                vehicle.update_time(self.cur_time)
                vehicle.access1 = False
                vehicle.access2 = False
                # 删除已经更新的车辆
                del rows[v_id]
            else:
                # 说明这个车辆在当前时间没有位置，就是消失了，需要不激活
                self.deactivate_veh(v_id)
                if self.vehicles[v_id].serving:
                    del self.serving_vehicles[v_id]
                else:
                    del self.task_vehicles[v_id]
                del self.vehicles[v_id]
        # 对于row_data中剩下的车辆，说明是新出现的车辆
        for vehicle_id in rows.keys():
            vehicle_id = int(vehicle_id)
            row_data = rows[vehicle_id]
            vehicle = self.generate_vehicle(row_data)
            self.vehicles[vehicle_id] = vehicle
            if vehicle.serving:
                self.serving_vehicles[vehicle_id] = vehicle
            else:
                self.task_vehicles[vehicle_id] = vehicle
            self.add_vehicle_channel(vehicle_id)
        assert self.n_Veh == self.V2VChannel.n_Veh
        sorted_vid_index = sorted(self.vid_index.items(), key=lambda x:x[1])
        sorted_vids = [vid for (vid, _) in sorted_vid_index]
        self.vehicle_by_index = [self.vehicles[vid] for vid in sorted_vids]
        self.index_vid = {v: k for k, v in self.vid_index.items()}
        self.update_distances()
        return removed_vehicle_id

    def update_distances(self):
        z = np.array([[c.position[0], c.position[1]] for c in self.vehicle_by_index])
        z2 = np.array([[c.position[0], c.position[1]] for c in self.UAVs])
        z3 = np.array([[c.position[0], c.position[1]] for c in self.BSs])

        self.v2u_distance = np.sqrt(np.sum((z[:, None] - z2)**2, axis=-1))
        self.v2i_distance = np.sqrt(np.sum((z[:, None] - z3)**2, axis=-1))
        self.v2v_distance = np.sqrt(np.sum((z[:, None] - z)**2, axis=-1))
        u2v_distance = np.sqrt(np.sum((z2[:, None] - z)**2, axis=-1))
        # 遍历每个BS，把task_v_num清空
        for bs in self.BSs:
            bs.task_v_num = 0

        # 记录每个task vehicle距离自身最近的v_neighbor_Veh个服务车辆，存储在vehicle.neighbor_vehicles中
        for vidx, vehicle in enumerate(self.vehicle_by_index):
            nearBS = self.BSs[np.argmin(self.v2i_distance[vidx])]
            vehicle.nearest_BS = nearBS
            if vehicle.serving: continue
            vehicle.neighbor_vehicles = []
            # 通过v2v_distance获取到所有车辆到自身的距离，然后排序，获取到最近的v_neighbor_Veh个车辆
            sorted_idx = np.argsort(self.v2v_distance[vidx])
            for i in range(1, len(sorted_idx)):
                if len(vehicle.neighbor_vehicles) >= self.v_neighbor_Veh:
                    break
                if self.vehicle_by_index[sorted_idx[i]].serving and self.vehicle_by_index[sorted_idx[i]].nearest_BS == vehicle.nearest_BS and self.distance(vehicle.position, self.vehicle_by_index[sorted_idx[i]].position) <= self.args.V2V_communication_range:
                # if self.vehicle_by_index[sorted_idx[i]].serving:
                    vehicle.neighbor_vehicles.append(self.vehicle_by_index[sorted_idx[i]])
            # 距离最近的BS
            if not vehicle.serving:
                nearBS.task_v_num += 1

        # 记录每个uav距离自身最近的v_neighbor_Veh个车辆，存储在uav.neighbor_vehicles中
        for uidx, uav in enumerate(self.UAVs):
            nearBS = self.BSs[np.argmin(self.v2i_distance[uidx])]
            uav.nearest_BS = nearBS
        for uidx, uav in enumerate(self.UAVs):
            uav.neighbor_vehicles = []
            sorted_idx = np.argsort(u2v_distance[uidx])
            for i in range(1, len(sorted_idx)):
                if len(uav.neighbor_vehicles) >= self.v_neighbor_Veh:
                    break
                if self.vehicle_by_index[sorted_idx[i]].serving and self.vehicle_by_index[sorted_idx[i]].nearest_BS == uav.nearest_BS and self.distance(uav.position, self.vehicle_by_index[sorted_idx[i]].position) <= self.args.UAV_communication_range:
                # if self.vehicle_by_index[sorted_idx[i]].serving:
                    uav.neighbor_vehicles.append(self.vehicle_by_index[sorted_idx[i]])
            # 距禇最近的BS
            nearBS.task_v_num += 1

    def add_vehicle_channel(self, v_id):
        # 更新vid_index
        self.vid_index[v_id] = self.n_Veh - 1
        self.V2VChannel.add_vehicle_shadow()
        self.V2IChannel.add_vehicle_shadow()
        self.V2UChannel.add_vehicle_shadow()
        # self.V2V_active_links 需要在最后额外多加一行和一列,都是False的数值
        self.V2V_active_links = np.pad(self.V2V_active_links, ((0, 1), (0, 1)), 'constant', constant_values=False)
        self.V2I_active_links = np.pad(self.V2I_active_links, ((0, 1), (0, 0)), 'constant', constant_values=False)
        self.I2V_active_links = np.pad(self.I2V_active_links, ((0, 1), (0, 0)), 'constant', constant_values=False)
        # self.V2U_active_links 需要在最后额外多加一行,都是False的数值
        self.V2U_active_links = np.pad(self.V2U_active_links, ((0, 1), (0, 0)), 'constant', constant_values=False)
        self.U2V_active_links = np.pad(self.U2V_active_links, ((0, 0), (0, 1)), 'constant', constant_values=False)

    def remove_vehicle_channel(self, v_id):
        
        self.V2VChannel.remove_vehicle_shadow(v_id, self.vid_index)
        self.V2IChannel.remove_vehicle_shadow(v_id, self.vid_index)
        self.V2UChannel.remove_vehicle_shadow(v_id, self.vid_index)
        index = self.vid_index[v_id]
        # 更新vid_index
        tmp_keys = list(self.vid_index.keys()).copy()
        for vid in tmp_keys:
            if self.vid_index[vid] > index:
                self.vid_index[vid] -= 1
            elif self.vid_index[vid] == index:
                del self.vid_index[vid]

    def deactivate_veh(self, v_id):
        # 1 车辆自身结束
        v_id = int(v_id)
        vi = self.vehicles[v_id]
        idx = self.vid_index[v_id]
        # 计算任务判断为失败
        # 2 所有V2V,V2I,V2U的链路都要deactivate, 假设是dict，直接删除
        slice_idx = [i for i in range(0,idx)] + [i for i in range(idx+1, len(self.vehicles.keys()))]
        self.V2V_active_links = self.V2V_active_links[slice_idx, :]
        self.V2V_active_links = self.V2V_active_links[:, slice_idx]
        self.V2I_active_links = self.V2I_active_links[slice_idx, :]
        self.V2U_active_links = self.V2U_active_links[slice_idx, :]
        self.U2V_active_links = self.U2V_active_links[:, slice_idx]
        # 3.1 遍历to_offload_task
        if not vi.serving:
            tmp_remove_task = []
            for task in self.to_offload_tasks:
                if task.g_veh == vi:
                    tmp_remove_task.append(task)
            for task in tmp_remove_task:
                self.to_offload_tasks.remove(task)
                # self.failed_tasks.append({
                #     'task':task,
                #     'reason':Task.OUT_OF_VEHICLE
                # })
        tmp_remove_task = []
        # 3.2 遍历offloading_tasks
        for task_and_path in self.offloading_tasks:
            '''{
                'task':self.to_offload_tasks[task_idx], 
                'path':offload_objs, # 指向任务卸载的对象
                'RBs':np.zeros(self.n_RB, dtype='bool'), #标识多少的RB被分配了，是一个0-1 vector
                'mode':None, # X2X 的连接
                'TX_idx':txidx,
                'RX_idx':rxidx,
                'Activated':False, # 当前的TTI是否被传输
            }'''
            task = task_and_path['task']
            path = task_and_path['path']
            if vi in task.routing or vi in path:
                tmp_remove_task.append(task_and_path)
        for task_and_path in tmp_remove_task:
            self.offloading_tasks.remove(task_and_path)
            # self.failed_tasks.append({
            #     'task':task_and_path['task'],
            #     'reason':Task.OUT_OF_VEHICLE
            # })
        # 3.3 遍历所有vehicle、UAV、RSU的task_queue，如果发送方是vi，则去除
        if vi.serving:
            for task in vi.task_queue:
                pass
            #     self.failed_tasks.append({
            #     'task':task,
            #     'reason':Task.OUT_OF_VEHICLE
            # })
        else:
            for tmp_v in self.vehicle_by_index:
                tmp_v.remove_task_of_vehicle(vi)
            for uav in self.UAVs:
                uav.remove_task_of_vehicle(vi)
            for bs in self.BSs:
                bs.remove_task_of_vehicle(vi)
        self.remove_vehicle_channel(v_id)

    def update_large_fading(self, veh_positions, uav_positions):
        # 涉及到车辆和无人机的channel需要更新位置
        self.V2IChannel.update_positions(veh_positions)
        self.V2VChannel.update_positions(veh_positions)
        self.U2IChannel.update_positions(uav_positions)
        self.U2UChannel.update_positions(uav_positions)
        self.V2UChannel.update_positions(veh_positions, uav_positions)
        
        # 更新path loss
        self.V2IChannel.update_pathloss()
        self.V2VChannel.update_pathloss()
        self.U2IChannel.update_pathloss()
        self.U2UChannel.update_pathloss()
        self.V2UChannel.update_pathloss()
        self.I2IChannel.update_pathloss()
        # 计算距离差，根据self.vid_index的index数值排序
        veh_delta_distance = self.time_step * np.asarray([c.velocity for c in sorted(self.vehicles.values(), key=lambda x: self.vid_index[x.id])])
        uav_delta_distance = self.time_step * np.asarray([c.velocity for c in self.UAVs]) # uav的顺序不用管
        # 更新阴影
        self.V2IChannel.update_shadow(veh_delta_distance)
        self.V2VChannel.update_shadow(veh_delta_distance)
        self.U2IChannel.update_shadow(uav_delta_distance)
        self.U2UChannel.update_shadow(uav_delta_distance)
        self.V2UChannel.update_shadow(veh_delta_distance, uav_delta_distance)
        self.I2IChannel.update_shadow()

    def update_small_fading(self):
        self.V2IChannel.update_fast_fading()
        self.V2VChannel.update_fast_fading()
        self.U2IChannel.update_fast_fading()
        self.U2UChannel.update_fast_fading()
        self.V2UChannel.update_fast_fading()
        # self.I2IChannel.update_fast_fading()
        
    def renew_channel(self):
        # ====================================================================================
        # This function updates all the channels including V2V, V2I, V2U, U2U, U2I channels
        # ====================================================================================
        # veh_positions根据self.vid_index的index数值排序
        veh_positions = [c.position for c in sorted(self.vehicles.values(), key=lambda x: self.vid_index[x.id])]
        uav_positions = [c.position for c in self.UAVs]
        self.update_large_fading(veh_positions, uav_positions)
        self.V2VChannel_abs = self.V2VChannel.PathLoss #+ self.V2VChannel.Shadow
        self.V2IChannel_abs = self.V2IChannel.PathLoss #+ self.V2IChannel.Shadow
        self.V2UChannel_abs = self.V2UChannel.PathLoss #+ self.V2UChannel.Shadow
        self.U2UChannel_abs = self.U2UChannel.PathLoss #+ self.U2UChannel.Shadow
        self.U2IChannel_abs = self.U2IChannel.PathLoss #+ self.U2IChannel.Shadow
        self.I2IChannel_abs = self.I2IChannel.PathLoss #+ self.I2IChannel.Shadow
    @property
    def U2VChannel_with_fastfading(self):
        return self.V2UChannel_with_fastfading.transpose(1,0,2)
    @property
    def I2UChannel_with_fastfading(self):
        return self.U2IChannel_with_fastfading.transpose(1,0,2)
    @property
    def I2VChannel_with_fastfading(self):
        return self.V2IChannel_with_fastfading.transpose(1,0,2)
    def renew_channels_fastfading(self):   
        # ====================================================================================
        # This function updates all the channels including V2V, V2I, V2U, U2U, U2I channels
        # ====================================================================================
        if self.n_Veh == 0:
            return
        self.renew_channel() # large fadings
        self.update_small_fading()
        # 为什么要减去快速衰落?
        V2VChannel_with_fastfading = np.repeat(self.V2VChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2VChannel_with_fastfading = V2VChannel_with_fastfading #- self.V2VChannel.FastFading
        V2IChannel_with_fastfading = np.repeat(self.V2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2IChannel_with_fastfading = V2IChannel_with_fastfading #- self.V2IChannel.FastFading
        V2UChannel_with_fastfading = np.repeat(self.V2UChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2UChannel_with_fastfading = V2UChannel_with_fastfading #- self.V2UChannel.FastFading
        U2UChannel_with_fastfading = np.repeat(self.U2UChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2UChannel_with_fastfading = U2UChannel_with_fastfading #- self.U2UChannel.FastFading
        U2IChannel_with_fastfading = np.repeat(self.U2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2IChannel_with_fastfading = U2IChannel_with_fastfading #- self.U2IChannel.FastFading
        I2IChannel_with_fastfading = np.repeat(self.I2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.I2IChannel_with_fastfading = I2IChannel_with_fastfading
    
    def Compute_Rate(self):
        '''按照资源块(频段)划分带宽资源，记录每一个RB上面的传输速率，正式传输的时候把所有的资源块数据相加'''
        avg_band = self.RB_bandwidth
        self.Compute_Rate_without_Bandwidth()
        self.V2V_Rate = avg_band * self.V2V_Rate
        self.V2U_Rate = avg_band * self.V2U_Rate
        self.V2I_Rate = avg_band * self.V2I_Rate
        self.U2U_Rate = avg_band * self.U2U_Rate
        self.U2V_Rate = avg_band * self.U2V_Rate
        self.U2I_Rate = avg_band * self.U2I_Rate
        self.I2U_Rate = avg_band * self.I2U_Rate
        self.I2V_Rate = avg_band * self.I2V_Rate
        self.I2I_Rate = avg_band * self.I2I_Rate

    def get_vehicle_by_index(self, index):
        # 通过vid_index获取vid，然后通过vid获取vehicle
        # vid_index是一个字典，key是vid，value是index
        vid = list(self.vid_index.keys())[list(self.vid_index.values()).index(index)]
        return self.vehicles[vid]

    def get_vid_by_index(self, index):
        vid = self.index_vid[index]
        return vid

    def Compute_Rate_without_Bandwidth(self):
        '''V2V, V2I, V2U, U2U, U2I的传输速率。各种干扰都要考虑, action是针对neighbor进行的二维数组，RB序号'''
        # 干扰是接收端收到的功率，单位mW
        X2I_Interference = np.zeros((self.n_BS, self.n_RB))
        X2V_Interference = np.zeros((self.n_Veh, self.n_RB))
        X2U_Interference = np.zeros((self.n_UAV, self.n_RB))
        V2V_Signal = np.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        V2U_Signal = np.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        V2I_Signal = np.zeros((self.n_Veh, self.n_BS, self.n_RB))
        U2U_Signal = np.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        U2V_Signal = np.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        U2I_Signal = np.zeros((self.n_UAV, self.n_BS, self.n_RB))
        I2U_Signal = np.zeros((self.n_BS, self.n_UAV, self.n_RB))
        I2V_Signal = np.zeros((self.n_BS, self.n_Veh, self.n_RB))
        I2I_Signal = np.zeros((self.n_BS, self.n_BS, self.n_RB))
        interference_power_matrix_vtx_x2i = np.zeros((self.n_Veh, self.n_BS, self.n_RB))
        interference_power_matrix_vtx_x2v = np.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        interference_power_matrix_vtx_x2u = np.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        interference_power_matrix_utx_x2i = np.zeros((self.n_UAV, self.n_BS, self.n_RB))
        interference_power_matrix_utx_x2v = np.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        interference_power_matrix_utx_x2u = np.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        interference_power_matrix_itx_x2i = np.zeros((self.n_BS, self.n_BS, self.n_RB))
        interference_power_matrix_itx_x2v = np.zeros((self.n_BS, self.n_Veh, self.n_RB))
        interference_power_matrix_itx_x2u = np.zeros((self.n_BS, self.n_UAV, self.n_RB))
        # 1. 计算所有的signal, 如果信号源同时传输多个数据,信号强度叠加,当然,干扰也叠加
        # 遍历所有的车辆
        for task_idx, offloading_task_and_path in enumerate(self.offloading_tasks):
            '''{
                'task':self.to_offload_tasks[task_idx], 
                'path':offload_objs, # 指向任务卸载的对象
                'RBs':np.zeros(self.n_RB, dtype='bool'), #标识多少的RB被分配了，是一个0-1 vector
                'mode':None, # X2X 的连接
                'TX_idx':txidx,
                'RX_idx':rxidx,
                'Activated':False, # 当前的TTI是否被传输
            }'''
            activated = offloading_task_and_path['Activated']
            if not activated:
                continue
            mode = offloading_task_and_path['mode']
            txidx = offloading_task_and_path['TX_idx']
            rxidx = offloading_task_and_path['RX_idx']
            rb_nos = offloading_task_and_path['RBs'] # boolean vector
            power_db = None
            if mode == 'V2V':
                V2V_Signal[txidx, rxidx, :] += 10 ** ((self.V2V_power_dB - self.V2VChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2v[txidx, rxidx, :] -= 10 ** ((self.V2V_power_dB - self.V2VChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2V_power_dB
            elif mode == 'V2U':
                V2U_Signal[txidx, rxidx, :] += 10 ** ((self.V2U_power_dB - self.V2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2u[txidx, rxidx, :] -= 10 ** ((self.V2U_power_dB - self.V2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2U_power_dB
            elif mode == 'V2I':
                V2I_Signal[txidx, rxidx, :] += 10 ** ((self.V2I_power_dB - self.V2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2i[txidx, rxidx, :] -= 10 ** ((self.V2I_power_dB - self.V2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2I_power_dB
            elif mode == 'U2U':
                U2U_Signal[txidx, rxidx, :] += 10 ** ((self.U2U_power_dB - self.U2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2u[txidx, rxidx, :] -= 10 ** ((self.U2U_power_dB - self.U2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.U2U_power_dB
            elif mode == 'U2V':
                U2V_Signal[txidx, rxidx, :] += 10 ** ((self.U2V_power_dB - self.V2UChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2v[txidx, rxidx, :] -= 10 ** ((self.U2V_power_dB - self.V2UChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.U2V_power_dB
            elif mode == 'U2I':
                U2I_Signal[txidx, rxidx, :] += 10 ** ((self.U2I_power_dB - self.U2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2i[txidx, rxidx, :] -= 10 ** ((self.U2I_power_dB - self.U2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.U2I_power_dB
            elif mode == 'I2U': # channel有对称性，所以直接用现有的channel就行了
                I2U_Signal[txidx, rxidx, :] += 10 ** ((self.I2U_power_dB - self.U2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2u[txidx, rxidx, :] -= 10 ** ((self.I2U_power_dB - self.U2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.I2U_power_dB
            elif mode == 'I2V':
                I2V_Signal[txidx, rxidx, :] += 10 ** ((self.I2V_power_dB - self.V2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2v[txidx, rxidx, :] -= 10 ** ((self.I2V_power_dB - self.V2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.I2V_power_dB
            elif mode == 'I2I':
                I2I_Signal[txidx, rxidx, :] += 10 ** ((self.I2I_power_dB - self.I2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2i[txidx, rxidx, :] -= 10 ** ((self.I2I_power_dB - self.I2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.I2I_power_dB
            # power_db = power_db * np.ones((1, 1, 1))
            if mode[0] == 'V':
                interference_power_matrix_vtx_x2i[txidx, :, :] += 10 ** ((power_db - self.V2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2VChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2u[txidx, :, :] += 10 ** ((power_db - self.V2UChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
            elif mode[0] == 'U':
                interference_power_matrix_utx_x2i[txidx, :, :] += 10 ** ((power_db - self.U2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                # tmp = np.transpose(10 ** ((power_db - self.V2UChannel_with_fastfading[:, txidx, :]) / 10), (1, 0, 2))
                interference_power_matrix_utx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2UChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2u[txidx, :, :] += 10 ** ((power_db - self.U2UChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
            elif mode[0] == 'I':
                interference_power_matrix_itx_x2i[txidx, :, :] += 10 ** ((power_db - self.I2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2IChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2u[txidx, :, :] += 10 ** ((power_db - self.U2IChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos

        # 2. 分别计算每个链路对X2I, X2U, X2V的干扰，同一个RB的情况下
        # 2.1 X2I Interference
        interference_v2x_x2i = np.sum(interference_power_matrix_vtx_x2i, axis = 0) # 车辆作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_u2x_x2i = np.sum(interference_power_matrix_utx_x2i, axis = 0) # 无人机作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_i2x_x2i = np.sum(interference_power_matrix_itx_x2i, axis = 0)
        X2I_Interference = interference_v2x_x2i + interference_u2x_x2i + interference_i2x_x2i

        # 2.2 X2V Interference
        interference_v2x_x2v = np.sum(interference_power_matrix_vtx_x2v, axis = 0) # 车辆作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_u2x_x2v = np.sum(interference_power_matrix_utx_x2v, axis = 0) # 无人机作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_i2x_x2v = np.sum(interference_power_matrix_itx_x2v, axis = 0)
        X2V_Interference = interference_v2x_x2v + interference_u2x_x2v + interference_i2x_x2v

        # 2.3 X2U Interference
        interference_v2x_x2u = np.sum(interference_power_matrix_vtx_x2u, axis = 0) # 车辆作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_u2x_x2u = np.sum(interference_power_matrix_utx_x2u, axis = 0) # 无人机作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_i2x_x2u = np.sum(interference_power_matrix_itx_x2u, axis = 0)
        X2U_Interference = interference_v2x_x2u + interference_u2x_x2u + interference_i2x_x2u

        # 3. 最后再计算rate
        # 对于每一个车辆V2I的干扰，如果他自身进行了传输，那么干扰就减去自身的传输功率
        V2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        V2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        V2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        self.V2V_Interference = V2V_Interference + self.sig2
        self.V2U_Interference = V2U_Interference + self.sig2
        self.V2I_Interference = V2I_Interference + self.sig2
        self.V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference)) # bps, 小b
        self.V2I_Rate = np.log2(1 + np.divide(V2I_Signal, self.V2I_Interference))
        self.V2U_Rate = np.log2(1 + np.divide(V2U_Signal, self.V2U_Interference))
        
        U2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        U2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        U2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        self.U2U_Interference = U2U_Interference + self.sig2
        self.U2V_Interference = U2V_Interference + self.sig2
        self.U2I_Interference = U2I_Interference + self.sig2
        self.U2U_Rate = np.log2(1 + np.divide(U2U_Signal, self.U2U_Interference))
        self.U2V_Rate = np.log2(1 + np.divide(U2V_Signal, self.U2V_Interference))
        self.U2I_Rate = np.log2(1 + np.divide(U2I_Signal, self.U2I_Interference))

        I2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_BS, axis = 0)
        I2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_BS, axis = 0)
        I2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_BS, axis = 0)
        self.I2U_Interference = I2U_Interference + self.sig2
        self.I2V_Interference = I2V_Interference + self.sig2
        self.I2I_Interference = I2I_Interference + self.sig2
        self.I2U_Rate = np.log2(1 + np.divide(I2U_Signal, self.I2U_Interference))
        self.I2V_Rate = np.log2(1 + np.divide(I2V_Signal, self.I2V_Interference))
        self.I2I_Rate = np.log2(1 + np.divide(I2I_Signal, self.I2I_Interference))

    def generate_tasks(self):
        '''根据泊松分布生成计算任务,每一个time_step执行一次，并且生成的任务在[cur_time, cur_time+time_step]之间出现'''
        def between(a,b):
            assert a <= b
            return a + np.random.rand() * (b-a)
        v_by_idx = list(self.task_vehicles.values())
        for idx, vi in enumerate(v_by_idx):
            if vi.serving:
                continue
            task_profile = (between(self.task_cpu_min, self.task_cpu_max), between(self.task_data_min, self.task_data_max), between(self.task_ddl_min, self.task_ddl_max))
            vi.task_profile = task_profile
            # time_slot_task = np.random.poisson(vi.task_lambda * self.TTI, size=int(self.time_step/self.TTI))
            # uniform random choice according to the task_lambda * self.TTI
            time_slot_task = np.random.choice([0,1], size=int(self.time_step/self.TTI), p=[1-vi.task_lambda * self.TTI, vi.task_lambda * self.TTI])
            total_task_cnt = sum(time_slot_task)
            for tti_slot in range(len(time_slot_task)):
                if time_slot_task[tti_slot] <= 0: continue
                # while total_task_cnt > 0:
                tmp_task = Task(self.task_id_cnt, vi, tti_slot*self.TTI+self.cur_time, task_profile[0], task_profile[1], task_profile[2])
                self.task_id_cnt += 1
                self.to_offload_tasks.append(tmp_task)
                total_task_cnt -= 1
                # break
        for uav in self.UAVs:
            # uav生成任务，不考虑是不是Serving
            task_profile = (between(self.task_cpu_min, self.task_cpu_max), between(self.task_data_min, self.task_data_max), between(self.task_ddl_min, self.task_ddl_max))
            uav.task_profile = task_profile
            time_slot_task = np.random.choice([0,1], size=int(self.time_step/self.TTI), p=[1-uav.task_lambda * self.TTI, uav.task_lambda * self.TTI])
            total_task_cnt = sum(time_slot_task)
            for tti_slot in range(len(time_slot_task)):
                if time_slot_task[tti_slot] <= 0: continue
                # while total_task_cnt>0:
                tmp_task = Task(self.task_id_cnt, uav, tti_slot*self.TTI+self.cur_time, task_profile[0], task_profile[1], task_profile[2])
                self.task_id_cnt += 1
                self.to_offload_tasks.append(tmp_task)
                total_task_cnt -= 1
                # break

    def Offload_Tasks(self, task_path_dict_list):
        '''只offload task.start_time == cur_time的任务，返回还未offload的任务
        输入的是字典，表示
        task_path_dict_list = [{
            'task': 当前等待卸载任务中的任务
            'offload_path':
                [{
                    'X_idx':设备idx，-1表示云计算，其他数字分别表示交给RSU,UAV,Veh计算,
                    'X_type':设备种类,'UAV','cloud','RSU','Veh'
                },{
                    'X_idx':设备idx,
                    'X_type':设备种类,'UAV','cloud','RSU','Veh'
                }]卸载的路径，可能多跳，第0跳是任务的发布车辆
        },{
            ...
        }]
        '''
        # 1. 判断是否有待传输的任务
        if len(self.to_offload_tasks)==0 or len(task_path_dict_list) == 0:
            return []
        # 2. 如果有，则遍历task_paths_dict进行决策，并且加入到offloading_tasks
        tmp_offload_task_and_path_list = []
        remained_task_path_dict_list = []
        for task_path_dict in task_path_dict_list:
            task_type = task_path_dict['task_type']
            task = task_path_dict['task']
            if task.start_time > self.cur_time:
                remained_task_path_dict_list.append(task_path_dict)
                continue # 还没到时候，下一个
            if task_type == 'verification':
                task = Task(task.id, task.g_veh, task.start_time, task.ini_cpu, task.ini_data_size, task_path_dict['ddl'], task_type=Task.VERIFY_TYPE)
                self.to_offload_tasks.append(task)
            assert task in self.to_offload_tasks
            offload_path = task_path_dict['offload_path']
            offload_objs = []
            for tmp_opath in offload_path:
                X_device = tmp_opath['X_device']
                offload_objs.append(X_device)
            RX_device = offload_objs[0]
            TX_type = 'V'
            RX_type = 'V' if isinstance(RX_device, Vehicle) else 'U' if isinstance(RX_device, UAV) else 'I'
            tmp_offload_task_and_path_list.append({
                'task':task, 
                'path':offload_objs, # 指向任务卸载的对象
                'RBs':np.zeros(self.n_RB, dtype='bool'), #标识多少的RB被分配了，是一个0-1 vector
                'mode':f'{TX_type}2{RX_type}', # X2X 的连接
                'Activated':False, # 当前的TTI是否被传输
                'TX_device':None,
                'RX_device':None
            })
            task.g_veh.served_last_period = True
            task.last_transmit_time = self.cur_time # 这是为了防止等待卸载的时间也算作等待传输时间
            
        for task_and_path in tmp_offload_task_and_path_list:
            self.to_offload_tasks.remove(task_and_path['task'])
            self.offloading_tasks.append(task_and_path)
        return remained_task_path_dict_list

    def Communication_RB_Allocation(self, activated_offloading_tasks_with_RB_Nos:np.array):
        '''
        判断offloading_tasks里面的任务，哪些是需要进行的？然后根据当前任务所在的位置和下一跳的位置，activate X2X links，并且分配对应的RB
        activated_offloading_tasks_with_RB_Nos必须是一个二维矩阵，大小为(len(self.offloading_tasks), n_RB)
        '''
        if len(self.offloading_tasks) == 0:
            return
        assert activated_offloading_tasks_with_RB_Nos.shape[0] == len(self.offloading_tasks) and activated_offloading_tasks_with_RB_Nos.shape[1] == self.n_RB
     
        # 初始化
        activated_offloading_tasks_with_RB_Nos = np.array(activated_offloading_tasks_with_RB_Nos, dtype='bool')
        self.V2V_active_links = np.zeros((self.n_Veh, self.n_Veh), dtype='bool')
        self.V2I_active_links = np.zeros((self.n_Veh, self.n_BS), dtype='bool')
        self.V2U_active_links = np.zeros((self.n_Veh, self.n_UAV), dtype='bool')
        self.U2U_active_links = np.zeros((self.n_UAV, self.n_UAV), dtype='bool')
        self.U2V_active_links = np.zeros((self.n_UAV, self.n_Veh), dtype='bool')
        self.U2I_active_links = np.zeros((self.n_UAV, self.n_BS), dtype='bool')
        self.I2U_active_links = np.zeros((self.n_BS, self.n_UAV), dtype='bool')
        self.I2V_active_links = np.zeros((self.n_BS, self.n_Veh), dtype='bool')
        self.I2I_active_links = np.zeros((self.n_BS, self.n_BS), dtype='bool')
        # 遍历激活连接
        for task_idx, offloading_task_and_path in enumerate(self.offloading_tasks):
            task = offloading_task_and_path['task']
            assert isinstance(task, Task)
            path = offloading_task_and_path['path']
            assert len(path)>0 
            allocated_RBs = activated_offloading_tasks_with_RB_Nos[task_idx, :]
            offloading_task_and_path['RBs'] = allocated_RBs
            offloading_task_and_path['Activated'] = np.any(allocated_RBs)
            to_offload_flag = True
            if task.start_time > self.cur_time:
                print(f'[{self.cur_time}]: task {task_idx} 的开始时间是 {task.start_time}')
                to_offload_flag = False
            # if task.routing[-1].transmitting == True:
            #     print(f'TX Device of task {task_idx} 正在传输中，分配RB失败！')
            #     to_offload_flag = False
            if task.routing[-1].receiving == True and self.half_duplex:
                print(f'TX Device of task {task_idx} 正在接收中，半双工场景，分配RB失败！')
                to_offload_flag = False
            # if path[0].receiving == True:
            #     print(f'RX Device of task {task_idx} 正在接收中，分配RB失败！')
            #     to_offload_flag = False
            if path[0].transmitting == True and self.half_duplex:
                print(f'RX Device of task {task_idx} 正在传输中，半双工场景，分配RB失败！')
                to_offload_flag = False
            offloading_task_and_path['Activated'] = to_offload_flag and offloading_task_and_path['Activated']
            if not offloading_task_and_path['Activated']:
                continue
            TX_device = task.routing[-1]
            RX_device = path[0]
            TX_device.transmitting = True
            RX_device.receiving = True
            cur_device = 'V'
            cur_device_idx = -1
            if isinstance(TX_device, BS):
                cur_device='I'
                cur_device_idx = self.BSs.index(TX_device)
            elif isinstance(TX_device, UAV):
                cur_device='U'
                cur_device_idx = self.UAVs.index(TX_device)
            elif isinstance(TX_device, Vehicle):
                cur_device = 'V'
                cur_device_idx = self.vehicle_by_index.index(TX_device)
            else:
                raise RuntimeError('type error')
            tar_device = 'V'
            tar_device_idx = -1
            if isinstance(RX_device, BS):
                tar_device = 'I'
                tar_device_idx = self.BSs.index(RX_device)
            elif isinstance(RX_device, UAV):
                tar_device = 'U'
                tar_device_idx = self.UAVs.index(RX_device)
            elif isinstance(RX_device, Vehicle):
                tar_device = 'V'
                tar_device_idx = self.vehicle_by_index.index(RX_device)
            else:
                raise RuntimeError('type error')
            offloading_task_and_path['mode'] = f'{cur_device}2{tar_device}'
            offloading_task_and_path['TX_device'] = TX_device
            offloading_task_and_path['RX_device'] = RX_device
            offloading_task_and_path['TX_idx'] = cur_device_idx
            offloading_task_and_path['RX_idx'] = tar_device_idx
            if offloading_task_and_path['mode'] == 'V2V':
                self.V2V_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'V2U':
                self.V2U_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'V2I':
                self.V2I_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'U2U':
                self.U2U_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'U2V':
                self.U2V_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'U2I':
                self.U2I_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'I2U':
                self.I2U_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'I2V':
                self.I2V_active_links[cur_device_idx, tar_device_idx] = True
            elif offloading_task_and_path['mode'] == 'I2I':
                self.I2I_active_links[cur_device_idx, tar_device_idx] = True

    def Execute_Communicate(self):
        ''' 基于offloading_tasks进行任务卸载，需要注意的是，传输完要把判断哪些传输失败？以及，每个节点的transmitting和receiving都要是False
        '''
        tmp_succeed_tasks = []
        tmp_failed_tasks = []
        for task_idx, offloading_task_and_path in enumerate(self.offloading_tasks):
            task = offloading_task_and_path['task'] # 需要判断当前传输的数据大小
            mode = offloading_task_and_path['mode']
            offload_objs = offloading_task_and_path['path']
            assert isinstance(task, Task)
            TX_device = task.routing[-1]
            X_device = offload_objs[0]
            TX_device.transmitting = False
            X_device.receiving = False # 务必每一回都要让这个玩意儿变成负数，然后每个TTI重新分配RB资源
            if task.wait_to_ddl(self.cur_time):
                tmp_failed_tasks.append((offloading_task_and_path, Task.OUT_OF_DDL))
                continue
            if not offloading_task_and_path['Activated']:
                # 判断跳过多少次，如果超过阈值，tmp_failed_tasks
                # if task.fail_to_transmit(self.cur_time, self.max_wait_transmit_threshold):
                #     tmp_failed_tasks.append((offloading_task_and_path, Task.OUT_OF_TTI))
                continue
            TX_idx = offloading_task_and_path['TX_idx']
            RX_idx = offloading_task_and_path['RX_idx']
            trans_data = 0
            if mode == 'V2V':
                trans_data += np.sum(self.V2V_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.vehicle_by_index[RX_idx]
            elif mode == 'V2I':
                trans_data += np.sum(self.V2I_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.BSs[RX_idx]
            elif mode == 'V2U':
                trans_data += np.sum(self.V2U_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.UAVs[RX_idx]
            elif mode == 'U2U':
                trans_data += np.sum(self.U2U_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.UAVs[RX_idx]
            elif mode == 'U2V':
                trans_data += np.sum(self.U2V_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.vehicle_by_index[RX_idx]
            elif mode == 'U2I':
                trans_data += np.sum(self.U2I_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.BSs[RX_idx]
            elif mode == 'I2U':
                trans_data += np.sum(self.I2U_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.UAVs[RX_idx]
            elif mode == 'I2V':
                trans_data += np.sum(self.I2V_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.vehicle_by_index[RX_idx]
            elif mode == 'I2I':
                trans_data += np.sum(self.I2I_Rate[TX_idx, RX_idx, :]) * self.TTI
                X_device = self.BSs[RX_idx]
            assert X_device == offload_objs[0]
            # 如果在传输过程中经过的车辆不见了，那么offloading task就算失败，这个在deactivate vehicle里面执行
            flag = task.transmit_to_X(X_device, trans_data, self.cur_time)
            if flag: # 表示本次传输成功，route需要调整
                del offload_objs[0]
            if len(offload_objs) == 0: # 完成所有的传输
                tmp_succeed_tasks.append(offloading_task_and_path)
            offloading_task_and_path['path'] = offload_objs # 以防万一覆盖
        for offloading_task_and_path in tmp_succeed_tasks:
            self.offloading_tasks.remove(offloading_task_and_path)
            offloading_task_and_path['RX_device'].append_task(offloading_task_and_path['task'])
        for offloading_task_and_path, reason in tmp_failed_tasks:
            self.offloading_tasks.remove(offloading_task_and_path)
            self.failed_tasks.append(
                {
                    'task':offloading_task_and_path['task'],
                    'reason':reason
                }
            )
    def Check_To_X_Tasks(self):
        '''
        检查to_offload, to_verify, to_pay的任务
        '''
        for task in self.to_offload_tasks.copy():
            self.to_offload_tasks.remove(task)
                
    def Execute_Compute(self, cpu_allocation_for_fog_nodes):
        '''进行计算任务，并且，每个veh检查自身的task queue，如果完成计算，则表示succeed，超时则failed'''
        tmp_all_finished_tasks = []
        tmp_is_reliable_list = []
        for info_dict in cpu_allocation_for_fog_nodes:
            is_to_cheat = info_dict['is_to_cheat']
            CPU_allocation = info_dict['CPU_allocation']
            device = info_dict['device']
            finished_tasks, is_reliable_list = device.calculate_task(self.TTI, CPU_allocation, is_to_cheat)
            tmp_all_finished_tasks.extend(finished_tasks)
            tmp_is_reliable_list.extend(is_reliable_list)
        # 检查所有的任务，是否超过了ddl
        for idx, task in enumerate(tmp_all_finished_tasks):
            if task.exceed_ddl(self.cur_time):
                self.failed_tasks.append({
                    'task':task,
                    'reason':Task.OUT_OF_DDL
                })
            else:
                task.succeed_compute(self.cur_time, tmp_is_reliable_list[idx])
                if task.task_type == Task.CALCULATE_TYPE:
                    self.succeed_tasks[task.id] = task # 由于之后会经常用到，所以直接存储dict
                elif task.task_type == Task.VERIFY_TYPE:
                    self.verified_tasks.append(task)
                # to_pay_tasks里面添加task，并且直到ddl会被移除
                # 此外，每一轮会algorithm得出一个dict，key是task id，value是一个list，里面是task，用来表示to_pay_list里面哪些是被取消掉的
                # if task.id not in self.to_pay_tasks:
                #     self.to_pay_tasks[task.id] = []
                # self.to_pay_tasks[task.id].append(task)
                
    def Mine_Blockchain(self, miner_and_revenues, validated = False):
        '''挖矿，指定对象（目前只有RSU），指定收益
        检查更新blockchain
        每个对象：
        r={
            'X_device':selected_bs,
            'consensus':'PoS',
            'target_block':block,
            'stake':stake, # 节点消耗的能量
            'revenue':revenue,
            'cheated':cheated
            }
        '''
        for info_dict in miner_and_revenues:
            selected_bs = info_dict['X_device']
            block = info_dict['target_block']
            stake = info_dict['stake']
            revenue = info_dict['revenue']
            cheated = info_dict['cheated']
            assert selected_bs in self.BSs, '当前的miner不是BS'
            selected_bs.miner_mine_block(block, stake, revenue, cheated)
            self.blockchain.add_block(block, selected_bs)

        # 检查是否有新的block
        self.blockchain.generate_to_mine_blocks(self.cur_time)
        if validated:
            validate_chain = self.blockchain.validate_chain()
            # 这里可以加入一些验证的信息和输出
            print('验证结果：', validate_chain)

        # 更新bs的stake
        for bs in self.BSs:
            bs.update_stake(self.TTI)

    def Pay_Revenues_And_Punish(self, evicted_task_dict, revenue_and_punishment):
        '''根据evicted_task_dict清空self.to_pay_tasks里面的task，然后根据revenue_and_punishment给出的信息，给出收益和惩罚
        最后记录在self.blockchain中,add_new_transaction
        '''
        # 在to_pay_tasks里面清空指定的task
        for info_dict in evicted_task_dict:
            task_id = info_dict['task_id']
            evicted_tasks = info_dict['evicted_tasks']
            for task in evicted_tasks:
                self.to_pay_tasks[task_id].remove(task)
            if len(self.to_pay_tasks[task_id]) == 0:
                del self.to_pay_tasks[task_id]

        # 根据revenue_and_punishment给出的信息，给出收益和惩罚
        for info_dict in revenue_and_punishment:
            device = info_dict['device']
            amount = info_dict['amount']
            relevant_task = info_dict['relevant_task']
            device.update_revenue(amount)
            self.blockchain.add_new_transaction((device, relevant_task, amount))

    def Update_Reputation_Score(self, reputation_score_dict):
        '''[
            {
                'device': env.vehicle_by_index[0],
                'current_reputation': 0.5
            }
        ]
        '''
        for info_dict in reputation_score_dict:
            device = info_dict['device']
            current_reputation = info_dict['current_reputation']
            device.set_reputation_score(current_reputation)


    def State_Info(self):
        '''获取当前的状态信息，可以由使用者自行开发'''
        pass

    def Compute_Statistics(self):
        '''计算并获取当前的统计信息，可以由使用者自行开发'''
        success_num = len(self.succeed_tasks)
        fail_num = len(self.failed_tasks)
        completion_ratio = 0 if success_num == 0 else success_num / (success_num + fail_num)
        avg_latency = 0
        for task in list(self.succeed_tasks.values()) + self.failed_tasks:
            avg_latency += task.service_delay
        avg_latency = 0 if success_num == 0 else avg_latency / (success_num + fail_num)
        return avg_latency, completion_ratio
    
    def Mobility_Module(self, vehicle_ids, row_data_dict, uav_directions, uav_speeds):
        '''移动性模块，更新UAV位置，更新车辆位置，更新邻居关系'''
        self.renew_veh_positions(vehicle_ids, row_data_dict)
        self.renew_uav_positions(uav_directions, uav_speeds)
    
    def Task_Generation_Module(self):
        '''任务生成模块，根据任务车辆自身的泊松分布生成任务'''
        self.generate_tasks()

    def FastFading_Module(self):
        '''根据位移信息，更新通信信道状态'''
        self.renew_channels_fastfading()

    def Time_Update_Module(self):
        self.cur_time += self.TTI
        self.cur_time = round(self.cur_time, 2)
        # 最好保证TTI能够通过整数次相加等价于一个time_step
        if self.last_time + self.time_step <= self.cur_time:
            self.cur_time = self.last_time + self.time_step
            self.last_time = self.cur_time
            return True
        return False

    # ---------------------------------------------------------------------------------------------------------------------------------------
    # 由使用者自行开发的部分

    def get_rewards(self, v2u_assigned, v2i_assigned):
        '''根据系统运行情况返回奖励，由使用者自己开发'''
        # 这里的v,u,i都是根据index编号的，假设0是没有assign，1是assign
        serve_u_task_cnt = np.zeros(v2u_assigned.shape[1])
        serve_i_task_cnt = np.zeros(v2i_assigned.shape[1])
        fail_u_task_cnt = np.zeros(v2u_assigned.shape[1])
        fail_i_task_cnt = np.zeros(v2i_assigned.shape[1])
        serve_u_qos = -np.ones(v2u_assigned.shape[1])
        serve_i_qos = -np.ones(v2i_assigned.shape[1])
        serve_u_res = np.ones(v2u_assigned.shape[1]) * self.uav_cpu
        serve_i_res = np.ones(v2i_assigned.shape[1]) * self.BS_cpu
        
        task_lambda_v = [veh.task_lambda for veh in self.vehicle_by_index]
        res_v = [veh.CPU_frequency for veh in self.vehicle_by_index]
        # 区域间频分，区域内时分
        avg_band = self.RB_bandwidth
        for vidx in range(v2u_assigned.shape[0]): 
            v2uidx = np.argmax(v2u_assigned[vidx])
            v2iidx = np.argmax(v2i_assigned[vidx])
            if v2u_assigned[vidx][v2uidx] == 1:
                self.vehicle_by_index[vidx].assigned_to = v2uidx
                scale = 1
                if self.v2u_distance[vidx][v2uidx] > 300:
                    scale = 0.2
                    # fail_u_task_cnt[v2uidx] += task_lambda_v[vidx]
                    # continue
                serve_u_task_cnt[v2uidx] += task_lambda_v[vidx]
                serve_u_res[v2uidx] += res_v[vidx]
                V2U_Signal = 10 ** ((self.V2U_power_dB - self.V2UChannel_with_fastfading[vidx, v2uidx, 0]) / 10)
                V2U_Rate = avg_band * np.log2(1 + np.divide(V2U_Signal, self.sig2 / scale))
                if task_lambda_v[vidx] > 0:
                    # V2U_Rate = avg_band * np.log2(1 + np.divide(V2U_Signal, V2U_Interference[vidx,v2uidx] - V2U_Signal))
                    # 加上等待的时间，直接假设是时分
                    serve_u_qos[v2uidx] += serve_u_qos[v2uidx] + task_lambda_v[vidx] * (self.task_data_max + self.task_data_min) / 2 / V2U_Rate
            elif v2i_assigned[vidx][v2iidx] == 1:
                self.vehicle_by_index[vidx].assigned_to = v2iidx + self.n_UAV
                scale = 1
                if self.v2i_distance[vidx][v2iidx] > 600:
                    scale = 0.2
                    # fail_i_task_cnt[v2iidx] += task_lambda_v[vidx]
                    # continue
                serve_i_task_cnt[v2iidx] += task_lambda_v[vidx]
                serve_i_res[v2iidx] += res_v[vidx]
                V2I_Signal = 10 ** ((self.V2I_power_dB - self.V2IChannel_with_fastfading[vidx, v2iidx, 0]) / 10)
                V2I_Rate = avg_band * np.log2(1 + np.divide(V2I_Signal, self.sig2 / scale))
                if task_lambda_v[vidx] > 0:
                    # V2I_Rate = avg_band * np.log2(1 + np.divide(V2I_Signal, V2I_Interference[vidx, v2uidx] - V2I_Signal))
                    serve_i_qos[v2iidx] += serve_i_qos[v2iidx] + task_lambda_v[vidx] * ((self.task_data_max + self.task_data_min) / 2 / V2I_Rate + (self.task_data_max + self.task_data_min) / 2 / self.BS2C_rate)
            else:
                print(1)
        # 基于接入的情况，计算reward，即qos。以1s为单位
        # 1. 区域内每个任务能够分配的计算资源，以任务为主体
        for idx in range(len(serve_u_qos)):
            if serve_u_task_cnt[idx] > 0:
                comp_time = (self.task_cpu_max+self.task_cpu_min) / 2 / (serve_u_res[idx] / serve_u_task_cnt[idx])
                serve_u_qos[idx] = serve_u_qos[idx] / serve_u_task_cnt[idx]
                serve_u_qos[idx] = 0 # 只记录计算的情况
                serve_u_qos[idx] += comp_time
                avg_ddl = (self.task_ddl_max+self.task_ddl_min) / 2
                serve_u_qos[idx] = avg_ddl/ serve_u_qos[idx]
        for idx in range(len(serve_i_qos)):
            if serve_i_task_cnt[idx] > 0:
                comp_time = (self.task_cpu_max+self.task_cpu_min) / 2 / (serve_i_res[idx] / serve_i_task_cnt[idx])
                serve_i_qos[idx] = serve_i_qos[idx] / serve_i_task_cnt[idx]
                serve_i_qos[idx] = 0 # 只记录计算的情况
                serve_i_qos[idx] += comp_time # 这里qos是时间
                avg_ddl = (self.task_ddl_max+self.task_ddl_min) / 2
                serve_i_qos[idx] = avg_ddl/ serve_i_qos[idx]
        tot_task = (serve_i_task_cnt.sum() + fail_i_task_cnt.sum() + serve_u_task_cnt.sum() + fail_u_task_cnt.sum()) + 1
        serve_u_qost = (serve_u_qos * serve_u_task_cnt - fail_u_task_cnt) / tot_task
        serve_i_qost = (serve_i_qos * serve_i_task_cnt - fail_i_task_cnt) / tot_task
        self.avg_qos = (np.mean(serve_i_qost) + np.mean(serve_u_qost)) / 2
        tot_qos = np.concatenate([serve_u_qost, serve_i_qost], axis = 0)
        flying_pen = np.zeros(self.n_UAV+self.n_BS)
        flying_pen[:self.n_UAV] = np.array([uav.velocity / 1000 for uav in self.UAVs])
        balance_omega = 1
        return 100 * ((balance_omega * tot_qos + (1 - balance_omega) * (self.avg_qos) * np.ones_like(tot_qos)))

    def update_UAV_Veh_direct_M(self):
        '''更新UAV和Veh之间的方向'''
        self.UAV_Veh_direct_M = np.zeros((self.n_UAV, self.n_Veh), dtype=np.int)
        self.RSU_Veh_direct_M = np.zeros((self.n_BS, self.n_Veh), dtype=np.int)
        for j, veh in enumerate(self.vehicle_by_index):
            veh_pos = veh.position
            for i, uav in enumerate(self.UAVs):
                uav_pos = uav.position
                self.UAV_Veh_direct_M[i][j] = self.get_direction(uav_pos, veh_pos)
            for i, bs in enumerate(self.BSs):
                bs_pos = bs.position
                self.RSU_Veh_direct_M[i][j] = self.get_direction(bs_pos, veh_pos)
    
    def get_direction(self, source_pos, target_pos):
        '''根据UAV和Veh的位置，获取方向'''
        x1, y1 = source_pos
        x2, y2 = target_pos
        dx = x2 - x1
        dy = y2 - y1
        # 0: stay, 1: right-up, 2:up, 3:left-up, 4:left, 5:left-down, 6:down, 7:right-down, 8:right
        # 获取角度,然后转换为0-8的数值,离散45度一个数字
        direction_degree = np.arctan2(dy, dx) / np.pi * 180
        if direction_degree < 0:
            direction_degree += 360
        direction_degree = int(direction_degree / 45)
        return direction_degree

    def get_graph_by_agent(self, agent_id, agent_type):
        '''根据UAV_id获取当前的图'''
        vehicle_list = self.vehicle_by_index
        if agent_type == 'UAV':
            agent = self.UAVs[agent_id]
            communication_range = self.args.UAV_communication_range
            direction_M = self.UAV_Veh_direct_M
            X2V_fading = self.V2UChannel_abs
        elif agent_type == 'RSU':
            agent_id -= self.n_UAV
            agent = self.BSs[agent_id]
            communication_range = self.args.RSU_communication_range
            direction_M = self.RSU_Veh_direct_M
            X2V_fading = self.V2IChannel_abs

        # Initialize node features and adjacency matrix
        node_features = []
        adjacency_matrix = np.zeros((len(vehicle_list), len(vehicle_list)))
        involved_vehicle_index = []
        for i, vehicle in enumerate(vehicle_list):
            # only vehicles within the communication range can be neighbors
            if self.distance(vehicle.position, agent.position) > communication_range:
                continue
            involved_vehicle_index.append(i)
            # Compute CPU request or resource
            cpu_request = vehicle.CPU_frequency if vehicle.serving else vehicle.task_lambda * ((self.task_cpu_max + self.task_cpu_min) / (self.task_ddl_max + self.task_ddl_min)) 

            # Compute one-hot direction vector
            direction = direction_M[agent_id][i]
            direction_one_hot = np.zeros(9)
            direction_one_hot[direction] = 1
            distance = self.distance(vehicle.position, agent.position)

            serve_or_not_one_hot = np.zeros(2)
            serve_or_not_one_hot[0] = 1 if vehicle.serving else 0
            serve_or_not_one_hot[1] = 1 if vehicle.task_lambda > 0 else 0
            assigned_to_one_hot = np.zeros(self.n_BS+self.n_UAV)
            if vehicle.assigned_to != -1:
                assigned_to_one_hot[vehicle.assigned_to] = 1

            # Add node features
            node_features.append(np.concatenate([[vehicle.velocity/30], serve_or_not_one_hot, [cpu_request, distance / communication_range, vehicle.position[0]/1000, vehicle.position[1]/1000], direction_one_hot, assigned_to_one_hot]))

            if not vehicle_list[i].serving:
                # Add edges to adjacency matrix
                for j in range(i, len(vehicle_list)):
                    # if not (vehicle_list[j] in vehicle_list[i].neighbor_vehicles):
                    #     continue
                    tot_fading = self.V2VChannel_abs[i][j]
                    tot_fading = max(50, tot_fading) # 50 is the minimum fading
                    # if tot_fading <= self.args.fading_threshold:
                    adjacency_matrix[j][i] = adjacency_matrix[i][j] = self.args.fading_threshold / tot_fading

        adjacency_matrix = adjacency_matrix[involved_vehicle_index][:, involved_vehicle_index]
        # 如果node_features为空，则添加一个agent作为一个节点
        # if len(node_features) == 0:
        # 添加agent作为一个节点，往node_features的第一个位置添加
        cpu_request = agent.CPU_frequency
        direction_one_hot = np.zeros(9)
        direction_one_hot[0] = 1
        assigned_to_one_hot = np.zeros(self.n_BS+self.n_UAV)
        assigned_to_one_hot[agent_id] = 1
        node_features.insert(0, np.concatenate([[agent.velocity/30, 1, 0, cpu_request, 0, agent.position[0]/1000, agent.position[1]/1000], direction_one_hot, assigned_to_one_hot]))
        adjacency_matrix = np.insert(adjacency_matrix, 0, 0, axis=0)
        adjacency_matrix = np.insert(adjacency_matrix, 0, 0, axis=1)
        for i, vehicle_id in enumerate(involved_vehicle_index):
            vehicle = vehicle_list[vehicle_id]
            if self.distance(vehicle.position, agent.position) > communication_range:
                continue
            tot_fading = X2V_fading[vehicle_id][agent_id]
            tot_fading = max(50, tot_fading)
            # if tot_fading <= self.args.fading_threshold:
            adjacency_matrix[0][i+1] = adjacency_matrix[i+1][0] = self.args.fading_threshold / tot_fading
        node_features = np.array(node_features)
        return node_features, adjacency_matrix, involved_vehicle_index

    def get_map_grid_vector(self, grid_width, grid_number, grid_x_range = [500, 2000], grid_y_range = [0, 1500]):
        #按照grid的排序，把车辆进行分类，然后按照[x, y, velocity,is_serve_or_not, CPU/task resource]来进行编码，一共是5*120个车辆=600维，不足的部分补0
        # grid_width表示每个网格的宽度，grid_number表示网格的数量
        # grid_x_range, grid_y_range表示网格的范围
        # feature_list = []
        # cnt = 0
        # for device in self.UAVs + self.BSs + self.vehicle_by_index:
        #     cnt += 1
        #     if cnt > 110:
        #         break
        #     x, y = device.position[0], device.position[1]
        #     x = int((x - grid_x_range[0]) // grid_width)
        #     y = int((y - grid_y_range[0]) // grid_width)
        #     # x = min(x, grid_number-1)
        #     # y = min(y, grid_number-1)
        #     # x = max(x, 0)
        #     # y = max(y, 0)
        #     cpu_or_request = device.CPU_frequency if device.serving else device.task_lambda * ((self.task_cpu_max + self.task_cpu_min) / (self.task_ddl_max + self.task_ddl_min))
        #     feature_list.append([x/1000, y/1000, device.velocity/30, 0 if device.serving else 1, cpu_or_request])
        # # 按照x, y排序
        # feature_list = sorted(feature_list, key=lambda x:(x[0], x[1]))
        # # 变成vector
        # feature_vector = np.array(feature_list).reshape(-1)
        # # 补0
        # feature_vector = np.concatenate([feature_vector, np.zeros((550 - len(feature_vector)))])
        # return feature_vector

        # 把地图分成grid_number*grid_number的网格，然后统计每个网格内的cpu资源,用正数表示资源,负数表示请求
        cpu_maps = np.zeros((grid_number, grid_number))
        for device in self.vehicle_by_index + self.UAVs + self.BSs:
            x, y = device.position[0], device.position[1]
            x = int((x - grid_x_range[0]) // grid_width)
            y = int((y - grid_y_range[0]) // grid_width)
            x = min(x, grid_number-1)
            y = min(y, grid_number-1)
            x = max(x, 0)
            y = max(y, 0)
            if device.serving:
                cpu_maps[x, y] += device.CPU_frequency 
            else:
                cpu_maps[x, y] -= device.task_lambda * ((self.task_cpu_max + self.task_cpu_min) / (self.task_ddl_max + self.task_ddl_min)) 
        cpu_maps /= 2.5
        cpu_map_vector = cpu_maps.reshape(-1)
        return cpu_map_vector
            