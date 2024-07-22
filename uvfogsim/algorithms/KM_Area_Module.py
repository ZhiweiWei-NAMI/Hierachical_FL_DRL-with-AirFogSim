# -*- encoding: utf-8 -*-
'''
@File    :   Cluster_Algorithm_Module.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''

from collections import deque
from typing import Set
from urllib import request
from .WHO_algorithm import WHO

import math
import numpy as np
from .Base_Algorithm_Module import Base_Algorithm_Module
from sklearn.cluster import KMeans
from ..environment import Environment
import warnings
import random
from ..task import Task

import matplotlib.pyplot as plt

def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
class KM_Area_Module(Base_Algorithm_Module):
    def __init__(self, env, args):
        super().__init__()
        warnings.filterwarnings("ignore")
        self.uav_weights = np.zeros(shape=(env.n_UAV))
        self.bs_weights = np.zeros(shape=(env.n_BS))
        self.max_weight = 1
        self.map_height = 40
        self.map_width = 40
        self.grid_length = 2000 // self.map_width # 10m一个格子
        self.request_nums = []
        self.env_cur_state = None
        self.env_next_state = None
        self.logged_computed_tasks = {}
        self.trans_per_second = []
        self.blockchain_length = 0
        self.last_succ_task_num = 0
        self.last_total_task_num = 0
        self.n_agent = env.n_UAV + env.n_BS
        self.env = env
        self.args = args
        from torch.utils.tensorboard import SummaryWriter
        import os
        import shutil
        writer_path = f"{args.tensorboard_writer_file}"
        if os.path.exists(writer_path):
            shutil.rmtree(writer_path)
        self.writer = SummaryWriter(writer_path)
        self.update_cnt = 0
        self.sum_rewards = deque(maxlen=200)
        self.avg_task_latency = []
        self.avg_task_ratio = []
        self.max_veh_in_region = 0
        self.last_cnt_succ_num = 0
        self.uav_rewards = []
        self.to_offload_task_ids = []


    def act_mining_and_pay(self, env:Environment):
        # RSU准备写到链上的交易信息
        blockchain = env.blockchain
        to_mine_block = blockchain.to_mine_blocks
        # 判断tran是否达到了区块大小，如果达到了，就进行挖矿，否则不挖矿
        result = []
        all_stakes = [bs.stake for bs in env.BSs]
        for block in to_mine_block:
            num=0
            stake=0
            cheated=False
            for bs in env.BSs:
                num+=bs.stake
            random_num = np.random.rand() * num
            num=0
            idx = 0
            for (idx,bs) in enumerate(env.BSs):
                stake = all_stakes[idx]
                num += stake
                if random_num<num:
                    selected_bs = bs
                    break
            selected_bs = env.BSs[idx]
            stake = all_stakes[idx]
            revenue = max(stake/10,1)
            all_stakes[idx] = 0
            r={
                'X_device':selected_bs,
                'consensus':'PoS',
                'target_block':block,
                'stake':stake, # 节点消耗的能量
                'revenue':revenue,
                'cheated':cheated
            }
            # 返回结果
            result.append(r)
        return result

    def act_pay_and_punish(self, env):
        to_pay_tasks = env.to_pay_tasks
        # to_pay_tasks = {'task_id':[若干个task]}
        # 根据每个任务的执行情况，给予奖励或者惩罚
        evicted_task_dict = []
        device_and_amount = []
        for task_id, task_list in to_pay_tasks.items():
            if task_id in self.logged_computed_tasks.keys():
                # 已经计算过了，不再计算，只需要进行验算相关的内容
                # 如果验算的信息都出现了，那么判断，是否需要原先的车辆进行惩罚
                pass
            evicted_tasks = []
            for task in task_list:
                gen_veh = task.g_veh
                comp_device = task.get_comp_device()
                ddl = task.ddl
                cpu = task.ini_cpu
                service_delay = task.service_delay
                task_type = task.task_type
                reputation_score = comp_device.reputation_score
                # 需要一些判断条件，满足条件再进行惩罚和奖励
                if task_type == Task.CALCULATE_TYPE:
                    device_and_amount.append({
                        'device':comp_device,
                        'amount':cpu*reputation_score,
                        'relevant_task':task
                    })
                    device_and_amount.append({
                        'device':gen_veh,
                        'amount':-cpu*reputation_score,
                        'relevant_task':task
                    })
                    self.logged_computed_tasks[task_id]={
                        'task':task,
                        'g_veh':gen_veh,
                        'comp_device':comp_device,
                        'revenue':cpu*reputation_score,
                    }
                    evicted_tasks.append(task) # 任务执行完毕，从任务列表中删除
                elif task_type == Task.VERIFY_TYPE:
                    pass
            evicted_task_dict.append({
                'task_id':task_id,
                'evicted_tasks':evicted_tasks
            })
        return evicted_task_dict, device_and_amount
    def collect_data(self, env):
        self.trans_per_second.append(env.blockchain.total_transaction_num - self.blockchain_length)
        self.blockchain_length = env.blockchain.total_transaction_num
        
    def print_result(self, env):
        self.update_cnt += 1
        succ_task_cnt = self.env.get_succeed_task_cnt()
        failed_task_cnt = self.env.get_failed_task_cnt()
        tot_task_cnt = succ_task_cnt + failed_task_cnt
        # succ_task_cnt = succ_task_cnt - self.last_succ_task_num
        # tot_task_cnt = tot_task_cnt - self.last_total_task_num
        # self.last_succ_task_num = self.env.get_succeed_task_cnt()
        self.last_total_task_num = self.env.get_succeed_task_cnt() + self.env.get_failed_task_cnt()
        avg_task_latency = self.env.get_avg_task_latency()
        avg_task_ratio = succ_task_cnt / max(1, tot_task_cnt)
        self.avg_task_ratio.append(avg_task_ratio)
        self.avg_task_latency.append(avg_task_latency)
        endings = '\r'
        if self.update_cnt % 200 == 0:
            endings = '\n'
        print("Step {}, ratio: {:.3f}, latency: {:.3f}, reward: {:.3f}, relevance: {:.3f}, nTV: {:d}, UAV_rew: {:.3f}".format(
            self.update_cnt, 
            avg_task_ratio, 
            avg_task_latency, 
            np.mean(self.sum_rewards),
            avg_task_ratio*100 / max(1, self.sum_rewards[-1]),
            self.env.n_TV,
            np.mean(self.uav_rewards)
        ), end=endings)
        if self.update_cnt % (self.args.update_every) == 0:
            self.writer.add_scalar('Metric/ratio', avg_task_ratio, self.update_cnt // self.args.update_every)
            self.writer.add_scalar('Metric/latency', avg_task_latency, self.update_cnt // self.args.update_every)
            self.writer.add_scalar('Metric/UAV_reward', np.mean(self.sum_rewards), self.update_cnt // self.args.update_every)
            self.writer.add_scalar('Metric/total_reward', np.mean(self.uav_rewards), self.update_cnt // self.args.update_every)
        if len(self.sum_rewards) > 200 and False:
            # 绘制图像，x为step, y为sum_rewards，以及另2个y为self.avg_task_ratio和self.avg_task_latency
            plt.figure()
            # 绘制三条曲线，全部归一化
            # sum_rewards是每个步长的总和，因此需要除以步长数量
            self.sum_rewards = np.array(self.sum_rewards) / np.arange(1, len(self.sum_rewards)+1)
            self.sum_rewards = (self.sum_rewards - np.min(self.sum_rewards)) / (np.max(self.sum_rewards) - np.min(self.sum_rewards))
            self.avg_task_ratio = (self.avg_task_ratio - np.min(self.avg_task_ratio)) / (np.max(self.avg_task_ratio) - np.min(self.avg_task_ratio))
            self.avg_task_latency = (self.avg_task_latency - np.min(self.avg_task_latency)) / (np.max(self.avg_task_latency) - np.min(self.avg_task_latency))
            plt.plot(range(len(self.sum_rewards)), self.sum_rewards, label='sum_rewards')
            plt.plot(range(len(self.avg_task_ratio)), self.avg_task_ratio, label='avg_task_ratio')
            plt.plot(range(len(self.avg_task_latency)), self.avg_task_latency, label='avg_task_latency')
            plt.xlabel('step')
            plt.ylabel('performance')
            plt.legend()
            plt.savefig(f'{self.args.tensorboard_writer_file}/sum_rewards.png')
            plt.close()
            print('save figure')

        # if self.update_cnt % self.args.fre_to_draw * self.args.update_every == 0:
        #     self.draw_figure_to_tensorboard()
    
    def get_map_grid_vector(self, grid_width, grid_num, grid_x_range = [500, 2000], grid_y_range = [0, 1500]):
        # return grid_num * grid_num, 每个grid存储的信息是预计获得的reward，需要更新self.env.renew_channel()
        # 1. 存储当前的UAV 0 位置， 先把vehicle按照grid存储起来
        serve_vehicle_by_grid = [[[] for _ in range(grid_num)] for _ in range(grid_num)]
        task_vehicle_by_grid = [[[] for _ in range(grid_num)] for _ in range(grid_num)]
        all_sv_list = []
        all_tv_list = []
        for idx, vehicle in enumerate(self.env.vehicle_by_index):
            vehicle.cur_index = idx
            grid_x = (int((vehicle.position[0] - grid_x_range[0]) // grid_width))
            grid_y = (int((vehicle.position[1] - grid_y_range[0]) // grid_width))
            if grid_x < 0 or grid_y < 0:
                continue
            if grid_x >= grid_num or grid_y >= grid_num:
                continue
            if vehicle.assigned_to == -1:
                continue
            agent = self.env.UAVs[vehicle.assigned_to] if vehicle.assigned_to < self.env.n_UAV else self.env.BSs[vehicle.assigned_to - self.env.n_UAV]
            comm_range = self.args.UAV_communication_range if vehicle.assigned_to < self.env.n_UAV else self.args.RSU_communication_range
            if vehicle.serving:
                serve_vehicle_by_grid[grid_x][grid_y].append(vehicle)
                all_sv_list.append(vehicle)
            else:
                vehicle.pCPU = 0 # 先把pCPU清零
                task_vehicle_by_grid[grid_x][grid_y].append(vehicle)
                all_tv_list.append(vehicle)
            if distance(vehicle.position, agent.position) > comm_range:
                vehicle.assigned_to = -1
                continue
        # 2. 遍历，修改uav 0的位置到每个grid的中心
        map_grid_vector = np.zeros((grid_num, grid_num))
        map_grid_vector_asUAV = np.zeros((grid_num, grid_num))
        X_device = self.env.UAVs[0]
        org_v2v = self.get_V2VRateWithoutBand(self.env)
        org_i2v = self.get_I2VRateWithoutBand(self.env)
        org_v2i = self.get_V2IRateWithoutBand(self.env)
        org_v2v_as_u2u = self.get_V2VAsU2URateWithoutBand(self.env)
        V2V_band = self.env.bandwidth * self.args.V2V_RB / self.env.n_RB
        V2I_band = self.env.bandwidth * self.args.V2I_RB / self.env.n_RB
        V2VRate = org_v2v.copy() * V2V_band
        V2IRate = org_v2i.copy() * V2I_band
        I2VRate = org_i2v.copy() * V2I_band
        V2VaURate = org_v2v_as_u2u.copy() * V2I_band
        # 考虑只有BS的情况
        for i in range(self.env.n_BS):
            X_device = self.env.BSs[i]
            for tvidx, tv in enumerate(all_tv_list):
                if self.env.distance(tv.position, X_device.position) <= self.args.RSU_communication_range:
                    vidx = tv.cur_index
                    intRate = V2IRate[vidx, i] * I2VRate[i, :] / (V2IRate[vidx, i] + I2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    # V2VaU
                    idxs = intRate[:] > V2VaURate[vidx, :]
                    V2VaURate[vidx, idxs] = intRate[idxs]
        addedCPU = self.env.uav_cpu / max(1,len(all_sv_list))
        after_sum_inverse_sumRcpu, sum_sv_reward = self.get_sum_inverse_sumRcpu(V2VaURate, all_tv_list, all_sv_list, provided_CPU = addedCPU, calculate_for_all_veh=True)
        
        grid_dist = max(0,int(self.args.UAV_communication_range // grid_width))
        least_grid_pos = []
        for grid_x in range(grid_num):
            for grid_y in range(grid_num):
                # # 从grid_x, grid_y的-4, +4 范围内，所有的grid信息加入到map_grid_vector中
                # for i in range(-grid_dist, grid_dist+1):
                #     for j in range(-grid_dist, grid_dist+1):
                #         if grid_x+i < 0 or grid_y+j < 0 or grid_x+i >= grid_num or grid_y+j >= grid_num:
                #             continue
                for tv in task_vehicle_by_grid[grid_x][grid_y]:
                    map_grid_vector_asUAV[grid_x, grid_y] += tv.pCPU
                least_grid_pos.append(([grid_x, grid_y], map_grid_vector_asUAV[grid_x, grid_y]))
        least_grid_pos.sort(key=lambda x:x[1], reverse=True) # 按照least_grid_pos从大排序

        # map_grid_vector_asUAV /= max(1, np.max(map_grid_vector_asUAV)) # 归一化
        return map_grid_vector_asUAV, least_grid_pos

    def get_sum_inverse_sumRcpu(self, V2V_Rate, tv_list, sv_list, provided_CPU, calculate_for_all_veh = False):
        avg_task_ddl = self.env.avg_task_ddl
        avg_task_data = self.env.avg_task_data / avg_task_ddl
        avg_task_cpu = self.env.avg_task_cpu / avg_task_ddl
        sumRcpu_X = {}
        candidate_sv4tv = {}
        total_lambda = 0
        for tv in tv_list:
            total_lambda += tv.task_lambda
        V2V_Rate = V2V_Rate / max(1, total_lambda)
        for tv in tv_list:
            candidate_sv = sv_list.copy()
            trans_data_list = []
            tot_trans_time = 0
            tot_cpu = 0
            for sv in candidate_sv:
                trans_data_list.append((sv, V2V_Rate[tv.cur_index][sv.cur_index]))
                tot_trans_time += V2V_Rate[tv.cur_index][sv.cur_index]
                tot_cpu += sv.CPU_frequency
            tot_cpu /= len(tv_list)
            tot_trans_time /= max(1, len(candidate_sv)) # avg throughout
            avg_comp_time = avg_task_cpu * tv.task_lambda / max(1, tot_cpu)
            avg_trans_time = avg_task_data * tv.task_lambda / max(1, tot_trans_time)
            avg_trans_time_thr = avg_trans_time / max(0.1, avg_trans_time + avg_comp_time)
            # 降序排列
            trans_data_list.sort(key=lambda x:x[1], reverse=True)
            threshold_num = 1
            # 判断trans_data_list>=avg_task_data*0.5*tv.task_lambda的最小的N为threshold_num
            for i in range(len(sv_list)):
                # if threshold_num >= 20:
                #     break
                if trans_data_list[i][1] >= avg_task_data * avg_trans_time_thr * tv.task_lambda:
                    threshold_num = i
                else:
                    break
            # threshold_num = min(10, threshold_num)
            candidate_sv = [trans_data_list[i][0] for i in range(min(threshold_num, len(sv_list)))]
            trans_data_list = np.array([trans_data_list[i][1] for i in range(min(threshold_num, len(sv_list)))])
            candidate_sv4tv[tv] = (candidate_sv, [], [])
            trans_time_list = []
            # trans_time = avg_task_data * tv.task_lambda / max(1, np.sum(trans_data_list))
            if len(candidate_sv) == 0: # 说明没有找到合适的N
                continue
            trans_time_list = (avg_task_data * tv.task_lambda / len(candidate_sv) / trans_data_list)
            alloc_task_portion = tv.task_lambda / len(candidate_sv)
            Rcpu_list = []
            for svid, sv in enumerate(candidate_sv):
                R_cpu = alloc_task_portion * avg_task_cpu
                Rcpu_list.append(R_cpu)
                # R_cpu是每一个tv对每一个sv请求的cpu资源，求和
                if sumRcpu_X.get(sv) is None:
                    sumRcpu_X[sv] = R_cpu
                else:
                    sumRcpu_X[sv] += R_cpu
            candidate_sv4tv[tv] = (candidate_sv, Rcpu_list, trans_time_list)
        sum_inverse_sumRcpu = 0
        sum_tv_reward = 0
        for tv in tv_list:
            if calculate_for_all_veh:
                tv.pCPU = 0
            candidate_sv_list, Rcpu_list, trans_time_list = candidate_sv4tv[tv]
            tmp_inverse_sum = 0
            for idx, sv in enumerate(candidate_sv_list):
                tmp_pCPU = (sv.CPU_frequency+provided_CPU) / max(0.1, sumRcpu_X[sv]) * Rcpu_list[idx] # tv可以被分配到的资源
                tmp_comp_time = (avg_task_cpu * tv.task_lambda) / len(candidate_sv_list) / max(0.1, tmp_pCPU)
                tmp_time = tmp_comp_time + trans_time_list[idx]
                tmp_utility = 1 + np.log(1 + 1 - tmp_time) if tmp_time < 1 else np.exp(2 * (1 - tmp_time))
                # tmp_utility = min(1, 1/tmp_time)
                tmp_utility = tmp_utility * tv.task_lambda / len(candidate_sv_list)
                tmp_inverse_sum += tmp_utility
                # tmp_inverse_sum += tmp_pCPU
            # tmp_total_time = (avg_task_cpu * tv.task_lambda) / max(0.1, tmp_inverse_sum) + trans_time
            # tmp_utility = 1 + np.log(1 + 1 - tmp_total_time) if tmp_total_time < 1 else np.exp(1 - tmp_total_time)
            # tmp_utility = min(1, 1/tmp_total_time)
            # tmp_utility *= tv.task_lambda
            if calculate_for_all_veh:
                # tv.pCPU -= (np.sum(Rcpu_list)) 
                tv.pCPU = tmp_inverse_sum
            sum_tv_reward += tv.pCPU 
            sum_inverse_sumRcpu += tmp_inverse_sum - tv.pCPU
        return sum_inverse_sumRcpu, sum_tv_reward
    def calculate_reward(self):
        '''根据env以及内部存储的task信息，对比传输的数据量和计算的数据量，计算reward，存储在reward_memory中'''
        reward_type = 1
        all_rewards = np.zeros(self.n_agent)
        total_all_reward = 0 # 用于计算整个系统的奖励，指导UAV轨迹
        if reward_type == 1:
            succeed_tasks = self.env.succeed_tasks # dict
            assigned_cpu = np.zeros(self.n_agent) + self.env.uav_cpu
            assigned_req = np.zeros(self.n_agent)
            for task_id in self.to_offload_task_ids:
                if succeed_tasks.get(task_id) is None:
                    continue
                task = succeed_tasks[task_id]
                gen_veh = task.g_veh
                assigned_to = gen_veh.assigned_to
                if assigned_to == -1:
                    continue
                agent = self.env.UAVs[assigned_to] if assigned_to < self.env.n_UAV else self.env.BSs[assigned_to-self.env.n_UAV]
                comm_range = self.args.UAV_communication_range if assigned_to < self.env.n_UAV else self.args.RSU_communication_range
                if self.env.distance(agent.position, gen_veh.position) > comm_range:
                    gen_veh.assigned_to = -1
                    continue
                all_rewards[assigned_to] += task.get_task_utility()
            for veh in self.env.vehicle_by_index:
                if veh.assigned_to != -1:
                    agent = self.env.UAVs[veh.assigned_to] if veh.assigned_to < self.env.n_UAV else self.env.BSs[veh.assigned_to-self.env.n_UAV]
                    comm_range = self.args.UAV_communication_range if veh.assigned_to < self.env.n_UAV else self.args.RSU_communication_range
                    if self.env.distance(agent.position, veh.position) > comm_range:
                        veh.assigned_to = -1
                        continue
                    assigned_cpu[veh.assigned_to] += veh.CPU_frequency
                    assigned_req[veh.assigned_to] += veh.task_lambda * self.env.avg_task_cpu 
            # for agent_id in range(self.n_agent):
            #     # 归一化
            #     baseline_time = assigned_req[agent_id] / max(1, assigned_cpu[agent_id])
            #     avg_task_ddl = self.env.avg_task_ddl
            #     baseline_utility = 1 + np.log(1 + avg_task_ddl - baseline_time) if baseline_time < avg_task_ddl else 0# np.exp(5 * (avg_task_ddl - baseline_time))
            #     baseline_utility *= assigned_req[agent_id] / self.env.avg_task_cpu # 乘以平均的task_lambda数量
            #     all_rewards[agent_id] = all_rewards[agent_id] #/ max(1,baseline_utility)
            all_rewards *= 1000 / self.env.total_task_lambda
            total_all_reward = np.sum(all_rewards)
        elif reward_type == 2:
            assigned_TV_list = [[] for _ in range(self.n_agent)]
            assigned_SV_list = [[] for _ in range(self.n_agent)]
            covered_TV_list = [[] for _ in range(self.n_agent)]
            covered_SV_list = [[] for _ in range(self.n_agent)]
            vehicle_list = self.env.vehicle_by_index
            total_task_per_second = 0
            for cur_idx, vehicle in enumerate(vehicle_list):
                vehicle.cur_index = cur_idx
                agent_id = vehicle.assigned_to
                if agent_id != -1: 
                    agent = self.env.UAVs[agent_id] if agent_id < self.env.n_UAV else self.env.BSs[agent_id-self.env.n_UAV]
                    comm_range = self.args.UAV_communication_range if agent_id < self.env.n_UAV else self.args.RSU_communication_range
                    if self.env.distance(agent.position, vehicle.position) > comm_range:
                        vehicle.assigned_to = -1
                        continue
                    if not vehicle.serving:
                        assigned_TV_list[agent_id].append(vehicle)
                        covered_TV_list[agent_id].append(vehicle)
                        total_task_per_second += vehicle.task_lambda
                    else:
                        assigned_SV_list[agent_id].append(vehicle)
                        covered_SV_list[agent_id].append(vehicle)
            
            # 先假设完全没有RSU和UAV，完全通过veh自己的决策，他们作为离散设备形成的coalition，计算reward
            all_sv_list = []
            for sv_list in assigned_SV_list:
                all_sv_list.extend(sv_list)
            all_tv_list = []
            for tv_list in assigned_TV_list:
                all_tv_list.extend(tv_list)
            org_v2v = self.get_V2VRateWithoutBand(self.env).copy() * self.args.V2V_RB / self.env.n_RB
            org_v2u = self.get_V2URateWithoutBand(self.env).copy() * (self.args.V2U_RB) / self.env.n_RB
            org_u2v = self.get_U2VRateWithoutBand(self.env).copy() * (self.args.V2U_RB) / self.env.n_RB
            org_v2i = self.get_V2IRateWithoutBand(self.env).copy() * (self.args.V2I_RB) / self.env.n_RB
            org_i2v = self.get_I2VRateWithoutBand(self.env).copy() * (self.args.V2I_RB) / self.env.n_RB
            assumed_band = self.env.bandwidth #/ max(1, total_task_per_second)
            # all_after_sum_inverse_sumRcpu, all_sum_sv_reward = self.get_sum_inverse_sumRcpu(org_v2v * assumed_band, all_tv_list, all_sv_list, provided_CPU = 0, calculate_for_all_veh=True)
            for agent_id in range(self.n_agent):
                if agent_id < self.env.n_UAV:
                    V2XRate = org_v2u.copy() * assumed_band
                    X2VRate = org_u2v.copy() * assumed_band
                    X_device = self.env.UAVs[agent_id]
                else:
                    V2XRate = org_v2i.copy() * assumed_band
                    X2VRate = org_i2v.copy() * assumed_band
                    X_device = self.env.BSs[agent_id-self.env.n_UAV]
                # # 获取V2V_fading和V2X_fading
                V2VRate = org_v2v.copy() * assumed_band
                tv_list = assigned_TV_list[agent_id]
                sv_list = assigned_SV_list[agent_id]
                # tv_list = covered_TV_list[agent_id]
                # sv_list = covered_SV_list[agent_id]
                sv_list.append(X_device)
                X_device.cur_index = len(sv_list)
                X_device.reward = 0
                if len(tv_list) == 0:
                    continue
                before_sum_inverse_sumRcpu = 0
                i = agent_id if agent_id < self.env.n_UAV else agent_id - self.env.n_UAV
                for tv in tv_list:
                    vidx = tv.cur_index
                    intRate = V2XRate[vidx, i] * X2VRate[i, :] / (V2XRate[vidx, i] + X2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                # V2VRate 在 第二个维度 加上一个X_device
                V2VRate = np.concatenate((V2VRate, np.zeros((V2VRate.shape[0], 1))), axis=1)
                # 这个X_device的V2VRate是所有的tv对它的V2XRate
                V2VRate[:, -1] = V2XRate[:, i]
                addedCPU = self.env.uav_cpu / max(1,len(all_sv_list))
                after_sum_inverse_sumRcpu, sum_sv_reward = self.get_sum_inverse_sumRcpu(V2VRate, tv_list, sv_list, provided_CPU = 0, calculate_for_all_veh=False)
                total_all_reward += after_sum_inverse_sumRcpu + sum_sv_reward
                all_rewards[agent_id] = after_sum_inverse_sumRcpu + sum_sv_reward
                # for tv in covered_TV_list[agent_id]:
                #     total_all_reward += tv.pCPU
                #     all_rewards[agent_id] += tv.pCPU
        
        all_rewards = all_rewards * 100 / self.env.n_TV
        self.uav_rewards = all_rewards[:self.env.n_UAV]
        self.last_cnt_succ_num = self.env.get_succeed_task_cnt()
        self.sum_rewards.append(np.sum(all_rewards)*6)
    def reset_state(self):
        self.last_cnt_succ_num = 0
        self.uav_rewards = []
        self.to_offload_task_ids = []
        self.failed_task_idx = 0

    def draw_figure_to_tensorboard(self):
        env = self.env
        # 从env获取vehicle_by_index，里面各个vehicle的位置，根据vehicle.serving判断是任务车辆还是服务车辆
        # 从env获取UAVs和BSs的位置
        # 使用matplotlib画图，保存到tensorboard
        vehicle_by_index = env.vehicle_by_index
        uavs = env.UAVs
        bss = env.BSs
        # 画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Env Time: {self.env.cur_time}')
        # Create dummy plots for the legend
        ax.scatter([], [], c='r', marker='o', label='serving')
        ax.scatter([], [], c='b', marker='o', label='task')
        ax.scatter([], [], c='g', marker='^', label='uav')
        ax.scatter([], [], c='y', marker='^', label='bs')

        # Plot the points
        for vehicle in vehicle_by_index:
            if vehicle.serving:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='r', marker='o')
            else:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='b', marker='o')
        for bs in bss:
            ax.scatter(bs.position[0], bs.position[1], c='y', marker='^')
        for uav in uavs:
            ax.scatter(uav.position[0], uav.position[1], c='g', marker='^')

        ax.legend()
        self.writer.add_figure('system_fig', fig, global_step=self.update_cnt)
        plt.close()

    def act_CPU_allocation(self, env):
        '''默认情况下返回的是None，表示FIFO，否则返回一个字典，表示X_device对于每个任务的CPU分配情况，总和为1
        return [{
            'X_device':device,
            'CPU_allocation':[1, 0, 0, 0, 0] # 这里假设只有5个任务
            'is_to_cheat':[True, False, False, False, False]
        }]
        '''
        cpu_allocation_for_fog_nodes = []
        serving_vehicles = env.serving_vehicles.values()
        uavs = env.UAVs
        bss = env.BSs
        devices = list(serving_vehicles) + list(uavs) + list(bss)
        for device in devices:
            task_len = len(device.task_queue)
            if task_len > 0:
                info_dict = {}
                cheat_or_not = np.zeros(shape=(task_len), dtype='bool')
                info_dict['device'] = device
                cpu_alloc = np.zeros(shape=(task_len), dtype='float')
                cpu_alloc[0] = 1
                info_dict['CPU_allocation'] = cpu_alloc
                info_dict['is_to_cheat'] = cheat_or_not
                cpu_allocation_for_fog_nodes.append(info_dict)
        return cpu_allocation_for_fog_nodes

    
    
    def act_RB_allocation(self, env):
        activated_offloading_tasks_with_RB_Nos = np.zeros((len(env.offloading_tasks), env.n_RB), dtype='bool')
        # 简单点，直接平均分配，不过需要考虑V2V_band, V2U_band, V2I_band: 6, 6, 8
        V2U_RB = self.args.V2U_RB
        V2V_RB = self.args.V2V_RB
        V2I_RB = self.args.V2I_RB
        V2U_cnt = 0
        V2V_cnt = 0
        V2I_cnt = 0
        max_serving_cnt = 20
        if len(env.offloading_tasks) > 0:
            # 遍历每个任务的routing[-1]，根据所在位置的横纵坐标添加到列表，然后排序，按照序号来进行资源平均分配，这样尽量保障干扰发生在比较远的两个车辆
            position_list = []
            serve_num = (len(env.offloading_tasks))
            for i in range(len(env.offloading_tasks)):
                device = env.offloading_tasks[i]['task'].routing[-1]
                last_transmit_time = env.offloading_tasks[i]['task'].last_transmit_time
                mode = env.offloading_tasks[i]['mode']
                if mode == 'V2U' or mode == 'U2V':
                    V2U_cnt += 1
                    if V2U_cnt > max_serving_cnt:
                        continue
                elif mode == 'V2I' or mode == 'I2V':
                    V2I_cnt += 1
                    if V2I_cnt > max_serving_cnt:
                        continue
                position_list.append((device.position, i, last_transmit_time))
                if len(position_list) > serve_num:
                    break
            serve_num = min(serve_num, len(position_list))
            V2U_cnt = 0
            V2V_cnt = 0
            V2I_cnt = 0
            position_list.sort(key=lambda x: last_transmit_time*10000 + x[0][0]//50 * 100 + x[0][1]//50)
            for pidx in range(serve_num):
                i = position_list[pidx][1]
                mode = env.offloading_tasks[i]['mode']
                if mode == 'V2U' or mode == 'U2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2U_cnt
                    V2U_cnt += 1
                elif mode == 'V2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2V_cnt
                    V2V_cnt += 1
                elif mode == 'V2I' or mode == 'I2V':
                    env.offloading_tasks[i]['RB_cnt'] = V2I_cnt
                    V2I_cnt += 1
            avg_V2V_RB_num = max(1, int(self.args.V2V_RB / max(1, V2V_cnt)))
            avg_V2U_RB_num = max(1, int(self.args.V2U_RB / max(1, V2U_cnt)))
            avg_V2I_RB_num = max(1, int(self.args.V2I_RB / max(1, V2I_cnt)))
            for pidx in range(serve_num):
                i = position_list[pidx][1]
                mode = env.offloading_tasks[i]['mode']
                RB_cnt = env.offloading_tasks[i]['RB_cnt']
                if mode == 'V2U' or mode == 'U2V':
                    activated_offloading_tasks_with_RB_Nos[i, RB_cnt*avg_V2U_RB_num%env.n_RB:(RB_cnt+1)*avg_V2U_RB_num%env.n_RB] = True
                elif mode == 'V2V':
                    activated_offloading_tasks_with_RB_Nos[i, V2V_RB + RB_cnt*avg_V2V_RB_num%env.n_RB : V2V_RB + (RB_cnt+1)*avg_V2V_RB_num%env.n_RB] = True
                elif mode == 'V2I' or mode == 'I2V':
                    activated_offloading_tasks_with_RB_Nos[i, V2V_RB + V2U_RB + RB_cnt*avg_V2I_RB_num%env.n_RB : V2V_RB + V2U_RB + (RB_cnt+1)*avg_V2I_RB_num%env.n_RB] = True
        return activated_offloading_tasks_with_RB_Nos
    def act_mobility2(self, env, states, vehicle_positions):
        uav_directions = np.zeros(shape=(env.n_UAV))
        uav_speeds = np.zeros(shape=(env.n_UAV))
        
        if len(vehicle_positions) < env.n_UAV:
            return uav_directions, uav_speeds
        # 获取cpu map and request map
        vehicles = env.vehicle_by_index
        rrows, rcols = [], []
        crows, ccols = [], []
        requests = []
        cpu_res = []
        for vidx, v in enumerate(vehicles):
            v.cur_index = vidx
            if not v.serving:
                rrows.append(v.position[0])
                rcols.append(v.position[1])
                requests.append(v.task_lambda)
            else:
                crows.append(v.position[0])
                ccols.append(v.position[1])
                cpu_res.append(v.CPU_frequency)
        
        rrows, rcols = np.array(rrows), np.array(rcols)
        crows, ccols = np.array(crows), np.array(ccols)
        positions = np.zeros((len(rrows),2))
        # 提取req map的坐标作为是position，同时使用一些公式计算出来weight
        for idx, (rrow, rcol) in enumerate(zip(rrows, rcols)):
            v = 0
            positions[idx, 0] = rrow
            positions[idx, 1] = rcol
        # K-means 聚类
        kmeans = KMeans(n_clusters=env.n_UAV, max_iter=5).fit(positions)

        # 获取聚类中心
        cluster_centers = kmeans.cluster_centers_
        x_range = [[1000, 1500], [500, 1000], [1000, 1500], [1500, 2000]]
        y_range = [[250, 750], [750, 1250], [750, 1250], [250, 750]]
        tmp_uav_reward = 0
        for i in range(self.env.n_UAV):
            uav_position = self.env.UAVs[i].position
            uav_gridx = min(max(0,int((uav_position[0] - x_range[i][0]) // self.args.grid_width)), 10 - 1)
            uav_gridy = min(max(0,int((uav_position[1] - y_range[i][0]) // self.args.grid_width)), 10 - 1)
            map_m, least_grid = self.get_map_grid_vector(self.args.grid_width, 10, grid_x_range=x_range[i], grid_y_range=y_range[i])
            tmp_uav_reward += map_m[uav_gridx, uav_gridy] #/ max(1, self.env.n_TV)
            tmp_center = (np.array(least_grid[0][0])) * self.args.grid_width
            tmp_center[0] += x_range[i][0]
            tmp_center[1] += y_range[i][0]
            cluster_centers[i] = tmp_center
        uav_positions = env.get_uav_positions()
        # 计算 UAV 的方向和速度
        # 为每个无人机分配最近的聚类中心
        for i, uav_position in enumerate(uav_positions):
            closest_center_index = np.argmin([distance(uav_position, center) for center in cluster_centers])
            target_position = cluster_centers[closest_center_index]
            
            # 需要保证每个UAV不非出x_range和y_range的范围
            target_position[0] = min(max(target_position[0], x_range[i][0]), x_range[i][1])
            target_position[1] = min(max(target_position[1], y_range[i][0]), y_range[i][1])
            # cluster_centers是ndarray，删除第closest_center_index行的元素
            cluster_centers = np.delete(cluster_centers, closest_center_index, axis=0)
            # 方向是四个方向，上下左右或者不动，等于弧度制的[0, 0.5pi, pi, 1.5pi] 以及 不动
            direction = (np.arctan2(target_position[1] - uav_position[1], target_position[0] - uav_position[0]) + 2 * np.pi) % ( 2* np.pi)
            direction = int(direction / (np.pi / 2)) * (np.pi / 2)
            
            uav_directions[i] = direction

            # 计算速度
            dist = distance(uav_position, target_position)
            uav_speeds[i] = min(dist / env.time_step, env.uav_max_speed)
        self.uav_weights = np.ones(shape=(env.n_UAV))
        self.bs_weights = np.ones(shape=(env.n_BS))
        return uav_directions, uav_speeds
    def act_mobility(self, env:Environment):
        vehicle_positions = env.get_vehicle_positions()
        states, pos_states = env.get_states(self.uav_weights, self.bs_weights, self.map_width, self.map_height)
        self.env_cur_state = states
        # # 随机放置UAV，目的是最大化attempt_reward返回的结果
        # best_reward = self.attemp_reward()
        # best_uav_position = env.get_uav_positions()
        # for _ in range(1000):
        #     uav_positions = np.random.rand(env.n_UAV, 2) * 2000
        #     for idx, uav in enumerate(env.UAVs):
        #         uav.position = uav_positions[idx]
        #     reward = self.attemp_reward()
        #     if reward > best_reward:
        #         best_reward = reward
        #         best_uav_position = uav_positions
        # for idx, uav in enumerate(env.UAVs):
        #     uav.position = best_uav_position[idx]

        # # 返回0即可，因为不需要移动
        # uav_directions = np.zeros(shape=(env.n_UAV))
        # uav_speeds = np.zeros(shape=(env.n_UAV))
        # return uav_directions, uav_speeds
        return self.act_mobility2(env, self.env_cur_state, vehicle_positions)
        
    
    def act_offloading(self, env):
        vehicle_list = env.vehicle_by_index
        V2VRate = self.get_V2VRateWithoutBand(env) * self.args.V2V_RB / env.n_RB
        V2URate = self.get_V2URateWithoutBand(env) * self.args.V2U_RB / env.n_RB
        U2VRate = self.get_U2VRateWithoutBand(env) * self.args.V2U_RB / env.n_RB
        V2IRate = self.get_V2IRateWithoutBand(env) * self.args.V2I_RB / env.n_RB
        I2VRate = self.get_I2VRateWithoutBand(env) * self.args.V2I_RB / env.n_RB
        tot_serving = 0
        n_tt = math.ceil(env.time_step / env.TTI)
        tot_upt = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_reqt = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_taut = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_genv_idx = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_task_idx = [[[] for j in range(n_tt)] for i in range(self.n_agent)]
        tot_genv_cnt = [set() for _ in range(self.n_agent)]
        task_path_dict_list = []
        offloading_tasks = [] # 用来记录这个time_step内所有的决策offloading的tasks，方便计算reward
        # 添加task的数据，遍历env.to_offload_tasks，并且看他的generate_vehicle在哪个基站的覆盖范围内
        self.to_offload_task_ids = []
        for task_idx, task in enumerate(env.to_offload_tasks):
            veh = task.g_veh
            arrival_time_slot = max(0,int((task.start_time - env.cur_time) / env.TTI))
            device_idx = veh.assigned_to
            if device_idx == -1:
                continue
            device = env.UAVs[device_idx] if device_idx < env.n_UAV else env.BSs[device_idx - env.n_UAV]
            comm_range = self.args.UAV_communication_range if device_idx < env.n_UAV else self.args.RSU_communication_range
            if env.distance(veh.position, device.position) > comm_range:
                veh.assigned_to = -1
                continue
            self.to_offload_task_ids.append(task.id)
            tot_upt[device_idx][arrival_time_slot].append(task.ini_data_size)
            tot_reqt[device_idx][arrival_time_slot].append(task.ini_cpu)
            tot_taut[device_idx][arrival_time_slot].append(task.ddl // env.TTI)
            vbidx = env.vehicle_by_index.index(veh)
            tot_genv_idx[device_idx][arrival_time_slot].append(vbidx) # 通过index可以定位所有的传输车辆，把每个任务的车辆看成是不同的k即可
            tot_task_idx[device_idx][arrival_time_slot].append(task_idx)
            tot_genv_cnt[device_idx].add(vbidx)

        for i in range(env.n_UAV):
            task_cnt = 0
            n_kt = 0
            n_jt = 1 # 直接假设无人机等价于两个车,并且放在最后
            veh_ids = []
            sv_ids = []
            tv_ids = []
            cpu_res = []
            # 添加serving vehicle的数据
            for vidx, veh in enumerate(vehicle_list):
                if veh.assigned_to == i:
                    device = env.UAVs[i]
                    if env.distance(veh.position, device.position) > comm_range:
                        veh.assigned_to = -1
                        continue
                    veh_ids.append(vidx)
                    intRate = V2URate[vidx, i] * U2VRate[i, :] / (V2URate[vidx, i] + U2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                        cpu_res.append(veh.CPU_frequency)
            cpu_res.append(env.UAVs[i].CPU_frequency)
            tot_serving += n_jt
            n_kt = len(tot_genv_cnt[i]) # 这个存储的是set
            if n_kt > 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                task_cnt = 0
                assigned_subscript = []
                tv_ids = list(tot_genv_cnt[i])
                for tt in range(n_tt):
                    task_num = len(tot_upt[i][tt]) # 每个时隙的任务数量最多只有1个
                    for gv_task in range(task_num):
                        k = tot_genv_idx[i][tt][gv_task] # 车辆在v_by_index里面的index
                        region_tv_index = tv_ids.index(k)
                        upt[tt, region_tv_index] = tot_upt[i][tt][gv_task]
                        reqt[tt, region_tv_index] = tot_reqt[i][tt][gv_task]
                        taut[tt, region_tv_index] = tot_taut[i][tt][gv_task]
                        
                        assigned_subscript.append((tt, region_tv_index, gv_task))
                        task_cnt += 1
                if task_cnt > 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    gt[:len(tv_ids), -1, 0] = V2URate[tv_ids, i]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    task_assignment = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = n_tt, F_jt = cpu_res, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = env.TTI, Bt = env.bandwidth)
                    for (tt, k, sub_cnt) in assigned_subscript:
                        j = np.argmax(task_assignment[tt,k,:])
                        if task_assignment[tt,k,j] == 0:
                            continue
                        k = tv_ids[k]
                        X_device = None
                        to_relay = False
                        if j + 1 >= n_jt: # 无人机
                            j = i # 无人机的idx
                            X_type = 'UAV'
                            X_device = env.UAVs[j]
                        else:
                            j = sv_ids[j]
                            X_type = 'Veh'
                            X_device = vehicle_list[j]
                            intRate = V2URate[k, i] * U2VRate[i, :] / (V2URate[k, i] + U2VRate[i, :])
                            idxs = intRate[:] > V2VRate[k, :]
                            to_relay = idxs[j]
                            if X_device in vehicle_list[k].neighbor_vehicles:
                                dist_v_indice = vehicle_list[k].neighbor_vehicles.index(X_device)
                        offload_path = [{
                                'X_device':X_device
                            }]    
                        if to_relay and X_type == 'Veh': # 如果卸载到车，并且这个车是通过UAV进行的relay
                            relay_device = env.UAVs[i]
                            offload_path.insert(0, {
                                'X_device':relay_device
                            })
                        task_path_dict_list.append({
                            'task':env.to_offload_tasks[tot_task_idx[i][tt][sub_cnt]],
                            'offload_path': offload_path,
                            'task_type':'offload'
                        })
                        offloading_tasks.append(task_path_dict_list[-1]['task'])

        for bsi in range(env.n_BS):
            i = bsi + env.n_UAV
            task_cnt = 0
            n_kt = 0
            n_jt = 1 # BS=10Veh
            veh_ids = []
            sv_ids = []
            tv_ids = []
            cpu_res = []
            for vidx, veh in enumerate(vehicle_list):
                if veh.assigned_to == i:
                    veh_ids.append(vidx)
                    intRate = V2IRate[vidx, bsi] * I2VRate[bsi, :] / (V2IRate[vidx, bsi] + I2VRate[bsi, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                        cpu_res.append(veh.CPU_frequency)
            cpu_res.append(env.BSs[bsi].CPU_frequency)
            tot_serving += n_jt
            n_kt = len(tot_genv_cnt[i])
            if n_kt > 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                task_cnt = 0
                assigned_subscript = []
                tv_ids = list(tot_genv_cnt[i])
                for tt in range(n_tt):
                    task_num = len(tot_upt[i][tt]) # 每个时隙的任务数量最多只有1个
                    for gv_task in range(task_num):
                        k = tot_genv_idx[i][tt][gv_task] # 车辆在v_by_index里面的index
                        region_tv_index = tv_ids.index(k)
                        upt[tt, region_tv_index] = tot_upt[i][tt][gv_task]
                        reqt[tt, region_tv_index] = tot_reqt[i][tt][gv_task]
                        taut[tt, region_tv_index] = tot_taut[i][tt][gv_task]
                        assigned_subscript.append((tt, region_tv_index, gv_task))
                        task_cnt += 1
                if task_cnt > 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    gt[:len(tv_ids), -1, 0] = V2IRate[tv_ids, bsi]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    task_assignment = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = n_tt, F_jt = cpu_res, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = env.TTI, Bt = env.bandwidth)
                    for (tt, k, sub_cnt) in assigned_subscript:
                        j = np.argmax(task_assignment[tt,k,:])
                        if task_assignment[tt,k,j] == 0:
                            continue
                        k = tv_ids[k]
                        X_type='Veh'
                        to_relay = False
                        if j + 1 >= n_jt: # BS
                            j = bsi # BS的id
                            X_type = 'RSU'
                            X_device = env.BSs[j]
                        else:
                            j = sv_ids[j]
                            X_type = 'Veh'
                            X_device = vehicle_list[j]
                            intRate = V2IRate[k, bsi] * I2VRate[bsi, :] / (V2IRate[k, bsi] + I2VRate[bsi, :])
                            idxs = intRate[:] > V2VRate[k, :]
                            to_relay = idxs[j]
                            if X_device in vehicle_list[k].neighbor_vehicles:
                                dist_v_indice = vehicle_list[k].neighbor_vehicles.index(X_device)
                            
                        offload_path = [{
                            'X_device':X_device
                            }]    
                        if to_relay and X_type == 'Veh': # 如果卸载到车，并且这个车是通过UAV进行的relay
                            relay_device = env.BSs[bsi]
                            offload_path.insert(0,{
                                'X_device':relay_device
                            })
                        task_path_dict_list.append({
                            'task':env.to_offload_tasks[tot_task_idx[i][tt][sub_cnt]],
                            'offload_path': offload_path,
                            'task_type':'offload'
                        })
                        offloading_tasks.append(task_path_dict_list[-1]['task'])
        self.cur_offloading_tasks = offloading_tasks
        return task_path_dict_list
            

    def act_verification(self, env):
        '''通过将要卸载的任务，来决定哪些任务进行验证，返回：[
            {
                'task':env.to_offload_tasks[0],
                'offload_path': [device],
                'task_type':'verification',
                'ddl':100 # 区别于任务的ddl，是针对验证任务的ddl
            }
        ]
        '''
        vehicle_list = env.vehicle_by_index
        to_offload_tasks = env.to_offload_tasks
        verified_task_path_dict_list = []
        return verified_task_path_dict_list

    def act_update_reputation(self, env):
        '''返回需要更新分数的设备列表
        [
            {
                'device': env.vehicle_by_index[0],
                'current_reputation': 0.5
            }
        ]
        '''
        return []
    def get_V2VAsU2URateWithoutBand(self, env):
        # 把vehicle里面的V2V信道替换为V2U信道
        V2VaU_Signal = 10 ** ((env.V2U_power_dB - env.V2VChannel_with_fastfading + 25) / 10)[:,:,0]
        V2VaU_Rate_without_Bandwidth = np.log2(1 + np.divide(V2VaU_Signal, env.sig2))
        return V2VaU_Rate_without_Bandwidth

    def get_V2VRateWithoutBand(self, env):
        V2V_Signal = 10 ** ((env.V2V_power_dB - env.V2VChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2V_Rate_without_Bandwidth = np.log2(1 + np.divide(V2V_Signal, env.sig2))
        return V2V_Rate_without_Bandwidth

    def get_V2URateWithoutBand(self, env):
        V2U_Signal = 10 ** ((env.V2U_power_dB - env.V2UChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2U_Rate_without_Bandwidth = np.log2(1 + np.divide(V2U_Signal, env.sig2))
        return V2U_Rate_without_Bandwidth

    def get_V2IRateWithoutBand(self, env):
        V2I_Signal = 10 ** ((env.V2I_power_dB - env.V2IChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        V2I_Rate_without_Bandwidth = np.log2(1 + np.divide(V2I_Signal, env.sig2))
        return V2I_Rate_without_Bandwidth

    def get_I2VRateWithoutBand(self, env):
        # env.I2VChannel_with_fastfading 是 env.V2IChannel_with_fastfading 的转置
        I2V_Signal = 10 ** ((env.I2V_power_dB - env.I2VChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        I2V_Rate_without_Bandwidth = np.log2(1 + np.divide(I2V_Signal, env.sig2))
        return I2V_Rate_without_Bandwidth

    def get_U2VRateWithoutBand(self, env):
        U2V_Signal = 10 ** ((env.U2V_power_dB - env.U2VChannel_with_fastfading) / 10)[:,:,0]
        U2V_Rate_without_Bandwidth = np.log2(1 + np.divide(U2V_Signal, env.sig2))
        return U2V_Rate_without_Bandwidth