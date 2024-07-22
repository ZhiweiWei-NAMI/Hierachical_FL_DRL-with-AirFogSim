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
from .WHO_algorithm import WHO
import torch
import torch.nn as nn
from .CNN_PSO import CNN_PPO
import os


class CNN_Weighted_Voronoi_Algorithm_Module(Base_Algorithm_Module):
    def __init__(self, env, training = True, model_path = None, pre_trained = None):
        super().__init__()
        n_hiddens = 256 # 隐含层数量
        actor_lr = 3e-4
        critic_lr = 5e-4
        gamma = 0.95
        self.map_width = self.map_height = 100
        lmbda = 0.97
        eps = 0.2
        self.device = 'cuda:0'
        device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.max_weight = 2 # 权重代表距离的额外加减部分
        self.max_weight_bs = 2
        self.n_agent = env.n_UAV + env.n_BS
        self.n_channel = 2
        self.uav_weights = np.zeros(env.n_UAV)
        self.bs_weights = np.zeros(env.n_BS)
        self.n_UAV = env.n_UAV
        self.n_BS = env.n_BS
        self.actor_loss = deque([0], maxlen=30)
        self.critic_loss = deque([0], maxlen=30)
        self.log_alpha = deque([0], maxlen=30)
        self.buffer_size = int(400)
        self.batch_size = 512
        self.transition_dicts = [{'actions_weight':deque([],maxlen=self.buffer_size),'pos_states':deque([],maxlen=self.buffer_size),'new_pos_states':deque([],maxlen=self.buffer_size),'states': deque([],maxlen=self.buffer_size),'actions1': deque([],maxlen=self.buffer_size),'actions2': deque([],maxlen=self.buffer_size),'weights': deque([],maxlen=self.buffer_size),'new_weights': deque([],maxlen=self.buffer_size),'next_states': deque([],maxlen=self.buffer_size),'rewards': deque([],maxlen=self.buffer_size),'dones': deque([],maxlen=self.buffer_size),'log_probs': deque([],maxlen=self.buffer_size)} for i in range(self.n_agent)]
        self.transition_dicts_2 = self.transition_dicts.copy()
        self.n_actions = np.array([4, 5]) # 悬停或者上下左右移动，另一个符号控制权重等级
        self.agents_uav = [CNN_PPO(input_states = [self.map_width, self.map_height, self.n_channel],
            n_hiddens = n_hiddens,
            n_actions = [5,6],
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            n_agent = self.n_agent,
            lmbda = lmbda,
            eps = eps,
            gamma = gamma,
            device = device) for _ in range(1)]
        if pre_trained is not None:
            self.load_agents(pre_trained)
            for agent_uav in self.agents_uav:
                agent_uav.epislon = 1
        if not training:
            for agent_uav in self.agents_uav:
                agent_uav.epislon = 0.1

        x = np.arange(self.map_height**2) // self.map_height
        y = np.arange(self.map_height**2) % self.map_height

        # 使用NumPy 的 broadcasting 机制计算欧式距离
        xi, xj = np.meshgrid(x, x)
        yi, yj = np.meshgrid(y, y)

        self.grid_distance = np.sqrt((xi - xj)**2 + (yi - yj)**2) + 25
        V2V_PL = 40.0 * np.log10(self.grid_distance) + 9.45 - 17.3 * np.log10(1.5) * 2 + 2.7 * np.log10(2/5)
        V2V_Signal = 10 ** ((env.V2V_power_dB - V2V_PL) / 10)
        self.V2V_grid_rate = np.log2(1 + np.divide(V2V_Signal, env.sig2))
        mask = self.grid_distance > 50 # 只允许附近格点的vehicle通信
        self.V2V_grid_rate[mask] = 0
        
    def load_agents(self, model_path):
        self.agents_uav[0].load_models(model_path)

    def save_agents(self, model_path):
        # self.agents_weight.save_agents(model_path)
        self.agents_uav[0].save_models(model_path)

    def act_mobility(self, env, pos_states, all_weights, states = None, vehicle_positions=None):
        # 根据全局车辆位置，以及全局计算资源需求（计算task queue的总和），归一化变成2D的map
        # states = env.get_states(vehicle_ids, vehicle_positions, 50, 50)
        uav_directions = []
        uav_weights = []
        bs_weights = []
        actions1 = []
        actions2 = []
        log_probs = []
        entropies = []
        # 输入4个map给每一个agent的CNN，再给到MADRL的每一个agent，得到一个action表示无人机的移动方向和权重
        for i in range(env.n_UAV):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i]
            state_i = np.expand_dims(state_i, axis=0)
            action, direction, weight, log_prob, entropy = self.agents_uav[0].take_action(state_i, i, pos_states, all_weights) 
            entropies.append(entropy)
            actions1.append(action)
            actions2.append(0)
            direction = direction / 4 * 2 * np.pi
            # if weight == 1:
            #     weight = self.uav_weights[i] + 1
            # elif weight == 2:
            #     weight = self.uav_weights[i] - 1
            # else:
            #     weight = self.uav_weights[i]
            weight = np.clip(weight, 0, self.max_weight)
            uav_directions.append(direction)
            uav_weights.append(weight)
            log_probs.append(log_prob)
        for i in range(env.n_BS):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i+env.n_UAV]
            state_i = np.expand_dims(state_i, axis=0)
            selfidx = i+env.n_UAV
            action, direction, weight, log_prob, entropy = self.agents_uav[0].take_action(state_i, selfidx, pos_states, all_weights) 
            entropies.append(entropy)
            actions1.append(action)
            actions2.append(0)
            # if weight == 1:
            #     weight = self.bs_weights[i] + 1
            # elif weight == 2:
            #     weight = self.bs_weights[i] - 1
            # else:
            #     weight = self.bs_weights[i]
            weight = np.clip(weight, 0, self.max_weight_bs)
            bs_weights.append(weight)
            log_probs.append(log_prob)
        self.uav_weights = np.array(uav_weights)
        self.bs_weights = np.array(bs_weights)
        return uav_directions, actions1, actions2, log_probs, entropies

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
        I2VChannel_with_fastfading = np.transpose(env.V2IChannel_with_fastfading, (1,0,2))
        I2V_Signal = 10 ** ((env.I2V_power_dB - I2VChannel_with_fastfading) / 10)[:,:,0] # 因为不考虑rb问题,带宽直接平分,所以无所谓
        I2V_Rate_without_Bandwidth = np.log2(1 + np.divide(I2V_Signal, env.sig2))
        return I2V_Rate_without_Bandwidth

    def get_U2VRateWithoutBand(self, env):
        U2V_Signal = 10 ** ((env.U2V_power_dB - env.U2VChannel_with_fastfading) / 10)[:,:,0]
        U2V_Rate_without_Bandwidth = np.log2(1 + np.divide(U2V_Signal, env.sig2))
        return U2V_Rate_without_Bandwidth

    def mock_offloading(self, env, states):
        # 通过模仿各个map格子之间的通信,假设V2V矩阵,代表不同格子之间的通信rate,V2U和V2I的要单独计算
        # self.grid_distance代表了40x40的map中作为vector第i个网格到达第j个网格的距离.
        # self.V2V_grid_rate
        assigned_map = (states[0, -1, :, :] * 5).astype(np.int64)
        cpu_map = states[0, 0, :, :] * 2.3
        req_map = states[0, 1, :, :] * 10
        tot_sec = 0
        tot_task_cnt = 1
        tot_succ_task_cnt = 0
        regional_sec = []
        region_task_cnt = []
        for i in range(env.n_UAV+env.n_BS):
            ass = assigned_map == i
            tidx = (req_map>0) * ass
            rows, cols = np.where(tidx)
            serving_indices = rows * self.map_width + cols
            tidx2 = (cpu_map>0) * ass
            rows, cols = np.where(tidx2)
            task_indices = rows * self.map_width + cols
            region_cpu = np.sum(cpu_map[tidx2])
            region_req = np.sum(req_map[tidx])
            n_kt = np.ceil(len(req_map[tidx])).__int__() # 有req的网格的数量,作为任务车辆
            n_jt = np.ceil(len(cpu_map[tidx2])).__int__() # 有cpu的网格的数量,作为服务车辆
            task_cnt = n_kt
            n_tt = 20
            sec, succ_task_cnt = 0, 0
            if n_kt > 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                upt[0,:] = np.reshape(req_map[tidx], -1) * (env.task_data_max + env.task_data_min) / 2
                reqt[0,:] = np.reshape(req_map[tidx], -1) * (env.task_cpu_min + env.task_cpu_max) / 2
                taut[0,:] = np.reshape(req_map[tidx], -1) * (env.task_ddl_min + env.task_ddl_max) * 5
                gt[:,:,0] = self.V2V_grid_rate[np.ix_(serving_indices, task_indices)]
                for gti in range(n_tt):
                    gt[:,:,gti] = gt[:,:,0]
                sec, succ_task_cnt = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = 20, F_jt = region_cpu / (n_jt+1e-5), upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = 0.1 , Bt = 10/6)
                # sec = task_cnt * region_req * (env.task_cpu_min + env.task_cpu_max) / 2 / (region_cpu+1e-3) # avg_latency * task_cnt
            regional_sec.append(sec)
            # regional_sec.append(sec + (task_cnt - succ_task_cnt) * 2)
            tot_sec += sec
            region_task_cnt.append(region_req)
            tot_task_cnt += task_cnt
            tot_succ_task_cnt += succ_task_cnt
        ass = np.where(assigned_map == -1)
        req = req_map[ass]
        tot_sec += 20 * sum(req)
        return tot_sec / tot_task_cnt, tot_succ_task_cnt / tot_task_cnt, regional_sec, region_task_cnt, np.sum(req_map)/(np.sum(cpu_map)+1)
    def mock_offloading2(self, env, states):
        # 对每个有请求的格子，如果在设备i下，根据资源远近加权计算自身得失。返回的reward为全局得失的总和
        assigned_map = (states[0, -1, :, :] * 5).astype(np.int64)
        cpu_map = states[0, 0, :, :] * 2.5
        req_map = states[0, 1, :, :] * 10
        grid_length = 2000 // self.map_height
        region_value = []
        tidx = (req_map>0) 
        rrows, rcols = np.where(tidx)
        tidx2 = (cpu_map>0)
        crows, ccols = np.where(tidx2)
        vposx = []
        vposy = []
        for i in range(env.n_UAV+env.n_BS):
            if i < env.n_UAV:
                posx, posy = env.UAVs[i].position
                posx, posy = int(posx / 2000 * self.map_width), int(posy / 2000 * self.map_width)
            else:
                posx, posy = env.BSs[i-env.n_UAV].position
                posx, posy = int(posx / 2000 * self.map_width), int(posy / 2000 * self.map_width)
            ass = assigned_map == i
            tidx = (req_map>0) * ass
            rrows, rcols = np.where(tidx)
            tidx2 = (cpu_map>0) * ass
            crows, ccols = np.where(tidx2)
            vposx = []
            vposy = []
            weight = []
            for rrow, rcol in zip(rrows, rcols):
                v = 0
                for crow, ccol in zip(crows, ccols):
                    distance_squared = ((crow - rrow)*grid_length) ** 2 + ((ccol - rcol)*grid_length) ** 2
                    v += req_map[rrow,rcol] / max(1e-5, 17 - 3.5 * 0.5 * np.log10(distance_squared) + cpu_map[crow, ccol] / 2) 
                weight.append(v)
                vposx.append(rrow)
                vposy.append(rcol)
            vposx = np.array(vposx)
            vposy = np.array(vposy)
            weight = np.array(weight)
            if len(rrows)!= 0:
                weighted_dist = np.sqrt((posx-vposx)**2+(posy-vposy)**2)*weight / max(1,sum(weight))
                weighted_dist_sum = np.sum(weighted_dist)
                region_value.append(1/max(1,(weighted_dist_sum)))
            else:
                region_value.append(0)
                # region_value.append(1 / np.sqrt(min(1,(posx-global_centx)**2+(posy-global_centy)**2)))
        return region_value
    def act_offloading(self, env, states, state_vector, computeWHO = True):
        # 根据车辆的位置,以及assigned到哪个设备,划分子区域,然后使用WHO算法进行offloading
        if computeWHO:
            return self.mock_offloading(env, states)
        return self.mock_offloading2(env, states)
        grid_length = 2000 // self.map_height
        vehicle_list = env.vehicle_by_index
        V2VRate = self.get_V2VRateWithoutBand(env)
        V2URate = self.get_V2URateWithoutBand(env)
        U2VRate = self.get_U2VRateWithoutBand(env)
        V2IRate = self.get_V2IRateWithoutBand(env)
        I2VRate = self.get_I2VRateWithoutBand(env)
        regional_sec = []
        region_task_cnt = []
        tot_sec = 0
        tot_task_cnt = 1
        tot_succ_task_cnt = 1
        tot_serving = 0
        for vidx, veh in enumerate(vehicle_list):
            for vehi in range(env.n_Veh):
                if env.distance(veh.position, vehicle_list[vehi].position) >= 50:
                    V2VRate[vidx, vehi] = 0
                    continue
        for i in range(env.n_UAV):
            assigned_uav_i = (states[i, -1, :, :]*5).astype(np.int64)
            task_cnt = 0
            n_kt = 0
            n_jt = 2 # 直接假设无人机等价于两个车,并且放在最后
            n_tt = 20
            task_lambda_lst = []
            veh_ids = []
            sv_ids = []
            tv_ids = []
            sec = 0
            succ_task_cnt = 0
            for vidx, veh in enumerate(vehicle_list):
                dx, dy = int(veh.position[0] / grid_length), int(veh.position[1]/ grid_length)
                # 保证dx,dy在0~39
                dx = min(dx, self.map_width-1)
                dy = min(dy, self.map_height-1)
                if assigned_uav_i[dx, dy] == i:
                    veh_ids.append(vidx)
                    intRate = V2URate[vidx, i] * U2VRate[i, :] / (V2URate[vidx, i] + U2VRate[i, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                    else:
                        n_kt += 1
                        task_lambda_lst.append(veh.task_lambda)
                        tv_ids.append(vidx)
            tot_serving += n_jt
            if n_kt != 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                # 遍历task lambda,每一个task lambda都代表了1秒内的任务数量.我们是2秒,然后使用2*lambda的泊松分布对于不同的时隙进行采样
                for tt in range(1):
                    for k in range(n_kt):
                        # hasTask = np.random.poisson(task_lambda_lst[k]) > 0 # 假设每个时隙最多只有一个任务
                        # if hasTask:
                        task_cnt += task_lambda_lst[k]
                        upt[tt, k] = (env.task_data_min+env.task_data_max)/2*task_lambda_lst[k]
                        reqt[tt, k] = (env.task_cpu_min+env.task_cpu_max)/2*task_lambda_lst[k]
                        taut[tt, k] = (env.task_ddl_min+env.task_ddl_max)*5
                if task_cnt != 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    for ttt in range(2):
                        gt[:len(tv_ids), -2+ttt, 0] = V2URate[tv_ids, i]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    sec, succ_task_cnt = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = 20, F_jt = 2.3, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = 0.1 , Bt = 10/6)
            regional_sec.append(sec + (task_cnt - succ_task_cnt) * 2)
            tot_sec += regional_sec[-1]
            region_task_cnt.append(task_cnt)
            tot_task_cnt += task_cnt
            tot_succ_task_cnt += succ_task_cnt
        for bsi in range(env.n_BS):
            task_cnt = 0
            i = bsi + env.n_UAV
            assigned_uav_i = (states[i, -1, :, :]*5).astype(np.int64)
            n_kt = 0
            n_jt = 10 # 基站等于10个车,但是速度要加
            n_tt = 20
            task_lambda_lst = []
            veh_ids = []
            tv_ids = []
            sv_ids = []
            for vidx, veh in enumerate(vehicle_list):
                dx, dy = int(veh.position[0] / grid_length), int(veh.position[1]/grid_length)
                # 保证dx,dy在0~39
                dx = min(dx, self.map_height-1)
                dy = min(dy, self.map_height-1)
                if assigned_uav_i[dx, dy] == i:
                    veh_ids.append(vidx)
                    intRate = V2IRate[vidx, bsi] * I2VRate[bsi, :] / (V2IRate[vidx, bsi] + I2VRate[bsi, :])
                    idxs = intRate[:] > V2VRate[vidx, :]
                    V2VRate[vidx, idxs] = intRate[idxs]
                    if veh.serving:
                        n_jt += 1
                        sv_ids.append(vidx)
                    else:
                        n_kt += 1
                        task_lambda_lst.append(veh.task_lambda)
                        tv_ids.append(vidx)
            sec = 0
            succ_task_cnt = 0
            tot_serving += n_jt
            if n_kt != 0: # 没有任务,则不需要offloading
                upt = np.zeros((n_tt, n_kt))
                reqt = np.zeros((n_tt, n_kt))
                taut = np.zeros((n_tt, n_kt), dtype=np.int32)
                gt = np.zeros((n_kt, n_jt, n_tt)) 
                # 遍历task lambda,每一个task lambda都代表了1秒内的任务数量.我们是2秒,然后使用2*lambda的泊松分布对于不同的时隙进行采样
                for tt in range(1):
                    for k in range(n_kt):
                        # hasTask = np.random.poisson(task_lambda_lst[k] * 0.1) > 0 # 假设每个时隙最多只有一个任务
                        # if hasTask:
                        task_cnt += task_lambda_lst[k]
                        upt[tt, k] = (env.task_data_min+env.task_data_max)/2*task_lambda_lst[k]
                        reqt[tt, k] = (env.task_cpu_min+env.task_cpu_max)/2*task_lambda_lst[k]
                        taut[tt, k] = (env.task_ddl_min+env.task_ddl_max)*5
                
                if task_cnt != 0:
                    # 根据veh_ids,从V2VRate中提取出对应的值作为gt
                    gt[:len(tv_ids), :len(sv_ids), 0] = V2VRate[np.ix_(tv_ids, sv_ids)]
                    
                    for ttt in range(10):
                        gt[:len(tv_ids), -10+ttt, 0] = V2IRate[tv_ids, bsi]
                    for gti in range(n_tt):
                        gt[:,:,gti] = gt[:,:,0]
                    sec, succ_task_cnt = WHO(n_kt = n_kt, n_jt = n_jt, n_tt = 20, F_jt = 2.3, upt = upt, reqt = reqt, taut = taut, gt = gt , dtt = 0.1 , Bt = 10/6, isBS=True)
            regional_sec.append(sec + (task_cnt - succ_task_cnt) * 20)
            tot_sec += regional_sec[-1]
            region_task_cnt.append(task_cnt)
            tot_task_cnt += task_cnt
            tot_succ_task_cnt += succ_task_cnt
        req_map = states[0, 1, :,:]*10
        failed_cnt = np.sum(req_map[np.where(assigned_uav_i==-1)])
        tot_sec += failed_cnt * 20
        tot_task_cnt += failed_cnt
        return tot_sec / tot_task_cnt, tot_succ_task_cnt / tot_task_cnt, regional_sec, region_task_cnt, (env.task_data_min+env.task_data_max)/2*tot_task_cnt/(tot_serving*2.3+1)
            

    def store_experiences(self, states, actions1, actions2, rewards, next_states, pos_states, new_pos_states, log_probs, done, weights, new_weights):
        tmp_dict = self.transition_dicts
        for i in range(self.n_agent):
            tmp_dict[i]['states'].append(states)
            tmp_dict[i]['actions1'].append(actions1[i])
            tmp_dict[i]['actions2'].append(actions2[i])
            tmp_dict[i]['rewards'].append((rewards[i]))
            tmp_dict[i]['next_states'].append(next_states)
            tmp_dict[i]['pos_states'].append(pos_states)
            tmp_dict[i]['new_pos_states'].append(new_pos_states)
            tmp_dict[i]['weights'].append(weights)
            tmp_dict[i]['new_weights'].append(new_weights)
            tmp_dict[i]['dones'].append(done)
            tmp_dict[i]['log_probs'].append(log_probs[i])


    def store_experiences_for_weight(self, states, actions_weight, rewards, next_states):
        for i in range(self.n_agent):
            self.transition_dicts[i]['states'].append(states[i])
            self.transition_dicts[i]['actions_weight'].append(actions_weight[i])
            self.transition_dicts[i]['rewards'].append(rewards[i])
            self.transition_dicts[i]['next_states'].append(next_states[i])
            self.transition_dicts[i]['dones'].append(False)

    def update_agent(self):
        if len(self.transition_dicts[0]['states']) < self.buffer_size:
            return
        for i in range(self.n_agent):
            # 随机生成batch_size个随机数，作为batch的索引，范围在0到len(self.transition_dicts[i])-1之间
            # if i < self.n_UAV:
            for j in range(3):
                batch_idx = np.arange(200) + np.random.randint(2) * 200
                loss_a, loss_c, log_alpha = self.agents_uav[0].update(self.transition_dicts[i], batch_idx, self.transition_dicts, i, traverse=j%3)
                self.actor_loss.append(np.mean(loss_a))
                self.critic_loss.append(np.mean(loss_c))
                self.log_alpha.append(log_alpha)