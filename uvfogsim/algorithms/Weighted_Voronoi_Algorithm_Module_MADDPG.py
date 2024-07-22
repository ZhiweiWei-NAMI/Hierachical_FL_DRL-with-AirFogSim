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
from .CNN_MADDPG import MADDPG
import os


class Weighted_Voronoi_Algorithm_Module_MADDPG(Base_Algorithm_Module):
    def __init__(self, env, training = True, model_path = None):
        super().__init__()
        self.map_width = 50
        self.map_height = 50
        n_hiddens = 128  # 隐含层数量
        actor_lr = 2e-4
        critic_lr = 1e-3
        gamma = 0.9
        lmbda = 0.97
        eps = 0.2
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.max_weight = 100 # 权重不可以是负数，0代表只作为中继的意愿，0~1代表计算的意愿
        self.n_agent = env.n_UAV
        self.n_UAV = env.n_UAV
        self.n_BS = env.n_BS
        input_states = 231
        self.uav_weights = np.zeros(self.n_UAV)
        self.bs_weights = np.zeros(self.n_BS)
        self.actor_loss = deque([], maxlen = 3000)
        self.critic_loss = deque([], maxlen = 3000)
        self.buffer_size = int(1e3)
        self.batch_size = 512
        self.transition_dicts = [{'states': deque([],maxlen=self.buffer_size),'actions1': deque([],maxlen=self.buffer_size),'actions2': deque([],maxlen=self.buffer_size),'actions3': deque([],maxlen=self.buffer_size),'next_states': deque([],maxlen=self.buffer_size),'rewards': deque([],maxlen=self.buffer_size),'dones': deque([],maxlen=self.buffer_size),'log_probs': deque([],maxlen=self.buffer_size)} for i in range(env.n_UAV)]
        self.n_actions = np.array([4, 3, 5]) # 平面方向，每15度一个动作，共24个动作，速度一共有10档，共10个动作，权重调节一共有10档，共10个动作，共28个动作
        self.agents_uav = MADDPG(input_states = input_states,
            n_hiddens = n_hiddens,
            n_actions = [5, 4, 6],
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            n_agent = self.n_agent,
            lmbda = lmbda,
            eps = eps,
            gamma = gamma,
            device = device, )
        if not training:
            self.load_agents(model_path)
            self.agent.epislon = 0
            
            
    def load_agents(self, model_path):
        # self.agents_weight.load_agents(model_path, self.device)
        for idx, agent_uav in enumerate(self.agents_uav.actors):
            agent_uav.load_state_dict(torch.load(model_path + f'/actor_uav_{idx}' + '.pth', map_location=self.device))
            
        self.agents_uav.critic.load_state_dict(torch.load(model_path + f'/critic_uav' + '.pth', map_location=self.device))

    def save_agents(self, model_path):
        # self.agents_weight.save_agents(model_path)
        for idx, agent_uav in enumerate(self.agents_uav.actors):
            torch.save(agent_uav.state_dict(), model_path + f'/actor_uav_{idx}' + '.pth')
        torch.save(self.agents_uav.critic.state_dict(), model_path + f'/critic_uav' + '.pth')

    def get_assigned(self, env):
        # 根据无人机，车辆，基站的位置，计算weighted voronoi，返回v2u_assigned和v2i_assigned。
        v2u_assigned = np.zeros((env.n_Veh, env.n_UAV))
        v2i_assigned = np.zeros((env.n_Veh, env.n_BS))
        # weighted voronoi的公式
        for idx, veh in enumerate(env.vehicle_by_index):
            min_dist = None
            min_j = 0
            min_cer_dist = None
            min_cj = 0
            uav_or_bs_cer = 0
            uav_or_bs = 0
            for j, uav in enumerate(env.UAVs):
                dist = env.v2u_distance[idx,j] - self.uav_weights[j] 
                dd =  env.v2u_distance[idx,j]
                # dist = env.V2UChannel_with_fastfading[idx, j, 0] - self.uav_weights[j] 
                #  * uav.CPU_frequency
                if dd <= 300 and (min_dist is None or (min_dist > dist)):
                    min_dist = dist
                    min_j = j
                    uav_or_bs = 1
                if (min_cer_dist is None or min_cer_dist > dd) and dd <= 300:
                    min_cer_dist = dd
                    uav_or_bs_cer = 1
                    min_cj = j
            for k, bs in enumerate(env.BSs):
                dist = env.v2i_distance[idx, k] - self.bs_weights[k] 
                dd = env.v2i_distance[idx,k]
                # dist = env.V2IChannel_with_fastfading[idx, k, 0] - self.bs_weights[k] 
                #  * bs.CPU_frequency
                if dd <= 600 and (min_dist is None or (min_dist > dist)):
                    min_dist = dist
                    min_j = k
                    uav_or_bs = 2
                if (min_cer_dist is None or min_cer_dist > dd) and dd <=600:
                    min_cer_dist = dd
                    min_cj = k
                    uav_or_bs_cer = 2
            if uav_or_bs == 1:
                v2u_assigned[idx, min_j] = 1
            elif uav_or_bs == 2:
                v2i_assigned[idx, min_j] = 1
            elif uav_or_bs_cer == 1:
                v2u_assigned[idx, min_cj] = 1
            elif uav_or_bs_cer == 2:
                v2i_assigned[idx, min_cj] = 1
            else:
                print("!")

        return v2u_assigned, v2i_assigned

    def act_mobility(self, env, vehicle_ids, vehicle_positions, states = None):
        # 根据全局车辆位置，以及全局计算资源需求（计算task queue的总和），归一化变成2D的map
        # states = env.get_states(vehicle_ids, vehicle_positions, 50, 50)
        uav_directions = []
        uav_speeds = []
        uav_weights = []
        bs_weights = []
        actions1 = []
        actions2 = []
        actions3 = []
        log_probs = []
        # 输入4个map给每一个agent的CNN，再给到MADRL的每一个agent，得到一个action表示无人机的移动方向和权重
        for i in range(env.n_UAV):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i]
            state_i = np.expand_dims(state_i, axis=0)
            other_states = np.expand_dims(states, axis = 0)
            selfidx = np.expand_dims([i], axis = 0)
            action1, action2, action3, log_prob = self.agents_uav.take_action(state_i, i) 
            actions1.append(action1)
            actions2.append(action2)
            actions3.append(action3)
            # action1表示移动方向，action2表示移动速度，action3表示权重，范围都是0-X
            # 针对移动方向，将action1转换成弧度
            action1 = (action1) / self.n_actions[0] * 2 * np.pi
            action2 = action2 / self.n_actions[1] * env.uav_max_speed
            action3 = action3 / self.n_actions[2] * self.max_weight
            uav_directions.append(action1)
            uav_speeds.append(action2)
            uav_weights.append(action3)
            log_probs.append(log_prob.cpu().detach().numpy())
        for i in range(env.n_BS):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i+env.n_UAV]
            state_i = np.expand_dims(state_i, axis=0)
            other_states = np.expand_dims(states, axis = 0)
            selfidx = np.expand_dims([i+env.n_UAV], axis = 0)
            action1, action2, action3, log_prob = 0,0,1,0
            actions1.append(action1)
            actions2.append(action2)
            actions3.append(action3)
            action3 = action3 / self.n_actions[2] * self.max_weight
            bs_weights.append(action3)
            log_probs.append(log_prob)
        # bs_weights = np.ones(env.n_BS) * self.max_weight
        self.uav_weights = np.array(uav_weights)
        self.bs_weights = np.array(bs_weights)
        return uav_directions, uav_speeds, actions1, actions2, actions3, log_probs


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

    def store_experiences(self, states, actions1, actions2, actions3, rewards, next_states, log_probs):
        for i in range(self.n_agent):
            self.transition_dicts[i]['states'].append(states[i])
            self.transition_dicts[i]['actions1'].append(actions1[i])
            self.transition_dicts[i]['actions2'].append(actions2[i])
            self.transition_dicts[i]['actions3'].append(actions3[i])
            self.transition_dicts[i]['rewards'].append(rewards[i])
            self.transition_dicts[i]['next_states'].append(next_states[i])
            self.transition_dicts[i]['dones'].append(False)
            self.transition_dicts[i]['log_probs'].append(log_probs[i])
    
    def update_agent(self):
        # for i in range(5):
        if len(self.transition_dicts[0]['states']) < self.buffer_size:
            return
        batch_idx = np.random.choice(len(self.transition_dicts), self.batch_size)
        loss_a, loss_c = self.agents_uav.update(self.transition_dicts, batch_idx)
        self.critic_loss.append(loss_c.mean())
        self.actor_loss.append(loss_a.mean())
        
