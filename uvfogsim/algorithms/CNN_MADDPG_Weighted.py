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
from .CNN_MADDPG2 import CNN_MADDPG
import os
from pyswarms.single.global_best import GlobalBestPSO
from .VDN import VDN_Agent


class CNN_MADDPG_Weighted(Base_Algorithm_Module):
    def __init__(self, env, training = True, model_path = None):
        super().__init__()
        n_hiddens = 256 # 隐含层数量
        actor_lr = 3e-4
        critic_lr = 5e-4
        gamma = 0.8
        lmbda = 0.9
        eps = 0.2
        self.device = 'cuda:2'
        device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.max_weight = 2.5 # 权重代表距离的额外加减部分
        self.max_weight_bs = 2.5
        self.n_agent = env.n_UAV + env.n_BS
        #  + env.n_BS
        self.map_width = 40
        self.map_height = 40
        self.n_channel = 5
        self.uav_weights = np.zeros(env.n_UAV)
        self.bs_weights = np.zeros(env.n_BS)
        self.n_UAV = env.n_UAV
        self.n_BS = env.n_BS
        self.actor_loss = deque([], maxlen=30)
        self.critic_loss = deque([], maxlen=30)
        self.buffer_size = int(5000)
        self.batch_size = 512
        self.transition_dicts = [{'actions_weight':deque([],maxlen=self.buffer_size),'pos_states':deque([],maxlen=self.buffer_size),'new_pos_states':deque([],maxlen=self.buffer_size),'states': deque([],maxlen=self.buffer_size),'actions1': deque([],maxlen=self.buffer_size),'actions2': deque([],maxlen=self.buffer_size),'actions3': deque([],maxlen=self.buffer_size),'next_states': deque([],maxlen=self.buffer_size),'rewards': deque([],maxlen=self.buffer_size),'dones': deque([],maxlen=self.buffer_size),'log_probs': deque([],maxlen=self.buffer_size)} for i in range(self.n_agent)]
        self.n_actions = np.array([5, 5]) # 悬停或者上下左右移动，另一个符号控制权重等级
        self.agents_uav = [CNN_MADDPG(input_states = [self.map_width, self.map_height, self.n_channel],
            n_hiddens = n_hiddens,
            n_actions = [6,6],
            actor_lr = actor_lr,
            critic_lr = critic_lr,
            n_agent = self.n_agent,
            lmbda = lmbda,
            eps = eps,
            gamma = gamma,
            device = device) for _ in range(self.n_UAV+self.n_BS)]
        if not training:
            self.load_agents(model_path)
            for agents_uav in self.agents_uav:
                agents_uav.epislon = 0.1
    def load_agents(self, model_path):
        # self.agents_weight.load_agents(model_path, self.device)
        for idx, agents_uav in enumerate(self.agents_uav):
            agents_uav.actor.load_state_dict(torch.load(model_path + f'/actor_uav_{idx}' + '.pth', map_location=self.device))
            agents_uav.critic.load_state_dict(torch.load(model_path + f'/critic_uav_{idx}' + '.pth', map_location=self.device))
    def save_agents(self, model_path):
        # self.agents_weight.save_agents(model_path)
        for idx, agents_uav in enumerate(self.agents_uav):
            torch.save(agents_uav.actor.state_dict(), model_path + f'/actor_uav_{idx}' + '.pth')
            torch.save(agents_uav.critic.state_dict(), model_path + f'/critic_uav_{idx}' + '.pth')

    def act_mobility(self, env, pos_states, states = None):
        # 根据全局车辆位置，以及全局计算资源需求（计算task queue的总和），归一化变成2D的map
        # states = env.get_states(vehicle_ids, vehicle_positions, 50, 50)
        uav_directions = []
        uav_weights = []
        bs_weights = []
        actions1 = []
        actions2 = []
        log_probs = []
        # 输入4个map给每一个agent的CNN，再给到MADRL的每一个agent，得到一个action表示无人机的移动方向和权重
        for i in range(env.n_UAV):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i]
            state_i = np.expand_dims(state_i, axis=0)
            pos_state = np.expand_dims(pos_states[i], axis = 0)
            action1, action2, log_prob = self.agents_uav[i].take_action(state_i, i, pos_state) 
            actions1.append(action1)
            actions2.append(action2)
            # action1表示移动方向，action2表示移动速度，action3表示权重，范围都是0-X
            # 针对移动方向，将action1转换成弧度
            action1 = action1 / self.n_actions[0] * 2 * np.pi
            action2 = action2 / self.n_actions[1] * self.max_weight
            uav_directions.append(action1)
            uav_weights.append(action2)
            log_probs.append(log_prob.cpu().detach().numpy())
        for i in range(env.n_BS):
            # 在maps前面多加1维度，表示batch_size = 1
            state_i = states[i+env.n_UAV]
            state_i = np.expand_dims(state_i, axis=0)
            pos_state = np.expand_dims(pos_states[i], axis = 0)
            selfidx = i+env.n_UAV
            action2, log_prob = self.agents_uav[i+self.n_UAV].take_action(state_i, selfidx, pos_state) 
            action1 = 0 # 用同一个模型来采取行为，但是采取的都是0
            actions1.append(action1)
            actions2.append(action2)
            action2 = action2 / self.n_actions[1] * self.max_weight_bs
            bs_weights.append(action2)
            log_probs.append(log_prob.cpu().detach().numpy())
        # bs_weights = np.ones(env.n_BS) * self.max_weight
        self.uav_weights = np.array(uav_weights)
        self.bs_weights = np.array(bs_weights)
        return uav_directions, actions1, actions2, log_probs

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
        vehicle_by_index = env.vehicle_by_index
        # 权重公式是： - gain - weight
        v2u_gains = np.mean(env.V2UChannel_with_fastfading, axis = 2) / 100 # 信道平均增益，shape = (n_Veh, n_UAV) 
        v2i_gains = np.mean(env.V2IChannel_with_fastfading, axis = 2) / 100
        u_wei = np.repeat(self.uav_weights[np.newaxis, :], env.n_Veh, axis = 0)
        v2u_gains_with_weight = v2u_gains - u_wei
        i_wei = np.repeat(self.bs_weights[np.newaxis, :], env.n_Veh, axis = 0)
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

    def store_experiences(self, states, actions1, actions2, rewards, next_states, pos_states, new_pos_states, log_probs):
        for i in range(self.n_agent):
            self.transition_dicts[i]['states'].append(states[i])
            self.transition_dicts[i]['actions1'].append(actions1[i])
            self.transition_dicts[i]['actions2'].append(actions2[i])
            self.transition_dicts[i]['rewards'].append(rewards[i])
            self.transition_dicts[i]['next_states'].append(next_states[i])
            self.transition_dicts[i]['pos_states'].append(pos_states[i])
            self.transition_dicts[i]['new_pos_states'].append(new_pos_states[i])
            self.transition_dicts[i]['dones'].append(False)
            self.transition_dicts[i]['log_probs'].append(log_probs[i])

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
        for selfidx in range(self.n_UAV+self.n_BS):
            batch_idx = np.random.choice(len(self.transition_dicts), self.batch_size)
            loss_a, loss_c = self.agents_uav[selfidx].update(self.transition_dicts, batch_idx, selfidx)
            self.critic_loss.append(loss_c.mean())
            self.actor_loss.append(loss_a.mean())
        