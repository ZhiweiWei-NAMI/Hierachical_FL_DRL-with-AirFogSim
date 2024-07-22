# -*- encoding: utf-8 -*-
'''
@File    :   V2UChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import numpy as np
import math
class V2UChannel:
    def __init__(self, n_Veh, n_RB, n_UAV, hei_UAV):
        '''多个vehicle和多个UAV之间的通信信道'''
        self.t = 0
        self.h_bs = hei_UAV
        self.h_ms = 1.5 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 8 # shadow的标准值
        self.n_Veh = n_Veh # 车辆数量
        self.n_UAV = n_UAV # 无人机数量
        self.n_RB = n_RB # RB数量
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_UAV))
        self.update_shadow([], [])
    
    
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.n_Veh -= 1
        
    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_UAV))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        # 更新n_Veh
        self.n_Veh += 1



    def update_positions(self, veh_positions, uav_positions):
        '''更新车辆和无人机的位置'''
        self.veh_positions = veh_positions
        self.uav_positions = uav_positions
        
    def update_pathloss(self):
        if self.n_Veh == 0:
            return
        self.PathLoss = self.get_path_loss_matrix(self.veh_positions, self.uav_positions)
        # self.PathLoss = np.zeros(shape=(len(self.veh_positions),len(self.uav_positions)))
        # for i in range(len(self.veh_positions)):
        #     for j in range(len(self.uav_positions)):
        #         self.PathLoss[i][j] = self.get_path_loss(self.veh_positions[i], self.uav_positions[j])
                
    def update_shadow(self, veh_delta_distance_list, uav_delta_distance_list):
        if len(self.Shadow) != len(veh_delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化（只会增加）
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_UAV))
            self.Shadow = Shadow
        delta_distance = np.zeros((len(veh_delta_distance_list), len(uav_delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance[0])):
                delta_distance[i][j] = veh_delta_distance_list[i] + uav_delta_distance_list[j]
        if len(veh_delta_distance_list) != 0 or len(uav_delta_distance_list) != 0: 
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_UAV))/10)))

    def update_fast_fading(self):
        '''快衰落，网上开源代码'''
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_UAV, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_UAV, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

    @staticmethod
    def calculate_path_loss(distance, h_bs=100, fc=2):
        # distance: the distance between UAV and vehicle
        # h_bs: the height of UAV
        # fc: the frequency of UAV
        
        def PL_Los(d):
            return 30.9 + (22.25 - 0.5 * np.log10(h_bs)) * np.log10(d) + 20 * np.log10(fc/5)
        
        def PL_NLos(d1, d2):
            return np.maximum(PL_Los(d1), 32.4 + (43.2 - 7.6 * np.log10(h_bs)) * np.log10(d2) + 20 * np.log10(fc/5))
        
        D_H = np.sqrt(distance**2 + h_bs**2)
        d_0 = np.maximum((294.05 * np.log10(h_bs) - 432.94), 18)
        p_1 = 233.98 * np.log10(h_bs) - 0.95

        P_Los = np.where(D_H <= d_0, 1.0, d_0 / D_H + np.exp(-(D_H / p_1) * (1 - (d_0 / D_H))))
        P_Los = np.clip(P_Los, 0, 1)

        P_NLos = 1 - P_Los
        PL = P_Los * PL_Los(np.hypot(distance, h_bs)) + P_NLos * np.minimum(PL_NLos(h_bs, distance), PL_NLos(distance, h_bs))

        return PL


    def get_path_loss_matrix(self, veh_positions, uav_positions):
        veh_positions = np.array(veh_positions)
        uav_positions = np.array(uav_positions)
        
        d1_matrix = np.abs(np.repeat(veh_positions[:, np.newaxis, 0], self.n_UAV, axis=1) - uav_positions[:, 0])
        d2_matrix = np.abs(np.repeat(veh_positions[:, np.newaxis, 1], self.n_UAV, axis=1) - uav_positions[:, 1])
        d_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001
        
        def PL_Los(d):
            return 30.9 + (22.25 - 0.5 * np.log10(self.h_bs)) * np.log10(d) + 20 * np.log10(self.fc)
        
        def PL_NLos(d1, d2):
            return np.maximum(PL_Los(d1), 32.4 + (43.2 - 7.6 * np.log10(self.h_bs)) * np.log10(d2) + 20 * np.log10(self.fc))
        
        D_H = np.sqrt(np.square(d_matrix) + np.square(self.h_bs))
        d_0 = np.maximum((294.05 * np.log10(self.h_bs) - 432.94), 18)
        p_1 = 233.98 * np.log10(self.h_bs) - 0.95
        
        P_Los = np.where(D_H <= d_0, 1.0, d_0 / D_H + np.exp(-(D_H / p_1) * (1 - (d_0 / D_H))))
        P_Los = np.clip(P_Los, 0, 1)
        
        P_NLos = 1 - P_Los
        # PL_matrix = P_Los * PL_Los(np.hypot(d_matrix, self.h_bs)) + P_NLos * np.minimum(PL_NLos(self.h_bs, d_matrix), PL_NLos(d_matrix, self.h_bs))
        PL_matrix = PL_Los(np.hypot(d_matrix, self.h_bs))
        return PL_matrix

    def get_path_loss(self, position_A, position_B):
        '''出自3GPP Release 15的LoS和NLoS模型'''
        # R. Zhong, X. Liu, Y. Liu and Y. Chen, "Multi-Agent Reinforcement Learning in NOMA-aided UAV Networks for Cellular Offloading,"in IEEE Transactions on Wireless Communications, doi: 10.1109/TWC.2021.3104633.
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2) + 0.001
        def PL_Los(d):
            return 30.9 + (22.25-0.5*math.log(self.h_bs,10))*math.log10(d) + 20*math.log10(self.fc)
        def PL_NLos(d1, d2):
            return np.max([PL_Los(d1), 32.4+(43.2-7.6*math.log10(self.h_bs))*math.log10(d2)+20*math.log10(self.fc)])
            
        D_H = np.sqrt(np.square(d)+np.square(self.h_bs)) # calculate distance
        d_0 = np.max([(294.05*math.log(self.h_bs,10)-432.94),18])
        p_1 = 233.98*math.log(self.h_bs,10) - 0.95
        if D_H <= d_0:
            P_Los = 1.0
        else:
            P_Los = d_0/D_H + math.exp(-(D_H/p_1)*(1-(d_0/D_H)))

        if P_Los>1:
            P_Los = 1

        P_NLos = 1 - P_Los
        PL = P_Los * PL_Los(math.hypot(d,self.h_bs)) + P_NLos * min(PL_NLos(self.h_bs, d),PL_NLos(d,self.h_bs))
        return PL