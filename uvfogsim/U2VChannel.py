# -*- encoding: utf-8 -*-
'''
@File    :   U2VChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import numpy as np
import math
class U2VChannel:
    def __init__(self, n_RB, n_UAV, n_Veh, hei_UAV):
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
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_Veh))
        self.update_shadow([], [])
    
    
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=1)
        self.n_Veh -= 1
        
    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, 1))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=1)
        self.n_Veh += 1



    def update_positions(self, veh_positions, uav_positions):
        '''更新车辆和无人机的位置'''
        self.veh_positions = veh_positions
        self.uav_positions = uav_positions
        
    def update_pathloss(self):
        if self.n_Veh == 0:
            return
        self.PathLoss = self.get_path_loss_matrix(self.veh_positions, self.uav_positions)
        
                
    def update_shadow(self, veh_delta_distance_list, uav_delta_distance_list):
        if len(self.Shadow) != len(veh_delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化（只会增加）
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_Veh))
            self.Shadow = Shadow
        delta_distance = np.zeros((len(uav_delta_distance_list), len(veh_delta_distance_list)))
        veh_delta_distance_list = np.array(veh_delta_distance_list)
        uav_delta_distance_list = np.array(uav_delta_distance_list)
        delta_distance = np.add.outer(uav_delta_distance_list, veh_delta_distance_list)
        if len(veh_delta_distance_list) != 0 or len(uav_delta_distance_list) != 0: 
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_Veh))/10)))

    def update_fast_fading(self):
        '''快衰落，网上开源代码'''
        gaussian1 = np.random.normal(size=(self.n_UAV, self.n_Veh, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_UAV, self.n_Veh, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

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
        PL_matrix = P_Los * PL_Los(np.hypot(d_matrix, self.h_bs)) + P_NLos * np.minimum(PL_NLos(self.h_bs, d_matrix), PL_NLos(d_matrix, self.h_bs))
        
        return PL_matrix.T
