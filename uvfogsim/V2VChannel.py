# -*- encoding: utf-8 -*-
'''
@File    :   V2VChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import numpy as np
import math
class V2VChannel:
    def __init__(self, n_Veh, n_RB):
        '''RB数量不变，车辆数量会变，参数设置来源3GPP TR36.885-A.1.4-1'''
        self.t = 0
        self.h_bs = 1.5 # 车作为BS的高度 
        self.h_ms = 1.5 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 10
        self.shadow_std = 3 # shadow的标准值
        self.n_Veh = n_Veh # 车辆数量
        self.n_RB = n_RB # RB数量
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        self.update_shadow([])

    def update_positions(self, positions):
        # 把字典转换成列表，通过vid_index来根据index排序
        self.positions = positions

    def update_pathloss(self):
        self.update_pathloss_matrix()
    

    # 考虑到车辆数量变动，需要更新上一时刻的阴影，删除的车辆阴影删除，新增的车辆阴影增加
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.Shadow = np.delete(self.Shadow, index, axis=1)
        self.n_Veh -= 1

    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_Veh))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        new_shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh + 1, 1))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=1)
        self.n_Veh += 1

    def update_shadow(self, delta_distance_list):
        '''输入距离变化，计算阴影变化，基于3GPP的规范'''
        if len(self.Shadow) == 0:
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
        if len(self.Shadow) != len(delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            self.Shadow = Shadow
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        delta_distance_list = np.array(delta_distance_list)
        delta_distance = np.add.outer(delta_distance_list, delta_distance_list)
        if len(delta_distance_list) != 0: 
            exp_term = np.exp(-delta_distance / self.decorrelation_distance)
            sqrt_term = np.sqrt(1 - exp_term**2)
            random_term = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_Veh))
            linear_value1 = 10 ** (self.Shadow / 10)
            linear_value2 = 10 ** (random_term / 10)
            shadow_linear = exp_term * linear_value1 + sqrt_term * linear_value2
            self.Shadow = 10 * np.log10(shadow_linear)
        np.fill_diagonal(self.Shadow, 0)

    def update_fast_fading(self):
        # 生成两个独立的高斯随机变量矩阵
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_Veh, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)
        # 对每一个RB, 将对角线的值设置为0
        for i in range(self.n_RB):
            np.fill_diagonal(self.FastFading[:, :, i], 0)

    def get_path_loss_vectorized(self, d, d1, d2):
        if d.shape[0] == 1:
            PL = np.zeros((1, d.shape[1]))
            return PL

        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)
        
        # PL_Los = np.where(d <= 3, 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5),
        #                 np.where(d < d_bp, 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5),
        #                         40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)))
        PL_Los = 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        n_j = np.maximum(2.8 - 0.0024 * d2, 1.84)
        PL_NLos_d1 = PL_Los + 20 - 12.5 * n_j + 10 * n_j * np.log10(np.maximum(d1, 1e-9)) + 3 * np.log10(self.fc/5)
        PL_NLos_d2 = PL_Los + 20 - 12.5 * n_j + 10 * n_j * np.log10(np.maximum(d2, 1e-9)) + 3 * np.log10(self.fc/5)

        # PL = np.where(np.minimum(d1, d2) < 10, PL_Los, np.minimum(PL_NLos_d1, PL_NLos_d2))
        PL = np.minimum(PL_NLos_d1, PL_NLos_d2)
        return PL

    def update_pathloss_matrix(self):
        if self.n_Veh == 0:
            return
        positions = np.array(self.positions)
        d1_matrix = np.abs(np.repeat(positions[:, np.newaxis, 0], self.n_Veh, axis = 1) - np.repeat(positions[np.newaxis, :, 0], self.n_Veh, axis = 0))
        d2_matrix = np.abs(np.repeat(positions[:, np.newaxis, 1], self.n_Veh, axis = 1) - np.repeat(positions[np.newaxis, :, 1], self.n_Veh, axis = 0))
        d_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001

        self.PathLoss = self.get_path_loss_vectorized(d_matrix, d1_matrix, d2_matrix)
        np.fill_diagonal(self.PathLoss, 0)
    def get_path_loss(self, position_A, position_B):
        '''出自IST-4-027756 WINNER II D1.1.2 V1.2 WINNER II的LoS和NLoS模型'''
        d1 = abs(position_A[0] - position_B[0]) # 单位是km
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
            n_j = max(2.8 - 0.0024*d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 10: # 以10m作为LoS存在的标准
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL