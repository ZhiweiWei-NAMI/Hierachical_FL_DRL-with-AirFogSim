# -*- encoding: utf-8 -*-
'''
@File    :   U2IChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import numpy as np
import math
class U2IChannel: 
    # U2I仿真信道
    def __init__(self, n_RB, n_BS, n_UAV, hei_UAV, BS_positions):
        '''V2I只存在于一个BS范围内的V2I channel'''
        self.h_bs = 25 # 基站高度25m
        self.h_ms = hei_UAV
        self.hei_UAV = hei_UAV
        self.Decorrelation_distance = 50 
        self.shadow_std = 8
        self.n_BS = n_BS
        self.n_UAV = n_UAV
        self.n_RB = n_RB
        self.fc = 2
        self.BS_positions = BS_positions
        self.update_shadow([])

    def update_positions(self, UAV_positions):
        self.UAV_positions = UAV_positions
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.UAV_positions),len(self.BS_positions)))
        for i in range(len(self.UAV_positions)):
            for j in range(len(self.BS_positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.UAV_positions[i], self.BS_positions[j])

    def update_shadow(self, delta_distance_list):
        if len(delta_distance_list) == 0:  # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_UAV, self.n_BS))
        else: 
            delta_distance = np.repeat(delta_distance_list[:,np.newaxis], self.n_BS, axis=1)
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.Decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_BS))/10)))

    def update_fast_fading(self):
        gaussian1 = np.random.normal(size=(self.n_UAV, self.n_BS, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_UAV, self.n_BS, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

    
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d_2 = math.hypot(d1,d2)+0.001 
        if d_2 > 4000:
            return 1000 # 只允许2d距离在4000米以内的传输，超过了就置为极小值
        d_3 = math.hypot(d_2, self.hei_UAV) + 0.001
        # 计算LoS概率
        log_hu = math.log10(self.hei_UAV)
        p1 = 4300 * log_hu - 3800
        d1 = max(460 * log_hu - 700, 18)
        poss_los = 0
        if d_2 <= d1:
            poss_los = 1
        else:
            poss_los = d1/d_2 + math.exp(-d_2 / p1) * (1-d1/d_2)
        if self.hei_UAV > 100:
            poss_los = 1
        def PL_NLOS(d):
            return -17.5 + (46-7*log_hu) * math.log10(d) + 20 * math.log10(40 * math.pi * self.fc / 3)
        def PL_LOS(d):
            return 28.0 + 22 * math.log10(d ) + 20 * math.log10(self.fc)
        
        return poss_los * PL_LOS(d_3) + (1-poss_los) * PL_NLOS(d_3)
