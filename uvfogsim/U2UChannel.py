# -*- encoding: utf-8 -*-
'''
@File    :   U2UChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


import numpy as np
import math

class U2UChannel:
    def __init__(self, n_RB, n_UAV, hei_UAV):
        # M. M. Azari, G. Geraci, A. Garcia-Rodriguez and S. Pollin, "UAV-to-UAV Communications in Cellular Networks," in IEEE Transactions on Wireless Communications, 2020, doi: 10.1109/TWC.2020.3000303.
        # A Survey of Channel Modeling for UAV Communications
        self.t = 0
        self.h_bs = hei_UAV # 车作为BS的高度 
        self.h_ms = hei_UAV # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 3 # shadow的标准值
        self.n_UAV = n_UAV
        self.n_RB = n_RB # RB数量
        self.update_shadow([])

    def update_positions(self, uav_positions):
        '''更新无人机的位置'''
        self.positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                if i == j: continue
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
                
    def update_shadow(self, delta_distance_list):
        '''输入距离变化，计算阴影变化，基于3GPP的规范'''
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0: 
            self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_UAV))
        else:
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_UAV, self.n_UAV))/10)))
        np.fill_diagonal(self.Shadow, 0)

    def update_fast_fading(self):
        '''快衰落，网上开源代码'''
        gaussian1 = np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_UAV, self.n_UAV, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)
        for i in range(self.n_RB):
            np.fill_diagonal(self.FastFading[:, :, i], 0)

    def get_path_loss(self, position_A, position_B):
        '''U2U path loss'''
        # A Survey of Channel Modeling for UAV Communications
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001 
        alpha_uu = 2.05
        PL = 10 * alpha_uu * math.log10(d)
        return PL