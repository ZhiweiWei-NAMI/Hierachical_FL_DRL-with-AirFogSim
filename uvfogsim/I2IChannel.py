# -*- encoding: utf-8 -*-
'''
@File    :   I2IChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
'''


import numpy as np
import math
class I2IChannel:
    def __init__(self, n_RB, n_BS, BS_positions):
        self.t = 0
        self.h_bs = 25 # 车作为BS的高度 
        self.h_ms = 25 # 车作为MS的高度
        self.fc = 2 # carrier frequency
        self.decorrelation_distance = 50
        self.shadow_std = 3 # shadow的标准值
        self.n_BS = n_BS # 车辆数量
        self.n_RB = n_RB # RB数量
        self.positions = BS_positions
        self.update_shadow()

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
                
    def update_shadow(self):
        self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_BS, self.n_BS))

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d < d_bp:
                return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
            else:
                return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 100: # 以100m作为LoS存在的标准
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4 
        return PL