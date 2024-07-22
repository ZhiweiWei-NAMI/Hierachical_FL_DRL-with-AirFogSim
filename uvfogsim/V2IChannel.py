# -*- encoding: utf-8 -*-
'''
@File    :   V2IChannel.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


from matplotlib.pyplot import axis
import numpy as np
import math
class V2IChannel: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_BS, n_RB, BS_positions):
        '''V2I只存在于一个BS范围内的V2I channel'''
        self.h_bs = 30 # 基站高度25m
        self.h_ms = 1.5 
        self.Decorrelation_distance = 50        
        self.BS_positions = BS_positions 
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_BS = n_BS
        self.n_RB = n_RB
        self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_BS))
        self.update_shadow([])

    
    def remove_vehicle_shadow(self, vid, vid_index):
        '''删除车辆，删除车辆的阴影'''
        index = vid_index[vid]
        self.Shadow = np.delete(self.Shadow, index, axis=0)
        self.n_Veh -= 1
        

    def add_vehicle_shadow(self):
        '''增加车辆，增加车辆的阴影'''
        new_shadow = np.random.normal(0, self.shadow_std, size=(1, self.n_BS))
        self.Shadow = np.concatenate((self.Shadow, new_shadow), axis=0)
        # 更新n_Veh
        self.n_Veh += 1

    def update_positions(self, veh_positions):
        # 把字典转换成列表，通过vid_index来根据index排序
        self.positions = veh_positions
        
    @staticmethod
    def calculate_path_loss(distance, h_bs = 25, h_ms = 1.5):
        distance_3D = np.sqrt(distance ** 2 + (h_bs - h_ms) ** 2)
        PL = 128.1 + 37.6 * np.log10(distance_3D / 1000)  # 根据3GPP，距离的单位是km
        return PL

    def update_pathloss(self):
        if self.n_Veh == 0:
            return
        positions = np.array(self.positions)
        BS_positions = np.array(self.BS_positions)
        
        d1_matrix = np.abs(np.repeat(positions[:, np.newaxis, 0], self.n_BS, axis = 1) - BS_positions[:, 0])
        d2_matrix = np.abs(np.repeat(positions[:, np.newaxis, 1], self.n_BS, axis = 1) - BS_positions[:, 1])
        distance_matrix = np.hypot(d1_matrix, d2_matrix) + 0.001
        
        distance_3D = np.sqrt(distance_matrix ** 2 + (self.h_bs - self.h_ms) ** 2)
        self.PathLoss = 128.1 + 37.6 * np.log10(distance_3D / 1000)  # 根据3GPP，距离的单位是km

        # self.PathLoss = np.zeros(shape=(len(self.positions), len(self.BS_positions)))
        # for i in range(len(self.positions)):
        #     for j in range(len(self.BS_positions)):
        #         d1 = abs(self.positions[i][0] - self.BS_positions[j][0])
        #         d2 = abs(self.positions[i][1] - self.BS_positions[j][1])
        #         distance = math.hypot(d1,d2)+0.001
        #         self.PathLoss[i][j] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000) # 根据3GPP，距离的单位是km

    def update_shadow(self, delta_distance_list):
        if len(self.Shadow) != len(delta_distance_list):
            # 1 如果过去一个时间片的车辆数量发生了变化
            Shadow = np.random.normal(0, self.shadow_std, size=(self.n_Veh, self.n_BS))
            self.Shadow = Shadow
        if len(delta_distance_list) != 0:
            delta_distance = np.repeat(delta_distance_list[:,np.newaxis], self.n_BS, axis=1)
            self.Shadow = 10 * np.log10(np.exp(-1*(delta_distance/self.Decorrelation_distance))* (10 ** (self.Shadow / 10)) + np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*(10**(np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_BS))/10)))

    def update_fast_fading(self):
        gaussian1 = np.random.normal(size=(self.n_Veh, self.n_BS, self.n_RB))
        gaussian2 = np.random.normal(size=(self.n_Veh, self.n_BS, self.n_RB))
        # 计算瑞利分布的信道增益
        r = np.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
        # 计算信道增益的平均值
        omega = np.mean(r ** 2)
        # 计算瑞利分布的概率密度函数
        p_r = (2 * r / omega) * np.exp(-r ** 2 / omega)
        # 转换为 dB 单位
        self.FastFading = 10 * np.log10(p_r)

    def get_path_loss(self, positionA, positionB):
        d1 = abs(positionA[0] - positionB[0])
        d2 = abs(positionA[1] - positionB[1])
        distance = math.hypot(d1,d2)+0.001
        return 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000) # 根据3GPP，距离的单位是km