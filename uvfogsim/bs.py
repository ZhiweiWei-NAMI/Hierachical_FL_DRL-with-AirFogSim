# -*- encoding: utf-8 -*-
'''
@File    :   bs.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
'''

from .FogNodeBase import FogNodeBase

class BS(FogNodeBase):
    def __init__(self, idx, start_position, cpu, bs2c_band, init_revenue=0, reputation_score = 100, cheat_possibility = 0):
        super().__init__(idx, start_position, reputation_score, cpu, cheat_possibility)
        self.id = idx
        self.power_capacity = 0 # 当前剩余的电量
        self.bs2c_band = bs2c_band
        self.type_name = 'RSU'
        self.stake = 0 # 股份
        self.total_revenues = init_revenue
        self.task_v_num = 0

    def miner_mine_block(self, block, cost_stake, revenue, cheated):
        assert self.stake >= cost_stake
        self.stake -= cost_stake
        self.total_revenues += revenue
    
    def update_stake(self, time_step):
        self.stake += self.total_revenues * 0.1 * time_step
