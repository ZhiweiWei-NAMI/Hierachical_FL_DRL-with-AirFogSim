# -*- encoding: utf-8 -*-
'''
@File    :   Base_Algorithm_Module.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''


from abc import ABC, abstractmethod

class Base_Algorithm_Module(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def act_mobility(self, env, vehicle_ids, vehicle_positions):
        '''
        这个函数接收env对象和vehicle_positions对象，并返回uav_directions和uav_speeds。
        '''
        pass

    @abstractmethod
    def act_offloading(self, env):
        '''
        这个函数接收env对象，并返回一系列offload决策和transmit决策。
        '''
        pass
    @abstractmethod
    def act_RB_allocation(self, env):
        '''
        return: activated_offloading_tasks_with_RB_Nos
        activated_offloading_tasks_with_RB_Nos必须是一个二维矩阵，大小为(len(env.offloading_tasks), env.n_RB)
        '''
        pass
    @abstractmethod
    def act_mining_and_pay(self, env):
        '''
        这个函数接收env
        返回执行挖矿的节点索引，其消耗的资源（计算资源或者钱）[使用字典来存储]，挖矿带来的奖励，和是否是恶意挖矿
        这种消耗本质上是该轮次迭代内的资源锁定，比如PoW中
        return [
            {
                'device':device,
                'task_type':'mining-PoW',
                'CPU':3,
                'revenue':0.3,
                'cheated':False
            },
            {
                'device':device,
                'task_type':'mining-PoW',
                'CPU':3,
                'revenue':0.0,
                'cheated':False
            },
            {
                'device':device,
                'task_type':'mining-PoW',
                'CPU':3,
                'revenue':0.3,
                'cheated':False
            }
            ]
        表示v1和v2节点在该轮次内3单位的CPU不可用于其他计算服务，但是v1挖矿成功获得奖励，v2一无所获。
        在PoS中涉及stake的抵押，
        return [
            {
                'device':device,
                'task_type':'mining-PoS',
                'stake':3,
                'revenue':0.3,
                'cheated':False
            },
            {
                'device':device,
                'task_type':'mining-PoS',
                'stake':3,
                'revenue':0.3,
                'cheated':False
            }
            ]
        由于PoS不存在竞争，所以直接指定挖矿节点以及其获得的收益即可.
        如果PoS中存在恶意挖矿，则有对应的惩罚措施，直接在收益中变为负数即可
        '''
        pass

    @abstractmethod
    def act_verification(self, env):
        '''
        接收env，读取env.succeed_tasks任务列表.默认验算过程是从RSU传输的
        返回被验证过的任务，以及对应节点消耗的计算资源，开始计算时间和结束计算时间，获取的奖励
        return [
            {
                'task':task,
                'device':device,
                'CPU':0.5,
                'start_time':3.2,
                'end_time':6.3,
                'revenue':2,
                'validated':True
            },
            {
                ...
            }
        ]
        这些信息也需要被记录在blockchain，所以格式上和task offloading的记录应该是一致的
        '''
        pass

    @abstractmethod
    def act_pay_and_punish(self, env):
        pass

    @abstractmethod
    def act_CPU_allocation(self, env):
        pass