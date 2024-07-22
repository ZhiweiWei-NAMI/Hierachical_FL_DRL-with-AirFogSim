# -*- encoding: utf-8 -*-
'''
@File    :   task.py
@Time    :   2023/05/14
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''

import numpy as np
class Task:
    # Task 失败的reason
    FAILURE_REASONS = {
        'OUT_OF_TTI': 1,
        'OUT_OF_DDL': 2,
        'OUT_OF_VEHICLE': 3
    }
    # 具体的失败原因
    OUT_OF_DDL = FAILURE_REASONS['OUT_OF_DDL']
    OUT_OF_TTI = FAILURE_REASONS['OUT_OF_TTI']
    OUT_OF_VEHICLE = FAILURE_REASONS['OUT_OF_VEHICLE']
    # Task的类型
    TASK_TYPES = {
        'CALCULATE_TYPE': 4,
        'VERIFY_TYPE': 5
    }
    # 具体的任务类型
    CALCULATE_TYPE = TASK_TYPES['CALCULATE_TYPE']
    VERIFY_TYPE = TASK_TYPES['VERIFY_TYPE']
        
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, task_id, vehicle, start_time, cpu, data_size, ddl, task_type = CALCULATE_TYPE):
        self.id = task_id
        self.g_veh = vehicle
        self.start_time = start_time
        self.ini_cpu = cpu
        self.ini_data_size = data_size
        self.ddl = ddl
        
        # 基础信息
        self.cpu = cpu # 还需要计算的cpu
        self.trans_delay = 0 # mod time_step就是本时间段内的传输消耗
        self.comp_delay = 0 # 计算时延
        
        # 计算完成后的信息
        self.result_reliable = True # 判断任务计算的结果是否正确，恶意节点会返回错误的结果
        self.is_finished = False # 判断任务是否完成
        self.finish_time = 0 # 记录完成的时间

        # 每个任务的路由过程信息
        self.routing = [self.g_veh] # 记录路由
        self.routed_time_slot = [start_time] # 记录到达路由的时间
        self.transmitted_data = 0
        self.last_transmit_time = start_time # 如果两次传输分配资源之间的间隔太久了(比如5个TTI)，就算失败
        assert task_type in Task.TASK_TYPES.values()
        self.task_type = task_type
        self.is_offloaded = False

    def get_proceeding_ratio(self):
        # 传输看作50%, 计算看作50%
        return (self.transmitted_data / self.ini_data_size + 1 - self.cpu / self.ini_cpu) / 2

    def get_task_utility(self):
        if self.service_delay < self.ddl:
            return np.log(1 + self.ddl - self.service_delay) + self.ini_cpu
        return -self.ini_cpu
    def get_comp_device(self):
        return self.routing[-1]
    def fail_to_transmit(self, cur_time, max_threshold):
        return self.last_transmit_time + max_threshold <= cur_time
    def succeed_compute(self, cur_time, result_reliable = True):
        self.is_finished = True
        self.finish_time = cur_time
        self.result_reliable = result_reliable
    @property
    def service_delay(self):
        return self.trans_delay + self.comp_delay
    
    def exceed_ddl(self, cur_time):
        self.trans_delay = self.routed_time_slot[-1] - self.start_time
        return self.comp_delay + self.trans_delay >= self.ddl
    def wait_to_ddl(self, cur_time):
        return self.start_time + self.ddl <= cur_time
        
    def transmit_to_X(self, X_device, trans_data, cur_time):
        assert X_device is not None
        X_device.consumed_trans += min(trans_data, max(0, self.ini_data_size - self.transmitted_data))
        self.transmitted_data += trans_data
        self.last_transmit_time = cur_time
        if self.transmitted_data >= self.ini_data_size:
            self.transmitted_data = self.ini_data_size
            self.routing.append(X_device) # 添加到已经路由的队列中
            self.routed_time_slot.append(cur_time)
            self.trans_delay = self.routed_time_slot[-1] - self.start_time
            return True
        return False

if __name__ == '__main__':
    print(Task.OUT_OF_DDL)