
from abc import abstractmethod
import numpy as np
from .task import Task

class FogNodeBase:
    UAV_FOG_NODE = 1
    VEHICLE_FOG_NODE = 2
    RSU_FOG_NODE = 3
    def __init__(self, id, position, reputation_score, CPU_frequency, estimate_punished_possibility):
        self.id = id
        self.transmitting = False
        self.receiving = False
        self.position = position
        self.reputation_score = reputation_score
        self.CPU_frequency = CPU_frequency
        self.estimate_punished_possibility = estimate_punished_possibility
        self.task_queue = []
        self.type_name = 'Fog Node'
        self.total_revenues = 0
        self.serving = True
        self.velocity = 0
        self.can_validate = False
        self.consumed_cpu = 0
        self.consumed_trans = 0 # 在task的transmit_to_X中添加

    def calculate_task(self, time_step, resource_allocation = None, cheat_tasks = None):
        '''直接使用mode进行任务计算，或者通过decisions来指定'''
        finished_tasks = []
        is_reliable_list = []
        if len(self.task_queue)>0:
            if resource_allocation is None:
                resource_allocation = np.zeros(len(self.task_queue))
                resource_allocation[0] = 1
            if cheat_tasks is None:
                cheat_tasks = np.zeros(len(self.task_queue), dtype = bool)
            resource_allocation = resource_allocation / np.sum(resource_allocation) # 归一化
            for idx, task in enumerate(self.task_queue.copy()):
                if cheat_tasks[idx]:
                    task.cpu = 0
                    finished_tasks.append(task)
                    is_reliable_list.append(False)
                    self.task_queue.remove(task)
                alloc_res = self.CPU_frequency * time_step * resource_allocation[idx]
                if task.cpu <= alloc_res:
                    self.consumed_cpu += task.cpu
                    task.comp_delay += task.cpu / alloc_res * time_step
                    if task.task_type == Task.VERIFY_TYPE:
                        self.can_validate = True
                    finished_tasks.append(task)
                    is_reliable_list.append(True)
                    self.task_queue.remove(task)
                    task.cpu = 0
                else:
                    self.consumed_cpu += alloc_res
                    task.comp_delay += time_step
                    task.cpu -= alloc_res
        return finished_tasks, is_reliable_list
    
    def get_task_delay(self):
        # 估算当前所有task完成需要多少时间（FIFO）
        total_delay = 0
        for task in self.task_queue:
            total_delay += task.cpu / (self.CPU_frequency)
        return total_delay

    def append_task(self, task):
        # 这里添加任务到队列的逻辑
        self.task_queue.append(task)

    def remove_task_of_vehicle(self, vehicle):
        self.task_queue = [task for task in self.task_queue if task.g_veh != vehicle]

    def update_revenue(self, revenue):
        self.total_revenues += revenue

    def set_reputation_score(self, reputation_score):
        self.reputation_score = reputation_score
        