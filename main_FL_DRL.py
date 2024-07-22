# -*- encoding: utf-8 -*-
'''
@File    :   visualization_tkinter.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''

from uvfogsim.vehicle_manager import VehicleManager
from uav_manager import UAVManager
import traci
import subprocess
import sys
from uvfogsim.environment import Environment
import numpy as np
import time
from uvfogsim.tkinter_utils import *
from uvfogsim.algorithms import set_seed
import math
from PPO_MARL_Algorithm import FL_PPO_Agent
from masac_config_parser import parse_arguments_for_MASAC
class DRLEnvironmentWrapper():
    def __init__(self, traci_connection, args):
        self.args = args
        self.traci_connection = traci_connection
        self.environment = None
        self.algorithm_module = None
        self.sumocfg_file = args.sumocfg_path
        self.osm_file_path = args.osm_path
        self.net_file_path = args.net_path
        self.step_per_episode = args.max_steps
        self.iteration_episodes = args.n_episode
        self.n_iteration = args.n_iter
        self.old_vehicle_position_dict = {}
        self.time_step = None
        self.bbox = None
        self.location_bound = None
        self.cur_episode = 0
        self.cur_iter = 0
        self.cur_step = 0
        self.calculate_bbox()

        self.n_veh = args.n_veh
        self.map_data = None
        self.simulation_delay = 0
        
    def calculate_bbox(self):
        # 从net.xml文件中读取地图的bbox,通过parse_location_info函数
        conv_boundary, orig_boundary, proj_params, netOffset = parse_location_info(self.net_file_path)
        orig_boundary = tuple(map(float, orig_boundary.split(',')))
        conv_boundary = tuple(map(float, conv_boundary.split(',')))
        min_x = orig_boundary[0]
        min_y = orig_boundary[1]
        max_x = orig_boundary[2]
        max_y = orig_boundary[3] 
        self.proj_params = proj_params
        self.netOffset = netOffset
        self.bbox = min_x, min_y, max_x, max_y
        self.location_bound = conv_boundary
    def update_vehicle_positions(self, vehicle_ids):
        row_data_dict = {}
        self.veh_positions = []
        for vehicle_id in vehicle_ids:
            x, y = self.traci_connection.vehicle.getPosition(vehicle_id)
            row_data = {}
            row_data['id'] = int(vehicle_id)
            row_data['x'] = x
            row_data['y'] = y
            row_data['angle'] = self.traci_connection.vehicle.getAngle(vehicle_id)
            row_data['speed'] = self.traci_connection.vehicle.getSpeed(vehicle_id)
            row_data['speedFactor'] = self.traci_connection.vehicle.getSpeedFactor(vehicle_id)
            row_data_dict[int(vehicle_id)] = row_data
            self.veh_positions.append((x, y))
        return row_data_dict

        
    def run(self):
        sumo_cmd = ["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", self.sumocfg_file]
        sumo_process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
        self.traci_connection.init(8823, host='127.0.0.1', numRetries=10)
        self.time_step = self.traci_connection.simulation.getDeltaT()
        print("仿真过程中决策的时隙等于SUMO仿真的时隙长度: ", self.time_step)
        env = Environment(args = self.args,draw_it = False, n_UAV = 40, time_step = self.time_step, TTI = self.args.TTI_length)
        env.initialize(self.location_bound, None)
        self.algorithm_module = FL_PPO_Agent(env, self.args)
        self.environment = env
        manager = VehicleManager(self.n_veh, self.net_file_path)
        uav_manager = UAVManager(self.args.n_UAV, self.args.UAV_path_file)
        cnt = 0
        self.traci_connection.simulationStep()
        for iteration in range(self.n_iteration):
            self.cur_iter = iteration
            cnt = 0
            # 0. 每一个iteration开始之前，调整初始化，以及车辆数量或最大服务车辆数量
            env.max_serving_vehicles = self.args.n_serving_veh #+ num_flag * 20
            # self.result_id = np.random.randint(0, 5)
            self.result_id = iteration % 5 + self.args.start_iter_id
            self.algorithm_module.reset_RSU_voronoi_and_no_fly(self.result_id)
            manager.clear_all_vehicles(self.traci_connection)
            tmp_rsu_positions = np.array(self.algorithm_module.rsu_positions) * 5
            env.initialize(self.location_bound, tmp_rsu_positions)
            # manager.turn_off_traffic_lights(self.traci_connection)
            with tqdm(total=self.iteration_episodes * self.step_per_episode, desc=f'iter:{iteration}/{self.n_iteration}') as pbar:
                while cnt <= self.iteration_episodes * self.step_per_episode:
                    pbar.update(1)
                    self.cur_step = cnt % self.step_per_episode
                    self.cur_episode = cnt // self.step_per_episode
                    step_start_time = time.time()
                    # 1 每个time_step进行，控制区域内车辆数量
                    manager.manage_vehicles(self.traci_connection)
                    self.traci_connection.simulationStep() 
                    vehicle_ids = self.traci_connection.vehicle.getIDList()
                    # 1.1 控制车辆在canvas显示的位置
                    row_data_dict = self.update_vehicle_positions(vehicle_ids)
                    # 1.2 更新车辆在仿真内的位置
                    removed_vehicle_ids = env.renew_veh_positions(vehicle_ids, row_data_dict)
                    env.FastFading_Module()  
                    # 1.4 更新UAV的位置.然后每个设备，都可以获取最近范围的n个服务车辆的信息
                    self.uav_positions = uav_manager.get_UAV_position(self.result_id) # [n_UAV, 3]
                    # self.uav_positions的前两项乘以5，因为存储的是网格信息
                    tmp_uav_positions = self.uav_positions.copy()
                    tmp_uav_positions[:, :2] = tmp_uav_positions[:, :2] * 5
                    env.renew_uav_positions(tmp_uav_positions)
                    # 2 根据位移信息，更新通信信道状态
                    env.FastFading_Module()  
                    # 3 任务生成
                    env.Task_Generation_Module()
                    # 4 通过算法获取卸载决策（每个time_step进行，一次性卸载当前step内所有的to_offload_tasks）
                    task_path_dict_list = self.algorithm_module.act_offloading(env)
                    TTI_flag = False
                    while not TTI_flag:
                        # 5 执行任务卸载（每个TTI进行）
                        task_path_dict_list = env.Offload_Tasks(task_path_dict_list) # 只offload task.start_time == cur_time的任务
                        # 每一个TTI都需要执行RB和计算的资源分配
                        activated_offloading_tasks_with_RB_Nos = self.algorithm_module.act_RB_allocation(env)
                        env.Communication_RB_Allocation(activated_offloading_tasks_with_RB_Nos)
                        env.Compute_Rate()
                        env.Execute_Communicate()
                        cpu_allocation_for_fog_nodes = self.algorithm_module.act_CPU_allocation(env)
                        env.Execute_Compute(cpu_allocation_for_fog_nodes)
                        # 11 更新环境状态
                        TTI_flag = env.Time_Update_Module()
                    self.algorithm_module.calculate_reward()
                    if not ('notLearn' in self.args.method) or (self.args.to_train is False):
                        self.algorithm_module.store_experience(self.veh_positions, self.uav_positions, self.cur_episode, False)
                        self.algorithm_module.update_agents(cnt)
                    # 12 检查超时的任务（包括计算和验算，以及to_pay）
                    env.Check_To_X_Tasks()
                    self.simulation_delay = time.time() - step_start_time
                    cnt += 1
                    if cnt % 10 == 0:
                        self.algorithm_module.log_reward(cnt+iteration*self.step_per_episode*self.iteration_episodes)
            self.traci_connection.close()
            time.sleep(5) # 等待一秒，确保sumo进程关闭
            self.traci_connection.start(sumo_cmd)
            manager.reset()
import torch
if __name__ == "__main__":
    torch.set_num_threads(6)
    args = parse_arguments_for_MASAC()
    set_seed(args.random_seed)
    app = DRLEnvironmentWrapper(traci, args)
    try:
        app.run()
    # 抓捕keyboard interrupt 和 runtime exception
    except (KeyboardInterrupt) as e:
        print('KeyboardInterrupt, stopping...')
        print("Exiting program.")