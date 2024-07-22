
import pickle

import numpy as np
rsu_positions_results = [[(112, 88), (141, 58), (244, 80), (369, 99), (124, 187), (189, 164), (297, 199), (320, 136), (74, 271), (207, 258), (254, 259), (334, 221)], [(131, 84), (187, 103), (281, 110), (369, 91), (107, 135), (211, 152), (263, 136), (378, 191), (128, 226), (179, 248), (256, 234), (322, 228)], [(112, 108), (204, 92), (306, 97), (391, 87), (57, 204), (177, 133), (261, 185), (338, 172), (121, 218), (186, 244), (263, 267), (364, 253)], [(89, 80), (158, 74), (247, 101), (355, 67), (78, 161), (175, 167), (269, 148), (341, 147), (100, 222), (202, 255), (276, 217), (344, 251)], [(108, 94), (213, 82), (299, 99), (357, 97), (95, 209), (175, 134), (270, 175), (379, 197), (63, 276), (177, 230), (309, 234), (374, 228)]]
no_fly_zone_results = [[(934, 809, 263.3870071425481), (1981, 1283, 315.82764458833543), (1997, 527, 117.4017041877551)], [(1746, 850, 159.46079502521124), (424, 1099, 209.13879180961072), (1100, 349, 237.02084218709663)], [(1451, 1005, 345.25277160683123), (673, 947, 230.31241074672877), (378, 1327, 121.44902277887451)], [(1824, 1006, 268.0090998233277), (1316, 793, 200.63560396801844), (541, 1393, 225.55923380031683)], [(1029, 680, 299.0083464063226), (861, 1158, 168.17489462537907), (1405, 1222, 134.47772905244267)]]
# no_fly_zone_results每个元素除以5
for i in range(len(no_fly_zone_results)):
    no_fly_zone_results[i] = [(x/5, y/5, r/5) for x, y, r in no_fly_zone_results[i]]
with open('./images/max_speed_map_grid.pkl', 'rb') as f:
    max_speed_map_grid = pickle.load(f)
# ./images/length_map_grid.pkl load map列表
with open('./images/length_map_grid.pkl', 'rb') as f:
    length_map_grid = pickle.load(f)
    
def is_in_no_fly_zone(x, y, no_fly_zones):
    """检查给定点是否在任何一个禁飞区内"""
    for zone_x, zone_y, radius in no_fly_zones:
        if (x - zone_x) ** 2 + (y - zone_y) ** 2 <= radius ** 2:
            return True
    return False

# 按照RSU的位置，以radius 60 为半径, 首先用voronoi图划分每个RSU的覆盖范围,形成coverage_map
    
def generate_voronoi_map(rsu_positions, radius = 60):
    coverage_map = np.ones((400, 400)) * -1
    # 对于每一个格点,判断最近的rsu,然后判断举例是否小于radius.如果小于radius,则认为是这个rsu的覆盖范围,coverage_map[y][x] = rsu_id
    for y in range(50, 300):
        for x in range(50, 400):
            min_dist = 100000
            rsu_id = -1
            for i, (rsu_x, rsu_y) in enumerate(rsu_positions):
                dist = np.sqrt((rsu_x - x) ** 2 + (rsu_y - y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    rsu_id = i
            if min_dist < radius:
                coverage_map[y][x] = rsu_id
    return coverage_map

# 然后,根据coverage_map,提取以RSU位置为中心,半径为radius的正方形区域(120*120)作为RSU的state.有3层,分别是length_map_grid, max_speed_map_grid, coverage_map.coverage_map中的值从rsu_id变为1,未覆盖范围变为0
def extract_state(veh_pos_maps, uav_pos_maps_2d, no_fly_zones, rsu_positions, coverage_map, length_map_grid, max_speed_map_grid, rsu_id, radius=60):
    # veh_pos_map是[400,400]，记录veh的数量；uav_pos_map是[400,400,2]，记录uav的数量，由于是2个层次，所以也要添加两个state
    output_size = 120
    half_size = output_size // 2

    rsu_x, rsu_y = rsu_positions[rsu_id]
    x_start = max(rsu_x - half_size, 0)
    x_end = min(rsu_x + half_size, len(max_speed_map_grid[0]))
    y_start = max(rsu_y - half_size, 0)
    y_end = min(rsu_y + half_size, len(max_speed_map_grid))

    # 初始化状态数组，加入禁飞层
    state_length = [[0] * output_size for _ in range(output_size)]
    state_speed = [[0] * output_size for _ in range(output_size)]
    state_coverage = [[0] * output_size for _ in range(output_size)]
    state_no_fly = [[0] * output_size for _ in range(output_size)]
    state_veh_num = [[0] * output_size for _ in range(output_size)]
    state_uav_num_100 = [[0] * output_size for _ in range(output_size)]
    state_uav_num_200 = [[0] * output_size for _ in range(output_size)]

    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            rel_x = x - rsu_x + half_size
            rel_y = y - rsu_y + half_size
            if 0 <= rel_x < output_size and 0 <= rel_y < output_size:
                state_length[rel_y][rel_x] = length_map_grid[y][x]
                state_speed[rel_y][rel_x] = max_speed_map_grid[y][x]
                state_coverage[rel_y][rel_x] = 1 if coverage_map[y][x] == rsu_id else 0
                # 检查并标记禁飞区,1表示在禁飞区内
                state_no_fly[rel_y][rel_x] = 1 if is_in_no_fly_zone(x, y, no_fly_zones) else 0
                # 记录veh的数量
                state_veh_num[rel_y][rel_x] = veh_pos_maps[y][x]
                # 记录uav的数量
                state_uav_num_100[rel_y][rel_x] = uav_pos_maps_2d[y][x][0]
                state_uav_num_200[rel_y][rel_x] = uav_pos_maps_2d[y][x][1]


    # 将四层状态合并为一个 [120, 120, 7] 的数组
    state = np.stack([state_length, state_speed, state_coverage, state_no_fly, state_veh_num, state_uav_num_100, state_uav_num_200], axis=-1)
    return state

def generate_uav_tracking_data(non_fly_zone, n_UAV=40, n_vel=3, n_iteration=2, step_per_episode=500):
    # 定义飞行区域
    x_min, x_max, y_min, y_max = 50, 400, 50, 300
    uav_traces = []

    # 随机部署UAV的位置在非禁飞区
    initial_positions = []
    destination_positions = []
    for _ in range(n_UAV):
        while True:
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            # 高度也分层，分别是100, 200两个高度
            z = np.random.choice([100, 200])
            if not is_in_no_fly_zone(x, y, non_fly_zone):
                initial_positions.append([x, y, z])
                break
        while True:
            x = np.random.randint(x_min, x_max)
            y = np.random.randint(y_min, y_max)
            if not is_in_no_fly_zone(x, y, non_fly_zone):
                destination_positions.append([x, y, z]) # 目的地的z和起飞的z相同
                break 

    # Let UAVs fly between points
    for i in range(n_UAV):
        trace = []
        current_position = initial_positions[i]
        destination = destination_positions[i]

        for _ in range(step_per_episode*n_iteration):
            trace.append(current_position)
            current_x, current_y, current_z = current_position
            dest_x, dest_y, dest_z = destination

            # Calculate direction vector normalized
            direction_vector = np.array([dest_x - current_x, dest_y - current_y, 0])
            distance = np.linalg.norm(direction_vector)
            normalized_vector = direction_vector / distance if distance != 0 else np.zeros(3)

            # Move UAV
            next_x = current_x + normalized_vector[0] * n_vel
            next_y = current_y + normalized_vector[1] * n_vel
            next_z = current_z 

            # Check if the next position is in the no-fly zone
            if is_in_no_fly_zone(next_x, next_y, non_fly_zone):
                # If entering a no-fly zone, treat the current position as the new destination
                destination = current_position
            else:
                # Update current position
                current_position = [next_x, next_y, next_z]

                # Check if destination is reached
                if np.linalg.norm(np.array([dest_x - next_x, dest_y - next_y, dest_z - next_z])) < n_vel:
                    # Swap current and destination
                    destination, current_position = current_position, destination

        uav_traces.append(trace)

    return uav_traces # [n_UAV, n_iteration*step_per_episode, 3]

import traci
import subprocess
import sys
import time
from uvfogsim.vehicle_manager import VehicleManager
def project_veh_position_to_map(veh_positions):
    veh_pos_maps = np.zeros((400, 400))
    for x, y in veh_positions:
        x /= 5
        y /= 5
        x = int(x)
        y = int(y)
        if 50 <= x < 400 and 50 <= y < 300:
            veh_pos_maps[int(y)][int(x)] += 1
    return veh_pos_maps

def project_uav_position_to_map(uav_traces):
    # uav_traces: [5, 40, 3]
    uav_pos_maps = np.zeros((5, 400, 400, 2)) # 5个禁飞区，每个禁飞区400*400的网格，每个网格内有2个值，分别代表不同层次的UAV数量（z=100, z=200）
    for i, uav_trace in enumerate(uav_traces):
        for x, y, z in uav_trace:
            x = int(x)
            y = int(y)
            z = int(z)
            if z == 200:
                a = 3
                pass
            if 50 <= x < 400 and 50 <= y < 300:
                uav_pos_maps[i][int(y)][int(x)][int(z//100-1)] += 1
    return uav_pos_maps

def collect_data_by_sumo(uav_trace_dataset_results, n_iteration=2, step_per_episode=500): # 生成50*1000条数据，再结合12*5=60个RSU位置，一共是3,000,000个state
    global rsu_positions_results, no_fly_zone_results
    rsu_coverage_maps = [None for _ in range(5)]
    train_dataset = [] # 前3个RSU部署的方案是训练集
    test_dataset = [] # 后2个RSU部署的方案是测试集
    sumocfg_file = "/home/weizhiwei/data/uav_compute/sumo_berlin/map.sumocfg"
    net_file_path = "/home/weizhiwei/data/uav_compute/sumo_berlin/map.net.xml"
    sumo_cmd = ["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", sumocfg_file]
    sumo_process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
    n_veh = 250
    traci_connection = traci
    traci_connection.init(8823, host='127.0.0.1', numRetries=10)
    time_step = traci_connection.simulation.getDeltaT()
    print("仿真过程中决策的时隙等于SUMO仿真的时隙长度: ", time_step)
    manager = VehicleManager(n_veh, net_file_path) # 指定200辆车
    traci_connection.simulationStep()
    import tqdm
    for iteration in range(n_iteration):
        manager.clear_all_vehicles(traci_connection)
        # manager.turn_off_traffic_lights(self.traci_connection)
        for n_step in tqdm.tqdm(range(step_per_episode)):
            # 1 每个time_step进行，控制区域内车辆数量
            manager.manage_vehicles(traci_connection)
            traci_connection.simulationStep()
            vehicle_ids = traci_connection.vehicle.getIDList()
            # 获取vehicle的位置信息，遍历RSU位置部署方案和禁飞区方案，extract_state添加到train_dataset或test_dataset
            # 每个stamp的无人机和veh的位置信息
            veh_positions = []
            for vehicle_id in vehicle_ids:
                x, y = traci_connection.vehicle.getPosition(vehicle_id)
                veh_positions.append((x, y))
            veh_pos_maps = project_veh_position_to_map(veh_positions) # [400, 400], 每个网格内的veh数量
            uav_trace_time_stamp = iteration * step_per_episode + n_step
            uav_pos_maps = project_uav_position_to_map(uav_trace_dataset_results[:, :, uav_trace_time_stamp, :]) # [5, 400, 400, 2], 每个网格内的uav数量, 5个uav禁飞区的方案
            for result_id, rsu_positions in enumerate(rsu_positions_results):
                no_fly_zones = no_fly_zone_results[result_id]
                if rsu_coverage_maps[result_id] is None:
                    coverage_map = generate_voronoi_map(rsu_positions)
                    rsu_coverage_maps[result_id] = coverage_map
                else:
                    coverage_map = rsu_coverage_maps[result_id]
                for i in range(len(rsu_positions)):
                    state = extract_state(veh_pos_maps, uav_pos_maps[result_id], no_fly_zones, rsu_positions, coverage_map, length_map_grid, max_speed_map_grid, i)
                    if result_id < 3:
                        train_dataset.append(state)
                    else:
                        test_dataset.append(state)
                
        traci_connection.close()
        time.sleep(5) # 等待一秒，确保sumo进程关闭
        traci_connection.start(sumo_cmd)
        manager.reset()
        # 存储iteration的数据
        # 保存到/data
        with open(f'./dataset/train_dataset.pkl_{iteration}', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'./dataset/test_dataset.pkl_{iteration}', 'wb') as f:
            pickle.dump(test_dataset, f)
        test_dataset = []
        train_dataset = []
        print(f"iteration {iteration} done, stored data.")

    return train_dataset, test_dataset



# 1. 生成UAV轨迹数据，在不同的禁飞区和RSU部署方案下
uav_trace_dataset_results = []
for i in range(5):
    uav_trace_dataset_results.append(generate_uav_tracking_data(no_fly_zone_results[i]))
uav_trace_dataset_results = np.array(uav_trace_dataset_results) # [5, 40, 50000, 3]
# 存储uav_trace_dataset_results

with open('./dataset/uav_trace_dataset_results.pkl', 'wb') as f:
    pickle.dump(uav_trace_dataset_results, f)

# 2. 生成数据集
train_dataset, test_dataset = collect_data_by_sumo(uav_trace_dataset_results)
print(len(train_dataset), len(test_dataset)) # 300,000