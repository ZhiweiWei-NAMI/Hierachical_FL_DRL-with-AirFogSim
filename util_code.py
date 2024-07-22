import numpy as np


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
def extract_RSU_state(veh_pos_maps, uav_pos_maps_2d, no_fly_zones, rsu_positions, coverage_map, length_map_grid, max_speed_map_grid, rsu_id, radius=60):
    # veh_pos_map是[400,400]，记录veh的数量；uav_pos_map是[400,400,2]，记录uav的数量，由于是2个层次，所以也要添加两个state
    output_size = 120
    half_size = output_size // 2

    rsu_x, rsu_y = rsu_positions[rsu_id]
    x_start = max(rsu_x - half_size, 0)
    x_end = min(rsu_x + half_size, len(max_speed_map_grid[0]))
    y_start = max(rsu_y - half_size, 0)
    y_end = min(rsu_y + half_size, len(max_speed_map_grid))

    # 初始化状态数组，加入禁飞层
    state_length = np.zeros((output_size, output_size))
    state_speed = np.zeros((output_size, output_size))
    state_coverage = np.zeros((output_size, output_size))
    state_no_fly = np.zeros((output_size, output_size))
    state_veh_num = np.zeros((output_size, output_size))
    state_uav_num_100 = np.zeros((output_size, output_size))
    state_uav_num_200 = np.zeros((output_size, output_size))

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

    # 归一化
    max_val = [232.68, 13.89, 1.0, 1.0, 3.0, 1.0, 2.0]
    state_length = np.array(state_length) / max_val[0]
    state_speed = np.array(state_speed) / max_val[1]
    state_coverage = np.array(state_coverage) / max_val[2]
    state_no_fly = np.array(state_no_fly) / max_val[3]
    state_veh_num = np.array(state_veh_num) / max_val[4]
    state_uav_num_100 = np.array(state_uav_num_100) / max_val[5]
    state_uav_num_200 = np.array(state_uav_num_200) / max_val[6]
    # 将四层状态合并为一个 [7, 120, 120] 的数组
    state = np.stack([state_length, state_speed, state_coverage, state_no_fly, state_veh_num, state_uav_num_100, state_uav_num_200], axis=0)
    return state

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
    uav_pos_maps = np.zeros((400, 400, 2)) # 400*400的网格，每个网格内有2个值，分别代表不同层次的UAV数量（z=100, z=200）
    for i, uav_trace in enumerate(uav_traces):
        x, y, z = uav_trace
        x = int(x)
        y = int(y)
        z = int(z)
        if 50 <= x < 400 and 50 <= y < 300:
            uav_pos_maps[int(y)][int(x)][int(z//100-1)] += 1
    return uav_pos_maps