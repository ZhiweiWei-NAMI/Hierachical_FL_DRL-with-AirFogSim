
import pickle

import numpy as np

def check_full_coverage(max_speed_map_grid, rsu_positions, radius=60):
    """检查所有 max_speed > 0 的点是否至少被一个 RSU 覆盖"""
    for y in range(50, 300):
        for x in range(50, 400):
            if max_speed_map_grid[y][x] > 0:
                if not any(np.sqrt((x - rx)**2 + (y - ry)**2) <= radius for rx, ry in rsu_positions):
                    return False
    return True

# map的大小是400*400,但实际上50 <= x < 400 and 50 <= y < 300, map[y][x] 有效.在这个范围内随机生成12个RSU,然后按照voronoi图划分区域,且coverage不超过半径300/5=60,需要判断是否max_speed_map_grid[y][x] > 0的点都被任意一个RSU覆盖

# ./images/max_speed_map_grid.pkl load map列表
with open('./images/max_speed_map_grid.pkl', 'rb') as f:
    max_speed_map_grid = pickle.load(f)
# ./images/length_map_grid.pkl load map列表
with open('./images/length_map_grid.pkl', 'rb') as f:
    length_map_grid = pickle.load(f)

def generate_rsu_positions(grid_bounds):
    rsu_positions = []
    for (x0, y0, x1, y1) in grid_bounds:
        x = np.random.randint(x0, x1)
        y = np.random.randint(y0, y1)
        rsu_positions.append((x, y))
    return rsu_positions

def ensure_full_coverage(max_speed_map_grid, radius=60, max_attempts=1000, max_available_positions = 5):
    attempts = 0
    rsu_positions_list = []
    while attempts < max_attempts:
        try:
            grid_bounds = calculate_grid_bounds()
            rsu_positions = generate_rsu_positions(grid_bounds)
            if check_full_coverage(max_speed_map_grid, rsu_positions, radius):
                rsu_positions_list.append(rsu_positions)
                if len(rsu_positions_list) >= max_available_positions:
                    return rsu_positions_list
        except ValueError:
            attempts += 1
    raise ValueError("Could not find a valid RSU configuration")

def calculate_grid_bounds():
    x_start, x_end = 50, 400
    y_start, y_end = 50, 300
    num_columns = 4
    num_rows = 3
    grid_width = (x_end - x_start) // num_columns
    grid_height = (y_end - y_start) // num_rows
    grid_bounds = []
    
    for i in range(num_rows):
        for j in range(num_columns):
            x0 = x_start + j * grid_width
            x1 = x0 + grid_width
            y0 = y_start + i * grid_height
            y1 = y0 + grid_height
            grid_bounds.append((x0, y0, x1, y1))
    
    return grid_bounds

# 尝试生成 RSU 位置
try:
    rsu_positions_results = ensure_full_coverage(max_speed_map_grid)
    print(rsu_positions_results)
    # rsu_positions_results记录了最多5组,12个RSU的位置.保存为./images/rsu_positions.pkl
    with open('./images/rsu_positions.pkl', 'wb') as f:
        pickle.dump(rsu_positions_results, f)
except ValueError as e:
    print(e)