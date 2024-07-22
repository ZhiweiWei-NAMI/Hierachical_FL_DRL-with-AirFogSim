import traci
def bresenham_line(x0, y0, x1, y1):
    """生成线段上的点集"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

def update_grid(shape, distance, max_speed):
    """根据线段更新栅格"""
    global length_map_grid, max_speed_map_grid
    (x1, y1), (x2, y2) = shape
    grid_points = bresenham_line(int(x1), int(y1), int(x2), int(y2))
    for (x, y) in grid_points:
        if 50 <= x < 400 and 50 <= y < 300:
            length_map_grid[y][x] = distance
            max_speed_map_grid[y][x] = max_speed

# 配置你的 SUMO 仿真环境
sumoCmd = ["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", "/home/weizhiwei/data/uav_compute/sumo_berlin/map.sumocfg"]

traci.start(sumoCmd)
# 获取所有边的列表
lanes = traci.lane.getIDList()
# traci.vehicle.add("1", routeID=None, typeID="DEFAULT_VEHTYPE")
cnt = 0
tmp_lanes = []
# 以5为网格，横跨[0,2000;0,2000]，一共是400*400个网格
length_map_grid = [[0 for i in range(400)] for j in range(400)]
max_speed_map_grid = [[0 for i in range(400)] for j in range(400)]
for lane in lanes:
    # 获取边的形状
    allowed = traci.lane.getAllowed(lane)
    if "passenger" in allowed:
        shape = traci.lane.getShape(lane) # [(x1, y1), (x2, y2)]
        # shape 需要除以5，因为我们的map_grid是以5为单位的
        shape = [(x / 5, y / 5) for (x, y) in shape]
        # 如果shape不是2，而是更多的，那么就遍历shape，每两个点为一条边
        for i in range(1, len(shape)):
            tmp_shape = [shape[i-1], shape[i]]
            # 直接根据tmp_shape计算distance
            distance = traci.lane.getLength(lane)
            max_speed = traci.lane.getMaxSpeed(lane) # m/s
            # 分析shape，对应到map_grid中，只要有一个点在map_grid中，就认为这个边是有效的，并且记录length和max_speed
            # 虽然shape是两个点，但实际上应该是一条线段。所以需要分析从起始点        
            update_grid(tmp_shape, distance, max_speed)

traci.close()
# 保存两个矩阵到文件
import pickle
with open('./images/length_map_grid.pkl', 'wb') as f:
    pickle.dump(length_map_grid, f)
with open('./images/max_speed_map_grid.pkl', 'wb') as f:
    pickle.dump(max_speed_map_grid, f)
# 可视化两个矩阵，length_map_grid和max_speed_map_grid，仅展示50-400, 50-300的部分
import matplotlib.pyplot as plt
import numpy as np
# 保存图像到文件./images/length_map_grid.png
plt.figure()
plt.imshow(length_map_grid, cmap='hot', interpolation='nearest')
plt.xlim(50, 400)
plt.ylim(50, 300)
# 横轴纵轴乘以5，就是250, 2000, 250, 1500
plt.xticks(np.arange(50, 401, 50), np.arange(250, 2001, 250))
plt.yticks(np.arange(50, 301, 50), np.arange(1500, 249, -250))

plt.colorbar()
plt.title('Length per Map Grid (m)')
plt.savefig('./images/length_map_grid.png', dpi=300, bbox_inches='tight')
# 保存图像到文件./images/max_speed_map_grid.png
plt.figure()
plt.imshow(max_speed_map_grid, cmap='hot', interpolation='nearest')
plt.xlim(50, 400)
plt.ylim(50, 300)
plt.xticks(np.arange(50, 401, 50), np.arange(250, 2001, 250))
plt.yticks(np.arange(50, 301, 50), np.arange(1500, 249, -250))
plt.colorbar()
plt.title('Max Speed per Map Grid (m/s)')
plt.savefig('./images/max_speed_map_grid.png', dpi=300, bbox_inches='tight')

