import numpy as np
import matplotlib.pyplot as plt
def is_overlapping(x1, y1, r1, x2, y2, r2):
    """计算两个圆是否重叠"""
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance < (r1 + r2)

def generate_no_fly_zones(num_zones=3, area=(250, 2000, 250, 1500), total_area_proportion=0.2):
    x_min, x_max, y_min, y_max = area
    total_area = (x_max - x_min) * (y_max - y_min) * total_area_proportion
    avg_area_per_zone = total_area / num_zones
    avg_radius = np.sqrt(avg_area_per_zone / np.pi)

    zones = []
    for _ in range(num_zones):
        while True:
            x_center = np.random.randint(x_min, x_max)
            y_center = np.random.randint(y_min, y_max)
            radius = np.random.normal(avg_radius, avg_radius * 0.3)  # 假设标准差是平均半径的30%

            # 检查新圆是否与现有圆重叠
            if all(not is_overlapping(x_center, y_center, radius, x, y, r) for x, y, r in zones):
                zones.append((x_center, y_center, radius))
                break
    return zones


def plot_no_fly_zones(zones, area, idx):
    x_min, x_max, y_min, y_max = area
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    # y轴反向
    ax.invert_yaxis()
    # y axis的坐标是[250, 1500, 50], x axis的坐标是[250, 2000, 50]
    ax.set_xticks(np.arange(x_min, x_max+1, 250))
    ax.set_yticks(np.arange(y_min, y_max+1, 250))
    ax.set_title("No-Fly Zones Visualization")
    ax.set_aspect('equal', adjustable='box')

    for x_center, y_center, radius in zones:
        circle = plt.Circle((x_center, y_center), radius, color='red', fill=True, alpha=0.5)
        ax.add_artist(circle)

    plt.show()
    plt.savefig(f'./images/no_fly_zones_{idx}.png', dpi=300, bbox_inches='tight')

# 种子
np.random.seed(0)
no_fly_zones_results = []
for i in range(5):
    no_fly_zones = generate_no_fly_zones()
    plot_no_fly_zones(no_fly_zones, (250, 2000, 250, 1500), idx=i)
    no_fly_zones_results.append(no_fly_zones)
print(no_fly_zones_results)
