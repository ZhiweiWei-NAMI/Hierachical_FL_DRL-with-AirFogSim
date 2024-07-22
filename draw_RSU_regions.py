import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

from scipy.spatial import Voronoi

def plot_RSU_regions_and_nofly(rsu_positions, area, idx, no_fly_zones):
    # 绘制RSU的voronoi图
    vor = Voronoi(rsu_positions)
    fig, ax = plt.subplots()
    voronoi_plot_2d(vor, ax=ax)
    # x轴从左到右，y轴从上到下；都是以250为间隔，从250到2000，从250到1500
    ax.set_xlim(area[0], area[1])
    ax.set_ylim(area[2], area[3])
    # x轴标签需要从250开始，到2000结束，间隔250
    ax.set_xticks(np.arange(area[0], area[1]+1, 250))
    # y轴标签需要从250开始，到1500结束，间隔250
    ax.set_yticks(np.arange(area[2], area[3]+1, 250))
    plt.title(f'RSU Regions and No-Fly Zones {idx}')
    # 把RSU的位置标记出来，用五角星
    for rid, rsu_position in enumerate(rsu_positions):
        ax.plot(rsu_position[0], rsu_position[1], 'r*', markersize=10)
        # 添加文字
        ax.text(rsu_position[0], rsu_position[1], f'RSU {rid+1}', fontsize=12)

    # 绘制no-fly zones，用透明度为
    for nid, (x_center, y_center, radius) in enumerate(no_fly_zones):
        circle = plt.Circle((x_center, y_center), radius, color='skyblue', fill=True, alpha=0.85)
        ax.add_artist(circle)
        # 添加文字
        # ax.text(x_center, y_center, f'No-Fly Zone {nid+1}', fontsize=12)
    plt.savefig(f'./images/rsu_regions_{idx}.png', dpi=300, bbox_inches='tight')


# with open('./images/rsu_positions.pkl', 'rb') as f:
rsu_positions_results = [[(112, 88), (141, 58), (244, 80), (369, 99), (124, 187), (189, 164), (297, 199), (320, 136), (74, 271), (207, 258), (254, 259), (334, 221)], [(131, 84), (187, 103), (281, 110), (369, 91), (107, 135), (211, 152), (263, 136), (378, 191), (128, 226), (179, 248), (256, 234), (322, 228)], [(112, 108), (204, 92), (306, 97), (391, 87), (57, 204), (177, 133), (261, 185), (338, 172), (121, 218), (186, 244), (263, 267), (364, 253)], [(89, 80), (158, 74), (247, 101), (355, 67), (78, 161), (175, 167), (269, 148), (341, 147), (100, 222), (202, 255), (276, 217), (344, 251)], [(108, 94), (213, 82), (299, 99), (357, 97), (95, 209), (175, 134), (270, 175), (379, 197), (63, 276), (177, 230), (309, 234), (374, 228)]]
no_fly_zone_results = [[(934, 809, 263.3870071425481), (1981, 1283, 315.82764458833543), (1997, 527, 117.4017041877551)], [(1746, 850, 159.46079502521124), (424, 1099, 209.13879180961072), (1100, 349, 237.02084218709663)], [(1451, 1005, 345.25277160683123), (673, 947, 230.31241074672877), (378, 1327, 121.44902277887451)], [(1824, 1006, 268.0090998233277), (1316, 793, 200.63560396801844), (541, 1393, 225.55923380031683)], [(1029, 680, 299.0083464063226), (861, 1158, 168.17489462537907), (1405, 1222, 134.47772905244267)]]
no_fly_zone_results = np.array(no_fly_zone_results)
rsu_positions_results = np.array(rsu_positions_results)
rsu_positions_results *= 5
# print(rsu_positions_results)
for idx, rsu_positions in enumerate(rsu_positions_results):
    plot_RSU_regions_and_nofly(rsu_positions, (250, 2000, 250, 1500), idx, no_fly_zone_results[idx])