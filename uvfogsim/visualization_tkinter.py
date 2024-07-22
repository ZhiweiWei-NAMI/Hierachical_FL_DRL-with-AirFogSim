# -*- encoding: utf-8 -*-
'''
@File    :   visualization_tkinter.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''



import tkinter as tk
import os
from shapely.affinity import scale
import traci
import subprocess
import sys
import math
from environment import Environment
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import time
from tkinter_utils import *
import math
from algorithms import *

class VisualizationApp(tk.Tk):
    def __init__(self, traci_connection, osm_file_path, net_file_path, poly_file_path, algorithm_module):
        super().__init__()
        
        # 判断algorithm_module是不是Base_Algorithm_Module的子类
        assert isinstance(algorithm_module, Base_Algorithm_Module), 'algorithm_module must be a subclass of Base_Algorithm_Module'
        self.traci_connection = traci_connection
        self.environment = None
        self.algorithm_module = algorithm_module
        self.osm_file_path = osm_file_path
        self.net_file_path = net_file_path
        self.poly_file_path = poly_file_path
        self.title("Vehicle Visualization")
        self.geometry("800x800")
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        # 绑定鼠标左键点击事件
        self.canvas.bind('<ButtonPress-1>', self.drag_start)
        # 绑定鼠标左键移动事件
        self.canvas.bind('<B1-Motion>', self.drag_move)
        self.vehicle_positions = []
        self.old_vehicle_position_dict = {}
        self.vehicle_directions = []
        self.map_data = get_map_data_from_osm(self.osm_file_path)
        self.time_step = traci.simulation.getDeltaT()
        self.bbox = None
        self.location_bound = None
        self.calculate_bbox()
        self.canvas_width = 800
        self.canvas_height = 800
        lon_width = self.bbox[2] - self.bbox[0]
        lat_height = self.bbox[3] - self.bbox[1]
        self.uav_positions = []
        aspect_ratio = self.canvas_width / self.canvas_height
        if lon_width / lat_height > aspect_ratio:
            # 地图的经度范围较大
            self.canvas_height = int(self.canvas_width * lat_height / lon_width)
        else:
            # 地图的纬度范围较大
            self.canvas_width = int(self.canvas_height * lon_width / lat_height)
        self.car_images = create_rotated_images(r'E:\scholar\papers\vfc_simulator\icon\car.png')
        self.uav_image = make_image_transparent(r'E:\scholar\papers\vfc_simulator\icon\uav.png', size=(30, 30))
        self.bs_image = make_image_transparent(r'E:\scholar\papers\vfc_simulator\icon\bs.png', size=(50, 50))
        self.map_image = tk.PhotoImage(width=self.canvas_width, height=self.canvas_height)
        self.scale_factor = 4
        self.bs_images = None
        self.map_images = None
        self.vehicle_images = {}
        self.uav_images = {}
        self.bs_positions = []
        self.time_image = None
        self.execution_time = 0
        

    # 拖拽开始函数
    def drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    # 拖拽移动函数
    def drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def calculate_bbox(self):
        # 从net.xml文件中读取地图的bbox,通过parse_location_info函数
        conv_boundary, orig_boundary = parse_location_info(self.net_file_path)
        orig_boundary = tuple(map(float, orig_boundary.split(',')))
        conv_boundary = tuple(map(float, conv_boundary.split(',')))
        min_x = orig_boundary[0]
        min_y = orig_boundary[1]
        max_x = orig_boundary[2]
        max_y = orig_boundary[3] 
        self.bbox = min_x, min_y, max_x, max_y
        self.location_bound = conv_boundary

    def draw_bss(self, bs_positions):
        if self.bs_images is None:
            self.bs_images = []
            for bs_position in bs_positions:
                bs_image = self.canvas.create_image(bs_position[0], bs_position[1], image=self.bs_image, anchor=tk.CENTER)
                self.bs_images.append(bs_image)

    def update_visualization(self, cur_time, vehicle_ids):
        vehicle_positions = self.vehicle_positions
        vehicle_directions = self.vehicle_directions
        # Draw the map and vehicles 
        if self.map_images is None:
            self.map_images = self.canvas.create_image(0, 0, image=self.map_image, anchor=tk.NW)
        self.draw_bss(self.bs_positions)
        self.draw_vehicles(vehicle_positions, vehicle_directions, vehicle_ids)
        self.draw_uavs(self.uav_positions)
        self.draw_time(cur_time)
        self.draw_links()

    def draw_time(self, cur_time):
        if self.time_image is not None:
            self.canvas.delete(self.time_image)
        posx = (self.canvas_width - 200) * self.scale_factor
        posy = (self.canvas_height - 250) * self.scale_factor
        # execution time 只显示小数点后两位
        self.time_image = self.canvas.create_text(posx, posy, text="Time: {} Vehicle Num: {} Execution Delay: {} ms".format(cur_time, len(self.vehicle_positions), round(self.execution_time * 1000, 3)), anchor=tk.NW)

    def draw_map(self, map_data):
        map_image = Image.new("RGBA", (int(self.canvas_width * self.scale_factor), int(self.canvas_height * self.scale_factor)), (255, 255, 255, 255))
        draw = ImageDraw.Draw(map_image)
        for road_segment in map_data:
            (x1, y1), (x2, y2) = road_segment
            x1, y1 = lonlat_to_pixel(x1, y1, self.bbox, self.canvas_width, self.canvas_height)
            x2, y2 = lonlat_to_pixel(x2, y2, self.bbox, self.canvas_width, self.canvas_height)
            # 应用缩放因子
            x1, y1 = x1 * self.scale_factor, y1 * self.scale_factor
            x2, y2 = x2 * self.scale_factor, y2 * self.scale_factor
            draw.line([(x1, y1), (x2, y2)], fill="gray", width=1)
        self.map_image = ImageTk.PhotoImage(map_image)

    def draw_uavs(self, uav_positions):
        # canvas删除tags为uav的
        self.canvas.delete("uav")
        for position in uav_positions:
            x, y = position
            self.canvas.create_image(x,y,image=self.uav_image, tags="uav")

    def draw_vehicles(self, vehicle_positions, directions, vehicle_ids):
        # 遍历self.vehicle_images，查看是否在vehicle_ids，如果不在则删除
        for vehicle_id in list(self.vehicle_images.keys()).copy():
            if vehicle_id not in vehicle_ids:
                self.canvas.delete(self.vehicle_images[vehicle_id])
                del self.vehicle_images[vehicle_id]
        # 遍历vehicle_ids，查看是否在self.vehicle_images中，如果不在则创建，否则move
        for idx, vehicle_id in enumerate(vehicle_ids):
            direction = directions[idx]
            # 根据direction选择self.car_images中最接近的图片
            tmp_car_image = self.car_images[(math.floor(direction/10)-9) % 36]
            if vehicle_id not in self.vehicle_images:
                self.vehicle_images[vehicle_id] = self.canvas.create_image(*vehicle_positions[idx],image=tmp_car_image, tags="vehicle")
                self.old_vehicle_position_dict[vehicle_id] = vehicle_positions[idx]
            else:
                self.canvas.move(self.vehicle_images[vehicle_id], vehicle_positions[idx][0] - self.old_vehicle_position_dict[vehicle_id][0], vehicle_positions[idx][1] - self.old_vehicle_position_dict[vehicle_id][1])
                self.old_vehicle_position_dict[vehicle_id] = vehicle_positions[idx]
    
    def draw_links(self):
        env = self.environment
        self.canvas.delete("v2v_link")
        for idx1, idx2 in zip(*np.where(env.V2V_active_links)):
            vid1 = env.get_vid_by_index(idx1)
            vid2 = env.get_vid_by_index(idx2)
            pos1 = self.old_vehicle_position_dict[f"{vid1}"]
            pos2 = self.old_vehicle_position_dict[f"{vid2}"]
            self.canvas.create_line(pos1, pos2, dash=(4, 4), fill="red", tags="v2v_link")

        self.canvas.delete("v2u_link")
        for idx1, idx2 in zip(*np.where(env.V2U_active_links)):
            vid1 = env.get_vid_by_index(idx1)
            pos1 = self.old_vehicle_position_dict[f"{vid1}"]
            pos2 = self.uav_positions[idx2]
            self.canvas.create_line(pos1, pos2, dash=(4, 4), fill="blue", tags="v2u_link")

        self.canvas.delete("v2i_link")
        for idx1, idx2 in zip(*np.where(env.V2I_active_links)):
            vid1 = env.get_vid_by_index(idx1)
            pos1 = self.old_vehicle_position_dict[f"{vid1}"]
            pos2 = self.bs_positions[idx2]
            self.canvas.create_line(pos1, pos2, dash=(4, 4), fill="green", tags="v2i_link")
        
        self.canvas.delete("u2u_link")
        for idx1, idx2 in zip(*np.where(env.U2U_active_links)):
            pos1 = self.uav_positions[idx1]
            pos2 = self.uav_positions[idx2]
            self.canvas.create_line(pos1, pos2, dash=(4, 4), fill="orange", tags="u2u_link")
        
        self.canvas.delete("u2i_link")
        for idx1, idx2 in zip(*np.where(env.U2I_active_links)):
            pos1 = self.uav_positions[idx1]
            pos2 = self.bs_positions[idx2]
            self.canvas.create_line(pos1, pos2, dash=(4, 4), fill="purple", tags="u2i_link")

    def position_to_pixel(self, x, y):
        x = (x - self.location_bound[0]) / (self.location_bound[2] - self.location_bound[0])
        y = (y - self.location_bound[1]) / (self.location_bound[3] - self.location_bound[1])
        x *= self.canvas_width * self.scale_factor
        y *= self.canvas_height * self.scale_factor
        return x,y
    def update_vehicle_positions(self, vehicle_ids):
        self.vehicle_directions = []
        vehicle_positions = []
        row_data_dict = {}
        for vehicle_id in vehicle_ids:
            x, y = self.traci_connection.vehicle.getPosition(vehicle_id)
            row_data = {}
            row_data['id'] = int(vehicle_id)
            row_data['x'] = x
            row_data['y'] = y
            row_data['angle'] = self.traci_connection.vehicle.getAngle(vehicle_id)
            row_data['speed'] = self.traci_connection.vehicle.getSpeed(vehicle_id)
            row_data_dict[int(vehicle_id)] = row_data
            x, y = self.position_to_pixel(x, y)
            vehicle_positions.append([x, y])
            self.vehicle_directions.append(row_data['angle'])
        self.vehicle_positions = vehicle_positions
        return row_data_dict

    def run(self):
        map_data = self.map_data
        self.draw_map(map_data)
        env = Environment(draw_it = False, n_UAV = 5)
        env.time_step = self.time_step
        assert env.time_step >= 0.1, 'time_step must be greater than 0.1'
        env.initialize(self.location_bound)
        self.environment = env
        cnt = 0
        bs_positions = []
        for bs_position in env.BS_positions:
            bs_positions.append(self.position_to_pixel(bs_position[0], bs_position[1]))
        self.bs_positions = bs_positions
        start_time = time.time()
        while self.traci_connection.simulation.getMinExpectedNumber() > 0:
            cnt += 1
            self.traci_connection.simulationStep() 
            vehicle_ids = self.traci_connection.vehicle.getIDList()
            row_data_dict = self.update_vehicle_positions(vehicle_ids)
            vehicle_positions = [[row_data['x'], row_data['y']] for row_data in row_data_dict.values()]
            # 采取移动行为的算法模块
            uav_directions, uav_speeds = self.algorithm_module.act_mobility(env, vehicle_ids, vehicle_positions)
            # （1）移动模块
            env.Mobility_Module(vehicle_ids, row_data_dict, uav_directions, uav_speeds)
            # （2）信道更新模块
            env.Communication_Module()
            # 采取任务卸载行为的算法模块
            v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2i_tran, v2u_tran, u2u_tran, u2v_tran, u2i_tran = self.algorithm_module.act_offloading(env)
            #  (3）任务卸载模块
            env.Task_Offloading_Module(v2v_offload, v2u_offload, v2i_offload, u2u_offload, u2v_offload, u2i_offload,v2v_tran, v2i_tran, v2u_tran, u2u_tran, u2v_tran, u2i_tran)
            # （4）资源分配与计算模块
            env.Computation_Module()
            # （5）任务生成模块
            env.Task_Generation_Module()
            # （6）时间步长更新模块
            env.Time_Update_Module()

            if cnt % 10 == 0:
                self.vehicle_images = {} # 每隔一度时间重新绘制一遍，保证方向问题
                self.canvas.delete("vehicle")
            # Update UAV position
            self.update_uav_position(env)
            self.update_visualization(self.traci_connection.simulation.getTime(), vehicle_ids)
            self.update()
            self.execution_time = (time.time() - start_time)
            start_time = time.time()
            
        self.traci_connection.close()

    def update_uav_position(self, env):
        # 从env获取所有无人机的位置，之后转化成像素坐标
        uav_positions = env.get_uav_positions()
        self.uav_positions = []
        for x, y in uav_positions:
            x, y = self.position_to_pixel(x, y)
            self.uav_positions.append([x, y])
        

if __name__ == "__main__":
    # 几个需要注意的点:
    # 1. 所有设备的传输可以调整为全双工或者半双工。如果是全双工，可以多址接入，但是干扰很大；如果是半双工，同一时刻只能接收或者发送，但是可能任务卸载失败
    # 2. 任务是全部卸载的,不包含那种部分卸载的情况;如果是可以划分的子任务,建议直接在生成任务的时候就划分好
    # 3. 对于通信信道, 传输过程中的SINR在时间片内是不变的,但是不同时间片之间是变化的,因此需要在每个时间片内计算SINR
    # 4. 即便是不同的时间片,对于已经正在传输的任务,并不会改变其传输的信道,因此在计算SINR的时候,需要考虑正在传输的任务
    # 5. 任务卸载/资源交易之类的决策每个时间片只能做一次,只能针对一个任务进行一次决策，决定是V2V, V2I, 还是V2U等
    # 6. 任务卸载的逻辑是车辆->无人机,车辆->基站,无人机->基站,无人机->无人机.当前版本下基站并不会卸载给无人机,基站也不会卸载给车辆
    # 7. 整个平台分为6个基础模块（1）任务生成模块（2）移动模块（3）任务卸载模块（4）通信信道模块（5）资源分配及计算模块（6）事件记录模块，以上模块顺序执行，执行过程统计相关数据作为评价指标
    # 基于基础模块，还有其他可拓展模块，比如基于资源分配，可以提供资源交易模块和区块链模块；基于任务卸载模块，可以提供任务卸载决策算法模块；基于通信信道模块，可以提供信道分配算法模块；基于移动模块，可以提供无人机轨迹优化模块等等，这些模块以input_action的形式在module_algorithms.py中实现，然后在main.py中调用
    dir_path = r"E:\scholar\papers\vfc_simulator\sumo_wujiaochang"
    sumocfg_file = os.path.join(dir_path, "map.sumo.cfg")
    osm_file_path = os.path.join(dir_path, "map.osm")
    net_file_path = os.path.join(dir_path, "my_net.net.xml")
    poly_file_path = os.path.join(dir_path, "my_poly.poly.xml")
    sumo_cmd = ["sumo", "-c", sumocfg_file]
    algorithm_module = Cluster_Algorithm_Module()
    sumo_process = subprocess.Popen(sumo_cmd, stdout=sys.stdout, stderr=sys.stderr)
    set_seed(42)

    traci.init(8813, host='127.0.0.1', numRetries=10)
    app = VisualizationApp(traci, osm_file_path, net_file_path, poly_file_path, algorithm_module)
    app.run()
    sumo_process.wait()