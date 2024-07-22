
import sumolib
import random
from uvfogsim.algorithms import set_seed

class VehicleManager:
    def __init__(self, n_veh, net_file):
        self.n_veh = n_veh
        self.vehicle_ids = list()
        self.vid = 0
        
        # 获取路网的所有边
        # self.edges = [edge_id for edge_id in traci.edge.getIDList() if traci.edge.getLaneNumber(edge_id) > 0]
        self.edges = self.get_valid_edges(net_file)
        # 将edge随机划分为一半的departure edges和arrival edges
        self.departure_edges = random.sample(self.edges, len(self.edges) // 2)
        self.arrival_edges = list(set(self.edges) - set(self.departure_edges))
    def turn_off_traffic_lights(self, traci):
        traffic_lights = traci.trafficlight.getIDList()
        # Set all traffic lights to 'off' mode
        for tl in traffic_lights:
            traci.trafficlight.setProgram(tl, "off")
    def reset(self):
        self.vehicle_ids = list()
        self.vid = 0
        # 设置random seed
        set_seed(32)

    def clear_all_vehicles(self, traci_connection):
        # 删除所有traci中的车辆
        for vehicle_id in traci_connection.vehicle.getIDList():
            traci_connection.vehicle.remove(vehicle_id)

    def generate_vehicle(self, vehicle_id, traci_connection):
        # 随机选择起点和终点
        # 随机选择两个不同的边作为起点和终点

        # 创建路线
        route_id = "route_" + vehicle_id
        while True:
            try:
                from_edge, to_edge = random.sample(self.edges, 2)
                # from_edge = random.choice(self.departure_edges)
                # to_edge = random.choice(self.arrival_edges)
                route = traci_connection.simulation.findRoute(from_edge, to_edge)
                while len(route.edges) == 0:
                    from_edge, to_edge = random.sample(self.edges, 2)
                    route = traci_connection.simulation.findRoute(from_edge, to_edge)
                traci_connection.route.add(route_id, route.edges)
                traci_connection.vehicle.add(vehicle_id, route_id, typeID="DEFAULT_VEHTYPE")
                break
            except traci_connection.exceptions.TraCIException as e:
                # 如果路线已经存在，就不再创建
                pass
        self.vehicle_ids.append(vehicle_id)
    @staticmethod
    def get_valid_edges(net_file):

        valid_edges = []
        net = sumolib.net.readNet(net_file)
        for e in net.getEdges():
            if e.allows("private"):
                valid_edges.append(e.getID())
        return valid_edges
    def set_vehicle_number(self, vnum):
        assert isinstance(vnum, int)
        self.n_veh = vnum
    def get_vehicle_number(self):
        return self.n_veh
    def manage_vehicles(self, traci_connection):
        # clear掉所有等待时间过长的车辆
        for vehicle_id in traci_connection.vehicle.getIDList():
            if traci_connection.vehicle.getWaitingTime(vehicle_id) > 60:
                traci_connection.vehicle.remove(vehicle_id)
        # 获取当前在路网中的所有车辆的ID
        current_vehicle_ids = list(traci_connection.vehicle.getIDList())
        self.vehicle_ids = current_vehicle_ids
        # 如果当前车辆数小于n_veh，生成新的车辆
        cnt = 0
        while len(self.vehicle_ids) < self.n_veh:
            cnt += 1
            new_vehicle_id = f'{self.vid}'
            self.vid += 1
            self.generate_vehicle(new_vehicle_id, traci_connection)
        # 删除多余的车辆
        for vehicle_id in current_vehicle_ids[self.n_veh:]:
            traci_connection.vehicle.remove(vehicle_id)