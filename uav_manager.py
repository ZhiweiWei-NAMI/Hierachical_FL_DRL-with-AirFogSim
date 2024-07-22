import pickle
import numpy as np
class UAVManager():
    def __init__(self, n_UAV, UAV_path_file):
        self.n_UAV = n_UAV
        # uav_trace [5, n_UAV, time_step, 3], time_step = 1000*50, 3 = [x, y, z]
        self.uav_trace_dataset_results = np.array(pickle.load(open(UAV_path_file, 'rb')))
        print('Episode step should mod 1000, x_min, x_max, y_min, y_max = 50, 400, 50, 300')
        self.current_time_step = -1


    def get_UAV_position(self, no_fly_type=0):
        self.current_time_step += 1
        self.current_time_step = self.current_time_step % 1000
        return self.uav_trace_dataset_results[no_fly_type, :, self.current_time_step, :]