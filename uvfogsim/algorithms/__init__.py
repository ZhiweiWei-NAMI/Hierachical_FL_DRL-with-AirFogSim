# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2023/05/18
@Author  :   Zhiwei Wei
@Contact :   2031563@tongji.edu.cn
@Site    :   https://github.com/Zhiwei-Wei
'''



# 在__init__.py中
# from .Base_Algorithm_Module import Base_Algorithm_Module
# from .Weighted_Voronoi_Algorithm_Module import Weighted_Voronoi_Algorithm_Module
# from .CNN_Weighted_Voronoi_Algorithm_Module import CNN_Weighted_Voronoi_Algorithm_Module
# from .CNN_MADDPG_Weighted import CNN_MADDPG_Weighted
# from .Random_Algorithm_Module import Random_Algorithm_Module
# from .Cluster_Algorithm_Module import Cluster_Algorithm_Module
# from .KM_Area_Module import KM_Area_Module
# from .PSO_Algorithm_Module import PSO_Algorithm_Module
# from ..environment import Environment
import random
import numpy as np
import torch
def set_seed(seed = 52):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
