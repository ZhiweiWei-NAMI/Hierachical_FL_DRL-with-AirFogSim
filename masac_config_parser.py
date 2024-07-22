# config_parser.py

import argparse
# 包括Fed的才是执行FL的，nFL表示仅局部训练，不联邦；Cen表示集中式学习
potential_methods = ['MAPPO_Cen_nMA','MAPPO_nFL','MAPPO_Cen','greedy_notLearn','Certain_FedAvg', 'MAPPO_AggFed_cluster3', 'MAPPO_AggFed_cluster2', 'MAPPO_AggFed_cluster3_local','MAPPO_AggFed_cluster2_local','MAPPO_AggFed_cluster1_local']

method = 'MAPPO_Cen' # 聚类内部更新权重，聚类共享模型 or 本地模型根据相似度获取共享经验
n_cluster = int(method.split('cluster')[1].split('_')[0]) if 'cluster' in method else 1
print('n_cluster: ', n_cluster)
st_iter_id = 0
max_FL_RSU = 12
method += f'_max{max_FL_RSU}'
method += 'nRB'
method += 'self' # reward from self
method += 'free' # 不约束同一FogV/BS服务的数量
method += f'_{st_iter_id}'
useMA = False if 'nMA' in method else True
print('method: ', method)
def parse_arguments_for_MASAC():
    parser = argparse.ArgumentParser(description='示例程序')
    # useMA
    parser.add_argument('--useMA', type=bool, default=useMA)
    parser.add_argument('--to_train', type=bool, default=True)
    parser.add_argument('--draw_env_figure', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--n_cluster', type=int, default=n_cluster)
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--save_model', type=bool, default=True)
    parser.add_argument('--max_FL_RSU', type=int, default=max_FL_RSU)
    # 添加参数
    # 1. sumo文件所在地址的参数 
    # method_name
    parser.add_argument('--method', type=str, help=f'方法名，属于{potential_methods}', default = method)
    parser.add_argument('--img_path', type=str, help='存储icon的文件夹路径', default= "/home/weizhiwei/data/uav_compute/python_240102/icon")
    parser.add_argument('--sumocfg_path', type=str, help='sumocfg文件路径', default= "./sumo_berlin/map.sumocfg")
    parser.add_argument('--osm_path', type=str, help='osm文件路径', default= "./sumo_berlin/map.osm")
    parser.add_argument('--net_path', type=str, help='net文件路径', default= "./sumo_berlin/map.net.xml")
    parser.add_argument('--tensorboard_writer_file', type=str, help='tensorboard_writer_file', default= f"./runs3/{method}")
    parser.add_argument('--saved_path', type=str, help='saved_path', default= f"./models/{method}_models")
    # uav_path_file
    parser.add_argument('--UAV_path_file', type=str, help='UAV_path_file', default= f"./dataset/uav_trace_dataset_results.pkl")

    # 2. 仿真环境的参数 
    parser.add_argument('--random_seed', type=int, help='随机种子', default=42)
    parser.add_argument('--n_iter', type=int, help='迭代次数', default=2)
    parser.add_argument('--start_iter_id', type=int, help='开始迭代的方案id', default=st_iter_id)
    parser.add_argument('--n_episode', type=int, help='每次迭代的episode数，等价于sumo重启的次数', default=100)
    parser.add_argument('--max_steps', type=int, help='每个episode的最大步数', default=80)
    parser.add_argument('--n_veh', type=int, help='车辆数', default=200) 
    # 初始情况下TaskV fogV五五分,n_serving_veh
    parser.add_argument('--n_serving_veh', type=int, help='服务车辆数', default=100)
    parser.add_argument('--n_UAV', type=int, help='UAV数', default=40)
    parser.add_argument('--n_RSU', type=int, help='RSU数', default=12)
    parser.add_argument('--TTI_length', type=float, help='TTI的长度 (s)', default=0.05)
    parser.add_argument('--fading_threshold', type=float, help='fading_threshold', default=130)
    parser.add_argument('--UAV_communication_range', type=float, help='UAV_communication_range', default=300)
    parser.add_argument('--RSU_communication_range', type=float, help='RSU_communication_range', default=300)
    parser.add_argument('--V2V_communication_range', type=float, help='V2V_communication_range', default=100)
    parser.add_argument('--v_neighbor_Veh', type=int, help='v_neighbor_Veh', default=10)
    #load
    parser.add_argument('--load_terminated', type=bool, default=False)
    parser.add_argument('--load_without_training', type=bool, default=False)
    parser.add_argument('--fre_to_save', type=int, default=500)
    # draw_env_figure


    # 3. FL-DRL
    parser.add_argument('--n_hidden',type = int, default=128)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--omega', type=float, default=0.) # 个体奖励和协作奖励的比例，0代表只有个体奖励，1代表只有协作奖励
    parser.add_argument('--retreive_n_episode', type=int, default=1) # 从最近的X个经验中挑选2个进行训练 X=1
    parser.add_argument('--train_n_episode', type=int, default=5) # 每次训练的episode数
    parser.add_argument('--FL_fre', type=int, default=5)

    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--lmbda', type=float, default=0.95)
    parser.add_argument('--grid_width', type=int, default=5) # 代表网格的宽度
    
    # print parser中的参数
    args = parser.parse_args()
    print(args)
    # 解析命令行参数
    return args
