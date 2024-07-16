
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
methods = ['MAPPO_Cen_max12nRBselffree','MAPPO_AggFed_cluster2_local_max12nRBselffree','MAPPO_AggFed_cluster2_max12nRBselffree','Certain_FedAvg_max12nRBselffree','MAPPO_nFL_max12nRBselffree','greedy_notLearn_max12nRBselffree']
name=['Cen-PPO','AFedPPO-CCP, Ours', 'AFedPPO-C','AFedPPO-Avg','AFedPPO-only','Greedy']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestypes = ['-', '--', '-.', ':','-', '--']
plt.figure(figsize=(10, 6))
max_step = 32000  # 我们关心的最大step数
last_step_start = 7000
task_cnt_file0 = 'run-MAPPO_nFL_max12nRBselffree_0-tag-Num_Task_Num_'
task_cnt_file2 = 'run-MAPPO_nFL_max12nRBselffree_2-tag-Num_Task_Num_'
for i, method in enumerate(methods):
    # 需要按照Step这个列，把0~9000, 9000~18000, 18000~27000, 27000~36000的数据分别求平均，按照task_cnt来求，所以需要先求和
    df_utility0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Reward_Total.csv')
    df_utility2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Reward_Total.csv')
    df_utility2['Step'] = df_utility2['Step'] + 16000
    df_utility = pd.concat([df_utility0, df_utility2], ignore_index=True)
    utility_for_dep = [0 for _ in range(4)]
    latency_for_dep = [0 for _ in range(4)]
    succ_ratio_for_dep = [0 for _ in range(4)]
    locally_exe_ratio_for_dep = [0 for _ in range(4)]
    task_cnt_for_dep = [0 for _ in range(4)]
    certain_task_cnt_for_dep = [[0 for _ in range(12)] for _ in range(4)] # 真正的task个数
    for rsu_id in range(12):
        df_certain_task_num0 = pd.read_csv(f'./all_csv_data/{task_cnt_file0}{rsu_id}.csv')
        df_certain_task_num2 = pd.read_csv(f'./all_csv_data/{task_cnt_file2}{rsu_id}.csv')
        df_certain_task_num2['Step'] = df_certain_task_num2['Step'] + 16000
        df_certain_task_num = pd.concat([df_certain_task_num0, df_certain_task_num2], ignore_index=True)
        for j in range(4):
            # task_cnt记录行数，即task的个数
            if rsu_id == 0:
                task_cnt_for_dep[j] += len(df_certain_task_num[df_certain_task_num['Step'] >= j * 8000+last_step_start][df_certain_task_num['Step'] < (j + 1) * 8000])
            certain_task_cnt_for_dep[j][rsu_id] = df_certain_task_num[df_certain_task_num['Step'] >= j * 8000+last_step_start][df_certain_task_num['Step'] < (j + 1) * 8000]['Value'].values.sum() / task_cnt_for_dep[j]
    certain_task_cnt_for_dep = np.array(certain_task_cnt_for_dep) # 4*12
    # 每个deployment中，将每个RSU的task个数除以总（12个RSU）task个数，得到每个RSU的task比例
    certain_task_ratio_for_dep = np.zeros((4, 12))
    # for rsu_id in range(12):
    df_latency0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Latency_Total_Avg_Latency.csv')
    df_latency2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Latency_Total_Avg_Latency.csv')
    df_latency2['Step'] = df_latency2['Step'] + 16000
    df_latency = pd.concat([df_latency0, df_latency2], ignore_index=True)
    df_failed_task_ratio0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Ratio_Total_Failed_Ratio.csv')
    df_failed_task_ratio2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Ratio_Total_Failed_Ratio.csv')
    df_failed_task_ratio2['Step'] = df_failed_task_ratio2['Step'] + 16000
    df_failed_task_ratio = pd.concat([df_failed_task_ratio0, df_failed_task_ratio2], ignore_index=True)
    df_offloaded_succ_ratio0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Ratio_Total_Offloaded_Ratio.csv')
    df_offloaded_succ_ratio2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Ratio_Total_Offloaded_Ratio.csv')
    df_offloaded_succ_ratio2['Step'] = df_offloaded_succ_ratio2['Step'] + 16000
    df_offloaded_succ_ratio = pd.concat([df_offloaded_succ_ratio0, df_offloaded_succ_ratio2], ignore_index=True)
    for j in range(4):
        # 求和
        latency_for_dep[j] += df_latency[df_latency['Step'] >= j * 8000+last_step_start][df_latency['Step'] < (j + 1) * 8000]['Value'].values.sum()
        locally_exe_ratio_for_dep[j] += df_offloaded_succ_ratio[df_offloaded_succ_ratio['Step'] >= j * 8000+last_step_start][df_offloaded_succ_ratio['Step'] < (j + 1) * 8000]['Value'].values.sum()
        succ_ratio_for_dep[j] += df_failed_task_ratio[df_failed_task_ratio['Step'] >= j * 8000+last_step_start][df_failed_task_ratio['Step'] < (j + 1) * 8000]['Value'].values.sum()
    # utility直接读取的是total，所以不需要for循环，直接求和
    for j in range(4):
        utility_for_dep[j] = df_utility[df_utility['Step'] >= j * 8000+last_step_start][df_utility['Step'] < (j + 1) * 8000]['Value'].values.sum()
    # 计算对于每个method，这四个属性的平均值
    utility_for_dep = np.array(utility_for_dep) / np.array(task_cnt_for_dep)
    latency_for_dep = np.array(latency_for_dep) / np.array(task_cnt_for_dep)
    succ_ratio_for_dep = (1- np.array(succ_ratio_for_dep) / np.array(task_cnt_for_dep))*100
    offload_exe_ratio_for_dep = ((np.array(locally_exe_ratio_for_dep)) / np.array(task_cnt_for_dep))*100
    if i == 0:
        cen_utility = utility_for_dep
        cen_latency = latency_for_dep
        cen_succ_ratio = succ_ratio_for_dep
        cen_offload_exe_ratio = offload_exe_ratio_for_dep
    print(f'Method {name[i]}:')
    # 按照deployement来print
    for j in range(4):
        # print(f'Deployment {j + 1}: Utility={utility_for_dep[j]} ({100*(utility_for_dep[j]-cen_utility[j])/max(1e-5,cen_utility[j])}%), Latency={latency_for_dep[j]} ({100*(latency_for_dep[j]-cen_latency[j])/max(1e-5,cen_latency[j])}%), Succ. Ratio={succ_ratio_for_dep[j]} ({100*(succ_ratio_for_dep[j]-cen_succ_ratio[j])/max(1e-5,cen_succ_ratio[j])}%), Off. Ratio={offload_exe_ratio_for_dep[j]} ({100*(offload_exe_ratio_for_dep[j]-cen_offload_exe_ratio[j])/max(1e-5,cen_offload_exe_ratio[j])}%)')
        print(f'Deployment {j + 1}: Utility={utility_for_dep[j]}, Latency={latency_for_dep[j]}, Succ. Ratio={succ_ratio_for_dep[j]}, Off. Ratio={offload_exe_ratio_for_dep[j]}')
