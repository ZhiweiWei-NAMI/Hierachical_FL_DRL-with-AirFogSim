
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
methods = ['MAPPO_Cen_max12nRBselffree','MAPPO_Cen_nMA_max12nRBselffree']
name=['Cen-PPO','Cen-PPO without Attention']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestypes = ['-', '--', '-.', ':','-', '--']
plt.figure()
max_step = 32000  # 我们关心的最大step数
window_size = 1000
for i, method in enumerate(methods):
    entropy = []
    interpolated_values = []
    df0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Reward_Total.csv')
    df2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Reward_Total.csv')
    # 把df0和df2合成df，注意df2的Step需要加上16000
    df2['Step'] = df2['Step'] + 16000
    df = pd.concat([df0, df2], ignore_index=True)
    df = df[df['Step'] < max_step]
    # 如果不足1000步，进行插值
    if len(df) < max_step:
        # 创建插值函数，使用线性插值
        interp_func = interp1d(df['Step'], df['Value'], kind='linear', fill_value='extrapolate')
        # 创建新的Step数组
        new_steps = np.arange(max_step)
        # 插值得到新的Value数组
        new_values = interp_func(new_steps)
        interpolated_values = new_values
    else:
        interpolated_values = df['Value'].values
    reward = np.array(interpolated_values)
    reward_pd1 = pd.DataFrame(reward[:8000], columns=['Reward'])
    reward_pd1['Smoothed'] = reward_pd1['Reward'].rolling(window=window_size, center=True).mean()
    reward_pd1 = reward_pd1['Smoothed']
    reward_pd2 = pd.DataFrame(reward[8000:16000], columns=['Reward'])
    reward_pd2['Smoothed'] = reward_pd2['Reward'].rolling(window=window_size, center=True).mean()
    reward_pd2 = reward_pd2['Smoothed']
    reward_pd3 = pd.DataFrame(reward[16000:24000], columns=['Reward'])
    reward_pd3['Smoothed'] = reward_pd3['Reward'].rolling(window=window_size, center=True).mean()
    reward_pd3 = reward_pd3['Smoothed']
    reward_pd4 = pd.DataFrame(reward[24000:32000], columns=['Reward'])
    reward_pd4['Smoothed'] = reward_pd4['Reward'].rolling(window=window_size, center=True).mean()
    reward_pd4 = reward_pd4['Smoothed']
    # 把四个部分合并，注意需要把每个部分加上自己的offset
    reward = np.concatenate([reward_pd1.values, reward_pd2.values, reward_pd3.values, reward_pd4.values])
    plt.plot(reward, label=name[i], color=colors[i], linestyle=linestypes[i])
plt.xlabel('Simulation Steps (s)')
plt.ylabel('Total Reward (Task Utility)')
plt.title(f'Total Reward Curve (Smoothed with Window={window_size})')
plt.xlim(0, 32000)
# x轴 3000, 6000, ..., 36000
plt.xticks(np.arange(0, 32001, 4000))
# x轴用科学计数法，乘以1000
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))

plt.axvline(8000, linestyle='--', color='k')
plt.axvline(16000, linestyle='--', color='k')
plt.axvline(24000, linestyle='--', color='k')
plt.axvline(32000, linestyle='--', color='k')
al_pha = 0.1
plt.text(4000, 65, 'Deployment 1', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(12000, 65, 'Deployment 2', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(20000, 65, 'Deployment 3', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(28000, 65, 'Deployment 4', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
# y轴范围
plt.ylim(-10, 70)
# y轴使用log
# plt.yscale('log')
plt.legend(loc='lower right')
# plt.grid 虚线
plt.grid(linestyle='--')
# grid
# savefig到../figure/entropy_curve.png
plt.savefig('../latex/png/reward_curve_nMA.png',dpi=300,bbox_inches='tight')
plt.show()

