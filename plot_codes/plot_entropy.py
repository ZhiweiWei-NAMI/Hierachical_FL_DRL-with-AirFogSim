
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# 从./data_entropy 读取run-{methods}-tag-Loss_Entropy_{idx}.csv文件，将每一个method的0~11的entropy绘制曲线的平均值标准差阴影图
methods = ['MAPPO_AggFed_cluster2_local_max12nRBselffree','MAPPO_AggFed_cluster2_max12nRBselffree','Certain_FedAvg_max12nRBselffree','MAPPO_nFL_max12nRBselffree','MAPPO_Cen_max12nRBselffree']
name=['AFedPPO-CCP, Ours', 'AFedPPO-C','AFedPPO-Avg','AFedPPO-only','Cen-PPO']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
linestypes = ['-', '--', '-.', ':','-']
# 要求长一点的图，所以figsize=(10, 6)
plt.figure()
# 每个csv有1500个Step，代表了5个episode。我需要在平滑的时候按照episode进行平滑，而不是step
for i, method in enumerate(methods):
    entropy = []
    for idx in range(12):
        df0 = pd.read_csv(f'./all_csv_data/run-{method}_0-tag-Loss_Entropy_{idx}.csv')
        df2 = pd.read_csv(f'./all_csv_data/run-{method}_2-tag-Loss_Entropy_{idx}.csv')
        df2['Step'] = df2['Step'] + 400
        df = pd.concat([df0, df2], ignore_index=True)
        df = df[df['Step'] < 800]
        entropy.append(df['Value'].values)
        # 判断df的行数是否为1200，如果不是，进行插值
        if len(df) < 800:
            # 创建插值函数，使用线性插值
            interp_func = interp1d(df['Step'], df['Value'], kind='linear', fill_value='extrapolate')
            # 创建新的Step数组
            new_steps = np.arange(800)
            # 插值得到新的Value数组
            new_values = interp_func(new_steps)
            entropy[-1] = new_values
    entropy = np.array(entropy)
    entropy_mean = np.mean(entropy, axis=0)
    entropy_std = np.std(entropy, axis=0)
    plt.plot(entropy_mean, label=name[i], color=colors[i], linestyle=linestypes[i])
    plt.fill_between(range(len(entropy_mean)), entropy_mean - entropy_std, entropy_mean + entropy_std, alpha=0.3, color=colors[i])
plt.xlabel('Training Epochs')
plt.ylabel('Entropy Distribution')
plt.title('Entropy Curve of 12 RSUs for 4 Deployments')
# 在x轴300, 600, 900, 1200处画竖线，虚线，表示一个episode，并且在150, 450, 750, 1050, y轴2.7处标注文字，表示一个Deployment 1234，要求居中
plt.axvline(200, linestyle='--', color='k')
plt.axvline(400, linestyle='--', color='k')
plt.axvline(600, linestyle='--', color='k')
plt.axvline(800, linestyle='--', color='k')
# 给文字加上边框和底色，要求灰色，alpha=0.5
al_pha = 0.1
plt.text(100, 2.7, 'Deployment 1', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(300, 2.7, 'Deployment 2', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(500, 2.7, 'Deployment 3', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))
plt.text(700, 2.7, 'Deployment 4', ha='center', bbox=dict(facecolor='gray', alpha=al_pha))

# x轴范围
plt.xlim(0, 800)
# xlabel间隔为150
plt.xticks(np.arange(0, 801, 100))
# y轴范围
plt.ylim(0, 3)
# y轴使用log
# plt.yscale('log')
# legent在右下方
plt.legend(loc='lower right')
# plt.grid 虚线
plt.grid(linestyle='--')
# grid
# savefig到../figure/entropy_curve.png
plt.savefig('../latex/png/entropy_curve.png',dpi=300,bbox_inches='tight')
plt.show()

