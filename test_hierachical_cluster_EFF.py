import pickle
from model_code.MyAutoEncoder import AutoEncoder
import torch
import numpy as np

with open(f'./dataset/test_dataset.pkl_2', 'rb') as f:
    test_dataset = pickle.load(f)
# Layer 0, Max: 232.68, Min: 0.0
# Layer 1, Max: 13.89, Min: 0.0
# Layer 2, Max: 1.0, Min: 0.0
# Layer 3, Max: 1.0, Min: 0.0
# Layer 4, Max: 3.0, Min: 0.0
# Layer 5, Max: 1.0, Min: 0.0
# Layer 6, Max: 2.0, Min: 0.0
# 归一化
max_val = [232.68, 13.89, 1.0, 1.0, 3.0, 1.0, 2.0]
test_dataset = np.array(test_dataset)
for i in range(7):
    test_dataset[:, :, :, i] = test_dataset[:, :, :, i] / max_val[i]
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AE_model = AutoEncoder().to(device)
AE_model.load_state_dict(torch.load('./models/autoencoder_99.pth'))
# 加载完成之后，主要使用encode的部分，获取特征[64]
# 假设test_dataset中每12个元素是一个时刻下的RSU状态。我需要先将这些状态转换为特征向量，然后再使用KMeans聚类算法对这些特征向量进行聚类

# 1. 将test_dataset中的每个时刻的RSU状态转换为特征向量
# 全部喂给AE_model，然后获取特征向量
test_data = torch.tensor(test_dataset, dtype=torch.float32)
test_data = test_data.to(device).permute(0, 3, 1, 2) # (N, 7, 120, 120)
AE_model.eval()
with torch.no_grad():
    test_data = AE_model.encode(test_data)
test_data = test_data.cpu().numpy()
# reshape
old_test_data = test_data.reshape(-1, 12, 64) # (N, 12, 64)

# test_data在最后一个维度，64添加两个维度，一个是任务数量，一个是完成率
test_data = np.zeros((old_test_data.shape[0], 12, 64+2))
for i in range(test_data.shape[0]):
    test_data[i, :, :64] = old_test_data[i]
    t1 = np.random.rand(12)
    t2 = np.random.rand(12)
    test_data[i, :, 64] = t1
    test_data[i, :, 65] = t2
    
# 按照每5*12为一组，5代表了步长，12代表了RSU的数量
test_data = test_data.reshape(-1, 5, 12, 66)

# 由于是时序的，也就是说每个时刻其实都是由 S->A*S，其中A是状态转移矩阵。现在要做的就是用把每一个N内，t和t+1步长的数据，左乘S_t^T/||S_t||，用来表示A在这个方向上的影响；然后5个步长求平均，得到一个平均的A的影响
feature_matrix = np.zeros((test_data.shape[0], 12, 3))
for i in range(test_data.shape[0]):
    feature_tmp = np.zeros((12,3))
    amax = -100*np.ones(12)
    amax_idx = np.zeros(12)
    for j in range(4):
        for rsu_id in range(12):
            S_t = test_data[i, j, rsu_id, :64]
            S_t1 = test_data[i, j+1, rsu_id, :64]
            S_t = S_t / np.linalg.norm(S_t)
            A = np.matmul(S_t.T, S_t1) # [1]
            # A相当于特征值，求最大的特征值作为特征向量
            # print(A.shape)
            if A.max() > amax[rsu_id]:
                amax[rsu_id] = A.max()
                amax_idx[rsu_id] = j
                feature_tmp[rsu_id, 0] = A
                feature_tmp[rsu_id, 1] = test_data[i, j, rsu_id, 64]
                feature_tmp[rsu_id, 2] = test_data[i, j, rsu_id, 65]
    feature_matrix[i, :, :] = feature_tmp
    
def plot_dendrogram(model, **kwargs):
    #  创建链接矩阵，然后绘制树状图 
    #  创建每个节点的样本计数 
    counts = np.zeros(model.children_.shape[0]) 
    n_samples = len(model.labels_)  
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  #  叶子节点 
            else:
                current_count += counts[child_idx-n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    print(linkage_matrix)
    #  绘制相应的树状图 
    dendrogram(linkage_matrix, **kwargs)

# 2. 使用层次聚类，对每个时刻的RSU状态进行聚类
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
# N, samples_per_step, features = test_data.shape
from matplotlib import pyplot as plt
# 存储每个时间点的聚类结果，由于不预先知道聚类数，使用list存储
cluster_labels = []
from scipy.cluster.hierarchy import dendrogram, linkage
n_cluster = 3
from scipy.cluster.hierarchy import fcluster
for i in range(5):
    # 对每个时间步的数据进行聚类
    rsu_groups1 = feature_matrix[i*2, :, :]
    rsu_groups2 = feature_matrix[i*2+1, :, :]
    rsu_groups = np.concatenate((rsu_groups1, rsu_groups2), axis=0)
    # rsu_groups = rsu_groups1
    
    # 绘制rsu_groups的第一维度的坐标
    fig = plt.figure()
    plt.scatter(range(24), rsu_groups[:, 0])
    plt.xlabel('RSU ID')

    plt.savefig(f'./plots/feature_points_2d_{i}.png', dpi=300)


    clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = clustering.fit_predict(rsu_groups)
    # plot_dendrogram(clustering, truncate_mode='level', p=3)
    # plt.savefig(f'./plots/hierachical_clustering_{i}.png')
    print(labels)


# 转换成 array 以方便进一步处理，每个元素的长度可能不同
# cluster_labels = np.array(cluster_labels, dtype=object)

# print("Cluster labels:", cluster_labels)

