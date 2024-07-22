import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pickle

import numpy as np
# 搭建AE模型
import torch
import torch.nn as nn
from model_code.MyAutoEncoder import AutoEncoder
    
# 从./dataset/train_dataset.pkl_{iteration}读取数据集，一共是22个，分别是train_dataset和test_dataset
all_train_dataset = []
all_test_dataset = []
for i in range(0,2): 
    with open(f'./dataset/train_dataset.pkl_{i}', 'rb') as f:
        train_dataset = pickle.load(f)
    with open(f'./dataset/test_dataset.pkl_{i}', 'rb') as f:
        test_dataset = pickle.load(f)
    all_train_dataset.extend(train_dataset)
    all_train_dataset.extend(test_dataset[:len(test_dataset)//2])
    all_test_dataset.extend(test_dataset[len(test_dataset)//2:])
# 需要对test_dataset和train_dataset进行归一化
all_train_dataset = np.array(all_train_dataset)
all_test_dataset = np.array(all_test_dataset)
# 按照维度归一化。输入是[N, 120, 120, 7]，需要在最后的维度上进行归一化
# 输出各层的最大值和最小值
for i in range(7):
    max_val = np.max(all_train_dataset[:, :, :, i])
    min_val = np.min(all_train_dataset[:, :, :, i])
    all_train_dataset[:, :, :, i] = (all_train_dataset[:, :, :, i] - min_val) / (max_val - min_val)
    all_test_dataset[:, :, :, i] = (all_test_dataset[:, :, :, i] - min_val) / (max_val - min_val)
    print(f"Layer {i}, Max: {max_val}, Min: {min_val}")
    
# 假设 all_train_dataset 和 all_test_dataset 已经是加载到内存中的 numpy arrays
train_data = torch.tensor(all_train_dataset, dtype=torch.float32)
test_data = torch.tensor(all_test_dataset, dtype=torch.float32)

# 创建 DataLoader
batch_size = 20  # 可以根据机器的内存/显存调整
train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data), batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# tqdm
from tqdm import tqdm
# 训练过程
n_epochs = 100  # 根据需要调整
# 使用tqdm显示训练进度
# 将每一轮的loss写入txt文件。覆写
f = open('loss3.txt', 'w')
isTest = False
if not isTest:
    min_loss = 1000
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{n_epochs}", unit='batch') as pbar:
            for data in train_loader:
                inputs = data[0].to(device)
                inputs = inputs.permute(0, 3, 1, 2) # (N, 120, 120, 7) -> (N, 7, 120, 120)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                loss = loss.item()
                min_loss = min(min_loss, loss)
                pbar.set_postfix(loss=loss, train_loss=train_loss/len(train_loader))
                pbar.update(1)
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"./models/3dautoencoder_{epoch}.pth")
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in test_loader:
                inputs = data[0].to(device)
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                test_loss += loss.item()
        test_loss /= len(test_loader)
        
        f.write(f"{epoch}: {train_loss/len(train_loader)}: {test_loss}\n")
        print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader)}, Test Loss: {test_loss}")