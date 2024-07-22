# 和PPO离散模型基本一致
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy

class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def forward(self, other_states, idx):
        batch_size, n_agent, state_dim = other_states.shape
        Q = self.query(other_states)
        K = self.key(other_states)
        V = self.value(other_states)
        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / np.sqrt(state_dim), dim=-1)
        out = attention_weights @ V
        out_idx = out[:, idx, :].view(batch_size, -1)
        return out_idx


class ValueNet(nn.Module):
    def __init__(self, input_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.attention = SelfAttention(input_states, n_hiddens)
        self.fc1 = nn.Linear(input_states+n_hiddens, n_hiddens)  # size is doubled due to concatenation of attention output
        self.fc2 = nn.Linear(n_hiddens, n_hiddens//2)
        self.fc3 = nn.Linear(n_hiddens//2, 1)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')

    def forward(self, x, other_states, idx):
        att = self.attention(other_states, idx)
        x = torch.cat((x, att), dim=1)  # concatenate along the feature dimension
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.fc3(x)
        return x


class PolicyNet(nn.Module):
    def __init__(self, input_states, n_hiddens, n_action1, n_action2, n_action3):
        super(PolicyNet, self).__init__()
        self.attention = SelfAttention(input_states, n_hiddens)
        self.fc1 = nn.Linear(input_states+n_hiddens, n_hiddens)  # size is doubled due to concatenation of attention output
        self.fc2 = nn.Linear(n_hiddens, n_hiddens//2)
        self.fc3 = nn.Linear(n_hiddens//2, n_action1)
        self.fc4 = nn.Linear(n_hiddens//2, n_action2)
        self.fc5 = nn.Linear(n_hiddens//2, n_action3)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')

    def forward(self, x, other_states, idx):
        att = self.attention(other_states, idx)
        x = torch.cat((x, att), dim=1)  # concatenate along the feature dimension
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x1 = F.softmax(self.fc3(x), dim=1)
        x2 = F.softmax(self.fc4(x), dim=1)
        x3 = F.softmax(self.fc5(x), dim=1)
        return x1, x2, x3


# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class PPO:
    def __init__(self, input_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, n_agent,
                 lmbda, eps, gamma, device):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.max_grad_norm = 2  # 梯度裁剪的阈值
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = lmbda  # 优势函数的缩放因子
        self.eps = eps  # ppo截断范围缩放因子
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.n_agent = n_agent
        self.n_actions = n_actions
        # 网络实例化
        self.actor = PolicyNet(input_states, n_hiddens, *n_actions).to(device)  # 策略网络
        self.critic = ValueNet(input_states, n_hiddens).to(device)  # 价值网络
        self.critic_tar = copy.deepcopy(self.critic)  
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.train_num = 0  # 训练次数
        self.epislon = 0.1
    
    # 动作选择
    def take_action(self, state, other_states, selfidx):  # 1xn_channelxHxW
        state = torch.tensor(state, dtype=torch.float).to(self.device)  
        other_states = torch.tensor(other_states, dtype=torch.float).to(self.device)
        probs1, probs2, probs3 = self.actor(state, other_states, selfidx)  # 当前状态的动作概率 [b,n_actions]
        probs1 = torch.clamp(probs1, 0, 1) + 1e-5 
        probs2 = torch.clamp(probs2, 0, 1) + 1e-5 
        probs3 = torch.clamp(probs3, 0, 1) + 1e-5 
        if np.random.rand() < self.epislon:
            probs1 = torch.rand_like(probs1).to(self.device)
            probs2 = torch.rand_like(probs2).to(self.device)
            probs3 = torch.rand_like(probs3).to(self.device)
        action_dist1 = torch.distributions.Categorical(probs1)  # 构造概率分布
        action1 = action_dist1.sample().item()  # 从概率分布中随机取样 int
        action_dist2 = torch.distributions.Categorical(probs2)  # 构造概率分布
        action2 = action_dist2.sample().item()  # 从概率分布中随机取样 int
        action_dist3 = torch.distributions.Categorical(probs3)  # 构造概率分布
        action3 = action_dist3.sample().item()  # 从概率分布中随机取样 int
        # 计算所有log_probs的和
        # log_probs = torch.log((probs1.gather(1, torch.tensor(action1).view(-1,1).to(self.device)) + probs2.gather(1, torch.tensor(action2).view(-1,1).to(self.device)) + probs3.gather(1, torch.tensor(action3).view(-1,1).to(self.device))) / 3  + 1e-5)
        log_probs = torch.log((probs1.gather(1, torch.tensor(action1).view(-1,1).to(self.device)) * probs2.gather(1, torch.tensor(action2).view(-1,1).to(self.device)) * probs3.gather(1, torch.tensor(action3).view(-1,1).to(self.device)))  + 1e-5)
        return action1, action2, action3, log_probs
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # 训练
    def update(self, transition_dict, batch_idx, transition_dict_all, agent_idx):
        self.train_num += 1
        if self.epislon > 0.1:
            self.epislon *= 0.9
        # 取出数据集
        states = torch.tensor(np.array(transition_dict['states'])[batch_idx], dtype=torch.float).to(self.device)  # [b,n_states]
        other_states = torch.tensor(np.array([transition_dict['states'] for transition_dict in transition_dict_all]), dtype=torch.float).to(self.device)
        other_states = other_states[:,batch_idx,:]
        other_states = other_states.transpose(0, 1)
        actions1 = torch.tensor(np.array(transition_dict['actions1'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device)  # [b,1]
        actions2 = torch.tensor(np.array(transition_dict['actions2'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device)  # [b,1]
        actions3 = torch.tensor(np.array(transition_dict['actions3'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device)  # [b,1]
        next_states = torch.tensor(np.array(transition_dict['next_states'])[batch_idx], dtype=torch.float).to(self.device)  # [b,n_states]
        selfidx = torch.tensor(agent_idx)
        
        next_other_states = torch.tensor(np.array([transition_dict['next_states'] for transition_dict in transition_dict_all]), dtype=torch.float).to(self.device)
        next_other_states = next_other_states[:,batch_idx,:]
        next_other_states = next_other_states.transpose(0, 1)
        dones = torch.tensor(np.array(transition_dict['dones'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(np.array(transition_dict['rewards'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        old_log_probs = torch.tensor(np.array(transition_dict['log_probs'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        # rewards = (rewards - rewards.mean()) / rewards.std()
        # 价值网络
        next_state_value = self.critic_tar(next_states, next_other_states, selfidx)  # 下一时刻的state_value  [b,1]
        td_target = rewards + self.gamma * next_state_value * (1-dones)  # 目标--当前时刻的state_value  [b,1]
        td_value = self.critic(states, other_states, selfidx)  # 预测--当前时刻的state_value  [b,1]
        td_delta = td_target - td_value  # 时序差分  # [b,1]

        # 计算GAE优势函数，当前状态下某动作相对于平均的优势
        advantage = 0  # 累计一个序列上的优势函数
        advantage_list = []  # 存放每个时序的优势函数值
        td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
        for delta in td_delta[::-1]:  # 逆序取出时序差分值
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)  # 保存每个时刻的优势函数
        advantage_list.reverse()  # 正序
        advantage_list = np.array(advantage_list)
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        # advantage = (advantage - advantage.mean()) / advantage.std()
        # 每一轮更新一次策略网络预测的状态
        probs1, probs2, probs3 = self.actor(states, other_states, selfidx)  # 当前状态的动作概率 [b,n_actions]

        tot_probs = (probs1.gather(1, actions1) * probs2.gather(1, actions2) * probs3.gather(1, actions3)) 
        # tot_probs = (probs1.gather(1, actions1) + probs2.gather(1, actions2) + probs3.gather(1, actions3)) / 3
        log_probs = torch.log(tot_probs + 1e-5)
        # 新旧策略之间的比例
        ratio = torch.exp(log_probs - old_log_probs)
        # 近端策略优化裁剪目标函数公式的左侧项
        surr1 = ratio * advantage
        # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
        entropy = -torch.sum(tot_probs * log_probs, dim=1)
        # 策略网络的损失函数
        actor_loss = torch.mean(-torch.min(surr1, surr2)) 
        # - 0 * entropy.mean()
        # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
        critic_loss = torch.mean(F.mse_loss(self.critic(states, other_states, selfidx), td_target.detach()))

        # 梯度清0
        # 反向传播
        # 梯度更新
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 剪裁梯度
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        self.soft_update(self.critic_tar, self.critic, tau=0.005)
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()