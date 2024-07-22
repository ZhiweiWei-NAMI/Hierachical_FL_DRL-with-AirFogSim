# 和PPO离散模型基本一致
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_states, n_action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_states, 16)
        self.fc2 = nn.Linear(16, n_action_dim[0])
        self.fc3 = nn.Linear(16, n_action_dim[1])

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        action1 = F.softmax(self.fc2(x), dim = 1)
        action2 = F.softmax(self.fc3(x), dim = 1)
        action3 = torch.zeros((x.size(0), 1)).to(action1.device)
        return action1, action2, action3


class Critic(nn.Module):
    def __init__(self, input_states, action_dim, n_agent):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.fc1 = nn.Linear((input_states + action_dim) * n_agent, 64)  # Add action dimension to the linear layer input
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state, action1, action2):
        x = torch.cat([state, action1, action2], dim=1)  # Concatenate state and action tensors along dimension 1
        x = F.leaky_relu(self.fc1(x), 0.01)
        Q = self.fc2(x)
        return Q


# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class QMIX:
    def __init__(self, input_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, n_agent,
                 lmbda, eps, gamma, device):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.max_grad_norm = 0.5  # 梯度裁剪的阈值
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = lmbda  # 优势函数的缩放因子
        self.eps = eps  # ppo截断范围缩放因子
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.n_agent = n_agent
        self.n_actions = n_actions
        # 网络实例化
        self.actors = [Actor(input_states, n_actions).to(device) for _ in range(n_agent)]
        self.critic = Critic(input_states, n_actions[0]+n_actions[1], n_agent).to(device)
        self.critic2 = Critic(input_states, n_actions[0]+n_actions[1], n_agent).to(device)

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.critic_optimizer2 = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.train_num = 0  # 训练次数
        self.epislon = 1
    
    # 动作选择
    def take_action(self, state, agent_idx):  # 1xn_channelxHxW
        state = torch.tensor(state, dtype=torch.float).to(self.device)  
        probs1, probs2, probs3 = self.actors[agent_idx](state)  # 当前状态的动作概率 [b,n_actions]
        noise1 = 0.3 * torch.randn(probs1.size()).to(self.device)  # 生成标准正态分布的随机数
        probs1 = torch.clip(probs1 + self.epislon * noise1, 0, 1) + 1e-5  # 加入噪声
        noise2 = 0.3 * torch.randn(probs2.size()).to(self.device)  # 生成标准正态分布的随机数
        probs2 = torch.clip(probs2 + self.epislon * noise2, 0, 1) + 1e-5  # 加入噪声
        if np.random.rand() <= self.epislon:
            probs1 = torch.rand_like(probs1)
            probs2 = torch.rand_like(probs2)
        action_dist1 = torch.distributions.Categorical(probs1)  # 构造概率分布
        action1 = action_dist1.sample().item()  # 从概率分布中随机取样 int
        action_dist2 = torch.distributions.Categorical(probs2)  # 构造概率分布
        action2 = action_dist2.sample().item()  # 从概率分布中随机取样 int
        
        # 计算所有log_probs的和
        log_probs = torch.log(probs1.gather(1, torch.tensor(action1).view(-1,1).to(self.device)) * probs2.gather(1, torch.tensor(action2).view(-1,1).to(self.device))+1e-5)
        return action1, action2, 0, log_probs
    
    # 训练
    def update(self, transition_dict, batch_idx):
        self.train_num += 1
        if self.train_num > 100 and self.epislon > 0.1:
            self.epislon *= 0.99
        # 取出数据集
        states = [torch.tensor(np.array(transition_dict_i['states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [b,n_states]
        
        actions1 = [torch.tensor(np.array(transition_dict_i['actions1'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        actions1 = torch.cat(actions1, dim=1)  # [b,n_agent]
        actions2 = [torch.tensor(np.array(transition_dict_i['actions2'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        actions2 = torch.cat(actions2, dim=1)  # [b,n_agent]
        next_states = [torch.tensor(np.array(transition_dict_i['next_states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b, n_channel, H, W]
        # next_states_critic 只需要第一个agent的n_channel中的前n-1个通道，是4维
        next_states_critic = torch.cat(next_states, dim = 1)
        states_critic = torch.cat(states, dim = 1)
        dones = torch.tensor(np.array(transition_dict[0]['dones'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = [torch.tensor(np.array(transition_dict_i['rewards'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        # rewards 求和
        rewards = torch.sum(torch.cat(rewards, dim=1), dim=1).view(-1,1)  # [b,1]
        all_next_actions = [self.actors[i](next_states[i]) for i in range(self.n_agent)]
        all_next_actions1 = [all_next_actions[i][0].detach() for i in range(self.n_agent)]
        all_next_actions2 = [all_next_actions[i][1].detach() for i in range(self.n_agent)]
        next_actions1 = torch.cat(all_next_actions1, dim=1)  # [b, n_agent*n_actions]
        next_actions2 = torch.cat(all_next_actions2, dim=1)  # [b, n_agent*n_actions]
        Q_targets_next = self.critic(next_states_critic, next_actions1, next_actions2)
        # Q_targets_next2 = self.critic2(next_states_critic, next_actions1, next_actions2)
        # Q_targets = rewards + (self.gamma * torch.min(Q_targets_next, Q_targets_next2) * (1 - dones))
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_targets = Q_targets.detach()
        # actions1是 [b, 1]，用onehot把他变成[b, n_actions]
        actions1 = actions1.unsqueeze(2)
        actions1_onehot = torch.zeros(actions1.size(0), self.n_agent, self.n_actions[0]).to(self.device)
        actions1_onehot.scatter_(2, actions1, 1)
        actions2 = actions2.unsqueeze(2)
        actions2_onehot = torch.zeros(actions2.size(0), self.n_agent, self.n_actions[1]).to(self.device)
        actions2_onehot.scatter_(2, actions2, 1)
        # 把actions1_onehot每个agent的动作合并，变成[b, n_agent*n_actions]
        Q_expected = self.critic(states_critic, actions1_onehot.view(actions1_onehot.size(0), -1), actions2_onehot.view(actions2_onehot.size(0), -1))
        # Q_expected2 = self.critic2(states_critic, actions1_onehot.view(actions1_onehot.size(0), -1), actions2_onehot.view(actions2_onehot.size(0), -1))

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # critic_loss2 = F.mse_loss(Q_expected2, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # self.critic_optimizer2.zero_grad()
        # critic_loss2.backward()
        # self.critic_optimizer2.step()
        for i in range(self.n_agent):
            self.actor_optimizers[i].zero_grad()
            # 只改变第i个agent的actor的action
            action1_i, action2_i, action3_i = self.actors[i](states[i])  # [b,1,n_actions]
            action1_i = action1_i.unsqueeze(1)
            action2_i = action2_i.unsqueeze(1)
            tmp1 = actions1_onehot.clone()
            tmp1[:, i*self.n_actions[0]:(i+1)*self.n_actions[0], :] = action1_i
            tmp1 = tmp1.view(tmp1.size(0), -1)
            tmp2 = actions2_onehot.clone()
            tmp2[:, i*self.n_actions[1]:(i+1)*self.n_actions[1], :] = action2_i
            tmp2 = tmp2.view(tmp2.size(0), -1)
            Q_expected = self.critic(states_critic, tmp1, tmp2)
            # Q_expected2 = self.critic2(states_critic, tmp1, tmp2)
            # actor_loss = -(torch.min(Q_expected.mean(), Q_expected2.mean()))
            actor_loss = - Q_expected.mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()