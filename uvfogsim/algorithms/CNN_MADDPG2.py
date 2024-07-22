# 和PPO离散模型基本一致
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy
class ConvNet(nn.Module):
    def __init__(self, n_channel = 4, hidden_channel = 8):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, hidden_channel, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, int(hidden_channel*2), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(hidden_channel*2), int(hidden_channel*2), kernel_size=3, stride=1, padding=1, dilation=2)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(289 * hidden_channel * 2), 512)  
        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.leaky_relu(x)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.leaky_relu(x)
        # x = self.dropout2(x)
        return x




class Critic(nn.Module):
    def __init__(self, input_states, n_action_dim, n_hiddens):
        super(Critic, self).__init__()
        self.conv = ConvNet(n_channel=24, hidden_channel=16)
        self.fc1 = nn.Linear(584, 256)  # size is doubled due to concatenation of attention output
        self.fc2 = nn.Linear(256, 1)
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def forward(self, x, actions1, actions2, pos):
        actions = torch.cat([actions1, actions2, pos], dim=-1)
        x1 = self.conv(x)
        x2 = torch.cat((x1, actions), dim=1)  # concatenate along the feature dimension
        x3 = F.leaky_relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return x4


class Actor(nn.Module):
    def __init__(self, input_states, n_hiddens, n_action1, n_action2):
        super(Actor, self).__init__()
        self.n_UAV = 4
        self.n_agent = 6
        self.conv = ConvNet()
        self.fc1 = nn.Linear(512+3, 256)  # size is doubled due to concatenation of attention output
        self.fc2 = nn.ModuleList([nn.Linear(256, n_action1) for _ in range(self.n_UAV)])
        self.fc3 = nn.ModuleList([nn.Linear(256, n_action2) for _ in range(self.n_agent)])
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        for fc2t in self.fc2:
            torch.nn.init.kaiming_normal_(fc2t.weight, nonlinearity='leaky_relu')
        for fc3t in self.fc3:
            torch.nn.init.kaiming_normal_(fc3t.weight, nonlinearity='leaky_relu')

    def forward(self, x, idx, pos):
        tidx = idx.detach().cpu().numpy()[0,0].astype(np.int16)
        x1 = self.conv(x)
        x2 = torch.cat((x1, idx, pos), dim=1)  # concatenate along the feature dimension
        x3 = F.leaky_relu(self.fc1(x2))
        if tidx < self.n_UAV:
            out1 = F.softmax(self.fc2[tidx](x3), dim=1)
            out2 = F.softmax(self.fc3[tidx](x3), dim=1)
            return out1, out2
        out2 = F.softmax(self.fc3[tidx](x3), dim=1)
        return out2


# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class CNN_MADDPG:
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
        self.n_UAV = 4
        # 网络实例化
        self.actor = Actor(input_states, n_hiddens, *n_actions).to(device)  # 策略网络
        self.critic = Critic(input_states, n_actions[0] * self.n_UAV + self.n_agent * n_actions[1], n_hiddens).to(device)  # 价值网络
        self.critic_tar = copy.deepcopy(self.critic)  
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.train_num = 0  # 训练次数
        self.epislon = 1
    
    # 动作选择
    def take_action(self, state, idx, pos):  # 1xn_channelxHxW
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        selfidx = torch.tensor([idx], dtype=torch.float).to(self.device).view(1,-1)
        pos = torch.tensor(pos, dtype=torch.float).to(self.device).view(1, -1)
        if idx < self.n_UAV:
            probs1, probs2 = self.actor(state, selfidx, pos)  # 当前状态的动作概率 [b,n_actions]
            probs1 = torch.clamp(probs1, 0, 1) + 1e-5 
            probs2 = torch.clamp(probs2, 0, 1) + 1e-5 
            if np.random.rand() < self.epislon:
                probs1 = torch.rand_like(probs1).to(self.device)
                probs2 = torch.rand_like(probs2).to(self.device)
            action_dist1 = torch.distributions.Categorical(probs1)  # 构造概率分布
            action1 = action_dist1.sample().item()  # 从概率分布中随机取样 int
            action_dist2 = torch.distributions.Categorical(probs2)  # 构造概率分布
            action2 = action_dist2.sample().item()  # 从概率分布中随机取样 int
            log_probs = torch.log((probs1.gather(1, torch.tensor(action1).view(-1,1).to(self.device)) * probs2.gather(1, torch.tensor(action2).view(-1,1).to(self.device)))  + 1e-5)
            return action1, action2, log_probs
        else:
            probs2 = self.actor(state, selfidx, pos)  # 当前状态的动作概率 [b,n_actions]
            probs2 = torch.clamp(probs2, 0, 1) + 1e-5 
            if np.random.rand() < self.epislon:
                probs2 = torch.rand_like(probs2).to(self.device)
            action_dist2 = torch.distributions.Categorical(probs2)  # 构造概率分布
            action2 = action_dist2.sample().item()  # 从概率分布中随机取样 int
            log_probs = torch.log((probs2.gather(1, torch.tensor(action2).view(-1,1).to(self.device)))  + 1e-5)
            return action2, log_probs

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # 训练
    def update(self, transition_dict, batch_idx, selfidx):
        self.train_num += 1
        if self.epislon > 0.1:
            self.epislon *= 0.9
        # 取出数据集
        states = [torch.tensor(np.array(transition_dict_i['states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [b,n_states]
        pos_states = [torch.tensor(np.array(transition_dict_i['pos_states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [b,n_states]
        new_pos_states = [torch.tensor(np.array(transition_dict_i['new_pos_states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [b,n_states]
        actions1 = [torch.tensor(np.array(transition_dict_i['actions1'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        actions1 = torch.cat(actions1, dim=1)  # [b,n_agent]
        actions2 = [torch.tensor(np.array(transition_dict_i['actions2'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        actions2 = torch.cat(actions2, dim=1)  # [b,n_agent]
        next_states = [torch.tensor(np.array(transition_dict_i['next_states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b, n_channel, H, W]
        # next_states_critic 只需要第一个agent的n_channel中的前n-1个通道，是4维
        next_states_critic = torch.cat(next_states, dim = 1)
        states_critic = torch.cat(states, dim = 1)
        pos_states_critic = torch.cat(pos_states, dim = 1)
        new_pos_states_critic = torch.cat(new_pos_states, dim = 1)
        dones = torch.tensor(np.array(transition_dict[0]['dones'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = [torch.tensor(np.array(transition_dict_i['rewards'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device) for transition_dict_i in transition_dict]  # [n_agent, b,1]
        # rewards 求和
        rewards = torch.sum(torch.cat(rewards, dim=1), dim=1).view(-1,1)  # [b,1] 最大化所有agent当中最少的奖励
        all_next_actions = [self.actor(next_states[i], (torch.ones(batch_idx.shape).to(self.device) * i).view(-1, 1), new_pos_states[i]) for i in range(self.n_UAV)]
        all_next_bs_actions = [self.actor(next_states[i], (torch.ones(batch_idx.shape).to(self.device) * i).view(-1, 1), new_pos_states[i]).detach() for i in range(self.n_UAV, self.n_agent)]
        all_next_actions1 = [all_next_actions[i][0].detach() for i in range(self.n_UAV)]
        all_next_actions2 = [all_next_actions[i][1].detach() for i in range(self.n_UAV)]
        all_next_actions2.extend(all_next_bs_actions)
        next_actions1 = torch.cat(all_next_actions1, dim=1)  # [b, n_agent*n_actions]
        next_actions2 = torch.cat(all_next_actions2, dim = 1)
        Q_targets_next = self.critic_tar(next_states_critic, next_actions1, next_actions2, new_pos_states_critic)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_targets = Q_targets.detach()
        actions1 = actions1[:,:self.n_UAV].unsqueeze(2)
        actions1_onehot = torch.zeros(actions1.size(0), self.n_UAV, self.n_actions[0]).to(self.device)
        actions1_onehot.scatter_(2, actions1, 1)
        actions2 = actions2.unsqueeze(2)
        actions2_onehot = torch.zeros(actions2.size(0), self.n_agent, self.n_actions[1]).to(self.device)
        actions2_onehot.scatter_(2, actions2, 1)
        Q_expected = self.critic(states_critic, actions1_onehot.view(actions1_onehot.size(0), -1), actions2_onehot.view(actions2_onehot.size(0), -1), new_pos_states_critic)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        i = selfidx
        self.actor_optimizer.zero_grad()
        tmp1 = actions1_onehot.clone()
        # 只改变第i个agent的actor的action
        if i < self.n_UAV:
            action1_i, action2_i = self.actor(states[i], (torch.ones(batch_idx.shape).to(self.device) * i).view(-1, 1), pos_states[i])  # [b,1,n_actions]
            action1_i = action1_i.unsqueeze(1)
            tmp1[:, i*self.n_actions[0]:(i+1)*self.n_actions[0], :] = action1_i
        else:
            action2_i = self.actor(states[i], (torch.ones(batch_idx.shape).to(self.device) * i).view(-1, 1), pos_states[i])  # [b,1,
        tmp1 = tmp1.view(tmp1.size(0), -1)
        action2_i = action2_i.unsqueeze(1)
        tmp2 = actions2_onehot.clone()
        tmp2[:, i*self.n_actions[1]:(i+1)*self.n_actions[1], :] = action2_i
        tmp2 = tmp2.view(tmp2.size(0), -1)
        Q_expected = self.critic(states_critic, tmp1, tmp2, pos_states_critic)
        actor_loss = - Q_expected.mean()
        
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        self.soft_update(self.critic_tar, self.critic, tau=0.005)
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()