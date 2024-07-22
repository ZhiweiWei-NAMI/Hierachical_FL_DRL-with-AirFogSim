# 和PPO离散模型基本一致
from email import policy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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

class ConvNet(nn.Module):
    def __init__(self, n_channel = 3, hidden_channel = 6):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channel, hidden_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_channel, int(hidden_channel*2), kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv2d(int(hidden_channel*2), int(hidden_channel*2), kernel_size=7, stride=2, padding=1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6348, 1024)  
        self.reset_parameters()
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        # x = self.dropout2(x)
        return x


class ValueNet(nn.Module):
    def __init__(self, isBS):
        super(ValueNet, self).__init__()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(1024+6+12+1, 512)  # size is doubled due to concatenation of attention output
        if isBS:
            self.fc2 = nn.Linear(512, 3)
        else:
            self.fc2 = nn.Linear(512, 7)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, idx, pos_states, weights):
        x1 = self.conv(x)
        x2 = torch.cat((x1, idx, pos_states, weights), dim=1) 
        # x2 = torch.cat((x1, pos_states, weights), dim=1) 
        x3 = torch.tanh(self.fc1(x2))
        x4 = self.fc2(x3)
        return x4


class PolicyNet(nn.Module):
    def __init__(self, isBS):
        super(PolicyNet, self).__init__()
        self.conv = ConvNet()
        self.fc1 = nn.Linear(1024+1+2+1, 512)  # size is doubled due to concatenation of attention output
        self.fc2 = nn.Linear(512, 3 if isBS else 7)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x, idx, pos_states, all_weights):
        x1 = self.conv(x)
        selfidx = (idx[0,0]).to(torch.int64)
        x2 = torch.cat((x1, idx, (pos_states[:,selfidx*2:selfidx*2+2]).view(-1,2), (all_weights[:,selfidx]).view(-1,1)), dim=1)
        # x2 = torch.cat((x1, pos_states, all_weights), dim=1)  
        x3 = torch.tanh(self.fc1(x2))
        out1 = F.softmax(self.fc2(x3), dim=1)
        return out1


# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

class CNN_PPO:
    def __init__(self, input_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, n_agent,
                 lmbda, eps, gamma, device):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.max_grad_norm = 4  # 梯度裁剪的阈值
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = lmbda  # 优势函数的缩放因子
        self.eps = eps  # ppo截断范围缩放因子
        self.gamma = gamma  # 折扣因子
        self.device = device
        self.n_agent = n_agent
        self.n_actions = n_actions
        # 网络实例化
        self.actors = [PolicyNet(isBS = i>=4).to(device) for i in range(6)]  # 策略网络
        self.critics = [ValueNet(isBS = i>=4).to(device) for i in range(6)]  # 价值网络
        # 优化器
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_lr) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_lr) for critic in self.critics]
        self.train_num = 0  # 训练次数
        self.epislon = 1
        self.log_alphas = [torch.tensor(-np.e * 0.01 * np.log(7 if i < 4 else 3), dtype=torch.float).to(device).requires_grad_() for i in range(self.n_agent)]
        self.log_alpha_optimizers = [torch.optim.Adam([log_alpha], lr=critic_lr) for log_alpha in self.log_alphas]
        self.target_entropy = [0.83 * np.log(7 if i < 4 else 3) for i in range(self.n_agent)]
    def save_models(self, model_dir):
        for idx, actor in enumerate(self.actors):
            torch.save(actor, model_dir+f'/actor_{idx}')
        for idx, critic in enumerate(self.critics):
            torch.save(critic, model_dir+f'/critic_{idx}')
    def load_models(self, model_dir):
        for idx, _ in enumerate(self.actors):
            self.actors[idx].load_state_dict(torch.load(model_dir + f'/actor_{idx}', map_location=self.device).state_dict())
        for idx, _ in enumerate(self.critics):
            self.critics[idx].load_state_dict(torch.load(model_dir + f'/critic_{idx}', map_location=self.device).state_dict())
            # self.critics[idx] = torch.load(model_dir + f'/critic_{idx}', map_location=self.device)
    # 动作选择
    def take_action(self, state, agent_idx, pos_states, all_weights):  # 1xn_channelxHxW
        cur_agent_idx =  agent_idx # 0 if agent_idx < 4 else 4 
        # cur_agent_idx =  0 if agent_idx < 4 else 4  # agent_idx 
        now_actor = self.actors[cur_agent_idx]
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        selfidx = torch.tensor([agent_idx]).to(self.device).view(1,-1)
        pos_states = torch.tensor(pos_states, dtype=torch.float).to(self.device).view(1,-1)
        all_weights = torch.tensor(all_weights, dtype=torch.float).to(self.device).view(1,-1)
        probs = now_actor(state, selfidx, pos_states, all_weights)  # 当前状态的动作概率 [b,n_actions]
        probs = probs + torch.randn_like(probs)*self.epislon
        probs = torch.clamp(probs, 0, 1) + 1e-5 
        action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
        action = action_dist.sample()  # 从概率分布中随机取样 int
        log_probs = action_dist.log_prob(action)
        action = action.item()
        if action < 4:
            action1 = action + 1
            action2 = 0 # 不变
        else:
            action1 = 0
            action2 = action - 4 # {0,1,2}
        return action, action1, action2, log_probs.detach().cpu().numpy(), action_dist.entropy().detach().cpu().numpy()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    # 训练
    def update(self, transition_dict, batch_idx, transition_dict_all, agent_idx, traverse = 0):
        self.train_num += 1
        # cur_agent_idx =  0 if agent_idx < 4 else 4  # agent_idx 
        cur_agent_idx = agent_idx
        now_critic = self.critics[cur_agent_idx]
        now_actor = self.actors[cur_agent_idx]
        if self.epislon > 0.1 and agent_idx == 0:
            self.epislon *= 0.95
        # 取出数据集
        states = torch.tensor(np.array(transition_dict['states'])[batch_idx], dtype=torch.float).to(self.device)  # [b,n_states]
        actions1 = torch.tensor(np.array(transition_dict['actions1'])[batch_idx]).view(-1,1).type(torch.int64).to(self.device)  # [b,1]
        pos_states = torch.tensor(np.array(transition_dict['pos_states'])[batch_idx]).view(-1,12).float().to(self.device)  # [b,1]
        new_pos_states = torch.tensor(np.array(transition_dict['new_pos_states'])[batch_idx]).view(-1,12).float().to(self.device)  # [b,1]
        weights = torch.tensor(np.array(transition_dict['weights'])[batch_idx]).view(-1,6).float().to(self.device)  # [b,1]
        new_weights = torch.tensor(np.array(transition_dict['new_weights'])[batch_idx]).view(-1,6).float().to(self.device)  # [b,1]
        next_states = torch.tensor(np.array(transition_dict['next_states'])[batch_idx], dtype=torch.float).to(self.device)  # [b,n_states]
        if traverse == 1:
            states = torch.flip(states, [1])
            next_states = torch.flip(next_states, [1])
            x_indices = [0, 2, 4, 6, 8, 10]
            for idx in x_indices:
                pos_states[:, idx] = 1 - pos_states[:, idx]
                new_pos_states[:, idx] = 1 - new_pos_states[:, idx]
            if agent_idx < 4:
                mask_0 = actions1 == 1
                mask_180 = actions1 == 3
                actions1[mask_0] = 3
                actions1[mask_180] = 1
        elif traverse == 2:
            states = torch.flip(states, [2])
            next_states = torch.flip(next_states, [2])
            x_indices = [1, 3, 5, 7, 9, 11]
            for idx in x_indices:
                pos_states[:, idx] = 1 - pos_states[:, idx]
                new_pos_states[:, idx] = 1 - new_pos_states[:, idx]
            if agent_idx < 4:
                mask_90 = actions1 == 0
                mask_270 = actions1 == 2
                actions1[mask_90] = 2
                actions1[mask_270] = 0
        selfidx = (torch.ones(batch_idx.shape).to(self.device) * agent_idx).view(-1, 1)
        dones = torch.tensor(np.array(transition_dict['dones'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        rewards = torch.tensor(np.array(transition_dict['rewards'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        old_log_probs = torch.tensor(np.array(transition_dict['log_probs'])[batch_idx], dtype=torch.float).view(-1,1).to(self.device)  # [b,1]
        # rewards = (rewards - rewards.mean()) / rewards.std()
        # 价值网络
        # 每一轮更新一次策略网络预测的状态
        next_probs = now_actor(next_states, selfidx, new_pos_states, new_weights).detach()  # 当前状态的动作概率 [b,n_actions]
        entropy = -torch.sum(next_probs * torch.log(next_probs+1e-8), dim=-1)
        next_state_value = torch.sum(now_critic(next_states, selfidx, new_pos_states, new_weights) * next_probs, dim=-1) #+ self.log_alphas[cur_agent_idx].exp() * entropy # 下一时刻的state_value  [b,1]
        # next_state_value += torch.randn_like(next_state_value)
        td_target = rewards + self.gamma * next_state_value * (1-dones)  # 目标--当前时刻的state_value  [b,1]
        td_value = now_critic(states, selfidx, pos_states, weights).gather(1, actions1).view(-1, 1)  # 预测--当前时刻的state_value  [b,1]
        td_delta = td_target - td_value  # 时序差分  # [b,1]
        probs = now_actor(states, selfidx, pos_states, weights)  # 当前状态的动作概率 [b,n_actions]
        # cur_entropy = -torch.sum(probs * torch.log(probs+1e-5), dim=-1)
        cur_entropy = - torch.log(probs+1e-5).gather(1, actions1).view(-1, 1)
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
        # advantage = (advantage - advantage.mean()) / (advantage.std()+1e-5)
        # 每一轮更新一次策略网络预测的状态
        tot_probs = probs.gather(1,actions1)
        log_probs = torch.log(tot_probs + 1e-5)
        # 新旧策略之间的比例
        ratio = torch.exp(log_probs - old_log_probs)
        # 近端策略优化裁剪目标函数公式的左侧项
        surr1 = ratio * advantage
        # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
        surr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
        # entropy = -torch.sum(tot_probs * log_probs, dim=1)
        # 策略网络的损失函数
        actor_loss = torch.mean(-torch.min(surr1, surr2))  # - self.log_alphas[cur_agent_idx].exp().detach() * cur_entropy 0.05 * cur_entropy
        # - 0 * entropy.mean()
        # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
        critic_loss = torch.mean(F.mse_loss(now_critic(states, selfidx, pos_states, weights).gather(1,actions1).view(-1,1), td_target.detach()))
        # 更新log_alpha
        alpha_loss = torch.mean((cur_entropy-self.target_entropy[cur_agent_idx]).detach() * self.log_alphas[cur_agent_idx])
        self.log_alpha_optimizers[cur_agent_idx].zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizers[cur_agent_idx].step()
        # 梯度清0
        # 反向传播
        # 梯度更新
        self.actor_optimizers[cur_agent_idx].zero_grad()
        actor_loss.backward()
        # 剪裁梯度
        nn.utils.clip_grad_norm_(now_actor.parameters(), self.max_grad_norm)
        self.actor_optimizers[cur_agent_idx].step()
        self.critic_optimizers[cur_agent_idx].zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(now_critic.parameters(), self.max_grad_norm)
        self.critic_optimizers[cur_agent_idx].step()
        return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy(), self.log_alphas[cur_agent_idx].detach().cpu().numpy()