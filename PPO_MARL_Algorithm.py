from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
from collections import deque
from this import d
from uvfogsim.algorithms.Base_Algorithm_Module import Base_Algorithm_Module
from model_code.MyAutoEncoder import AutoEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from uvfogsim.vehicle import Vehicle
from uvfogsim.uav import UAV
from uvfogsim.bs import BS
import pickle
from util_code import extract_RSU_state, project_uav_position_to_map, project_veh_position_to_map
class PPO_Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, device, useMA=True):
        super(PPO_Actor, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.device = device
        self.useMA = useMA
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(device)
        
        # 注意力机制模块
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4).to(device)
        
        # 动作解码器
        self.action_decoder = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        ).to(device)

        self.actor_optimizer = torch.optim.Adam(self.parameters(), lr=self.actor_lr, eps=1e-5)
        self.epsilon = 0.9
        
    def forward(self, state, other_agents_embeddings):
        """
        state: 当前代理的状态，维度为 (batch_size, state_dim)
        other_agents_embeddings: 区域内其他代理的embedding，维度为 (batch_size, seq_len, hidden_dim)
        """
        # 编码当前状态
        state_encoded = self.state_encoder(state)  # 维度: (batch_size, hidden_dim)
        key_padding_mask = torch.all(other_agents_embeddings == 0, dim=-1)  # 维度: (batch_size, seq_len)
        if self.useMA:
            # 调整 state_encoded 维度以符合 MultiheadAttention 输入要求
            state_encoded = state_encoded.unsqueeze(0)  # 维度: (1, batch_size, hidden_dim)
            key_padding_mask = torch.all(other_agents_embeddings == 0, dim=-1)  # 维度: (batch_size, seq_len)
            other_agents_embeddings = other_agents_embeddings.permute(1, 0, 2)  # 维度: (seq_len, batch_size, hidden_dim)
            
            # 应用注意力机制
            attention_output, _ = self.attention(state_encoded, other_agents_embeddings, other_agents_embeddings, key_padding_mask=key_padding_mask)
            # 调整输出维度以适配解码器
            attention_output = attention_output.squeeze(0)  # 维度: (batch_size, hidden_dim)
            # 把动作的nan替换为0
            attention_output = torch.where(torch.isnan(attention_output), torch.zeros_like(attention_output), attention_output)
            state_encoded = state_encoded.squeeze(0)
        else:
            attention_output = state_encoded
        input_attention_with_state = torch.cat([attention_output, state_encoded], dim=-1)  # 维度: (batch_size, hidden_dim + state_dim)
        # 解码动作
        action_probs = self.action_decoder(input_attention_with_state)  # 维度: (batch_size, action_dim)
        action_probs = F.softmax(action_probs, dim=-1)
        return action_probs
    
    def take_action(self, state, other_agents_embeddings):
        """
        采取动作
        """
        # 先把state和other_agents_embeddings转为tensor
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        other_agents_embeddings = np.array(other_agents_embeddings)
        if other_agents_embeddings.shape[0] == 0:
            other_agents_embeddings = np.zeros((1, self.hidden_dim))
        if other_agents_embeddings.shape[0] < 15:
            pad = np.zeros((15 - other_agents_embeddings.shape[0], self.hidden_dim))
            other_agents_embeddings = np.concatenate([other_agents_embeddings, pad], axis=0)
        else:
            other_agents_embeddings = other_agents_embeddings[:15]
        other_agents_embeddings = torch.tensor(other_agents_embeddings, dtype=torch.float32, device=self.device)
        # 1. unsqueeze
        state = state.unsqueeze(0)
        other_agents_embeddings = other_agents_embeddings.unsqueeze(0)
        output = self.forward(state, other_agents_embeddings)
        output_value = output.cpu().detach().numpy().squeeze(0)
        # epsilon greedy
        # if np.random.rand() < self.epsilon:
        #     output_action = np.random.choice(self.action_dim)
        # else:
        # 重采样
        output_action = np.random.choice(self.action_dim, p=output_value)
        # output_action = np.argmax(output_value)
        return output_action
    
    def save_agent(self, path, rsu_id):
        '''按照RSU区域来存储模型'''
        torch.save(self.state_encoder.state_dict(), os.path.join(path, f'_state_encoder_{rsu_id}.pth'))
        torch.save(self.attention.state_dict(), os.path.join(path, f'_attention_{rsu_id}.pth'))
        torch.save(self.action_decoder.state_dict(), os.path.join(path, f'_action_decoder_{rsu_id}.pth'))

    def load_agent(self, path, rsu_id):
        '''按照RSU区域来加载模型'''
        self.state_encoder.load_state_dict(torch.load(os.path.join(path, f'_state_encoder_{rsu_id}.pth')))
        self.attention.load_state_dict(torch.load(os.path.join(path, f'_attention_{rsu_id}.pth')))
        self.action_decoder.load_state_dict(torch.load(os.path.join(path, f'_action_decoder_{rsu_id}.pth')))

    def opt_zero_grad(self):
        self.state_encoder.zero_grad()
        self.attention.zero_grad()
        self.action_decoder.zero_grad()

    def opt_step(self):
        self.actor_optimizer.step()
        self.epsilon = max(0.05, self.epsilon * 0.95)


class PPO_Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, critic_lr, device, useMA=True):
        super(PPO_Critic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.critic_lr = critic_lr
        self.device = device
        self.useMA = useMA
        
        # 状态编码器
        # self.state_encoder = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh()
        # ).to(device)
        
        # 注意力机制模块
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4).to(device)
        
        # 价值解码器
        self.value_decoder = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

        # self.critic_optimizer = torch.optim.SGD(self.parameters(), lr=self.critic_lr)
        self.critic_optimizer = torch.optim.Adam(self.parameters(), lr=self.critic_lr, eps=1e-5)
        
    def forward(self, state_encoded, other_agents_embeddings):
        """
        state: 当前代理的状态，维度为 (batch_size, state_dim)
        other_agents_embeddings: 区域内其他代理的embedding，维度为 (batch_size, seq_len, hidden_dim)
        """
        # 编码当前状态
        # state_encoded = self.state_encoder(state)  # 维度: (batch_size, hidden_dim)
        key_padding_mask = torch.all(other_agents_embeddings == 0, dim=-1)  # 维度: (batch_size, seq_len)

        if self.useMA:
            # 调整 state_encoded 维度以符合 MultiheadAttention 输入要求 
            state_encoded = state_encoded.unsqueeze(0)  # 维度: (1, batch_size, hidden_dim)
            # 计算mask，other_agents_embeddings中的0值不参与计算
            other_agents_embeddings = other_agents_embeddings.permute(1, 0, 2)  # 维度: (seq_len, batch_size, hidden_dim)
            # other_agents_embeddings = torch.cat([other_agents_embeddings, state_encoded], dim=0)
            # # key_padding_mask 最后一项是state_encoded，添加到最后
            # key_padding_mask = torch.all(other_agents_embeddings == 0, dim=-1)  # 维度: (seq_len, batch_size)
            # # 最后一个是True
            # # key_padding_mask[-1] = True
            # key_padding_mask = key_padding_mask.permute(1, 0)  # 维度: (batch_size, seq_len)
            
            # 应用注意力机制
            attention_output, _ = self.attention(state_encoded, other_agents_embeddings, other_agents_embeddings, key_padding_mask=key_padding_mask)
            # 调整输出维度以适配解码器
            attention_output = attention_output.squeeze(0)  # 维度: (batch_size, hidden_dim)
            # 把nan替换为0
            attention_output = torch.where(torch.isnan(attention_output), torch.zeros_like(attention_output), attention_output)
            state_encoded = state_encoded.squeeze(0)
        else:
            attention_output = state_encoded
        # 解码价值
        input_attention_with_state = torch.cat([attention_output, state_encoded], dim=-1)  # 维度: (batch_size, hidden_dim + state_dim)
        value = self.value_decoder(input_attention_with_state)  # 维度: (batch_size, 1)
        return value
    
    def save_agent(self, path, rsu_id):
        '''按照RSU区域来存储模型'''
        # torch.save(self.state_encoder.state_dict(), os.path.join(path, f'_state_encoder_{rsu_id}.pth'))
        torch.save(self.attention.state_dict(), os.path.join(path, f'_attention_{rsu_id}.pth'))
        torch.save(self.value_decoder.state_dict(), os.path.join(path, f'_value_decoder_{rsu_id}.pth'))

    def load_agent(self, path, rsu_id):
        '''按照RSU区域来加载模型'''
        # self.state_encoder.load_state_dict(torch.load(os.path.join(path, f'_state_encoder_{rsu_id}.pth')))
        self.attention.load_state_dict(torch.load(os.path.join(path, f'_attention_{rsu_id}.pth')))
        self.value_decoder.load_state_dict(torch.load(os.path.join(path, f'_value_decoder_{rsu_id}.pth')))

    def opt_zero_grad(self):
        # self.state_encoder.zero_grad()
        self.attention.zero_grad()
        self.value_decoder.zero_grad()

    def opt_step(self):
        self.critic_optimizer.step()

class ReplayBuffer():
    '''在线学习的replaybuffer，按照RSU的区域存储经验，统一用dict来存储经验列表；包含obs, action, other_agents_obs, reward, done, next_obs, next_other_agents_obs'''
    def __init__(self, episode_size, episode_num):
        '''episode_size: 每个episode的大小；episode_num: 总共的episode数量；memory_size: 总共的经验存储上限。由于每个区域会有不同数量的agent，因此要区分开来'''
        self.episode_size = episode_size
        self.episode_num = episode_num
        self.obs = {}
        self.action = {}
        self.other_agents_obs = {}
        self.reward = {}
        self.done = {}
        self.exp_agent_id = {}
        self.exp_episode_id = {}
        self.recent_rewards = {}
        self.cur_max_episode_num = 0
        self.initialize_dict()
        
    def initialize_dict(self):
        # 按照rsu_id，也就是0~11，在每个字典中初始化一个空的deque，maxlen=episode_size
        for rsu_id in range(12):
            self.obs[rsu_id] = deque(maxlen=self.episode_size)
            self.action[rsu_id] = deque(maxlen=self.episode_size)
            self.other_agents_obs[rsu_id] = deque(maxlen=self.episode_size)
            self.reward[rsu_id] = deque(maxlen=self.episode_size)
            self.done[rsu_id] = deque(maxlen=self.episode_size)
            self.exp_agent_id[rsu_id] = deque(maxlen=self.episode_size)
            self.exp_episode_id[rsu_id] = deque(maxlen=self.episode_size)
            self.recent_rewards[rsu_id] = deque(maxlen=100)
            
    def clear_all_memory(self):
        '''清空所有的memory'''
        self.obs = {}
        self.action = {}
        self.other_agents_obs = {}
        self.reward = {}
        self.done = {}
        self.exp_agent_id = {}
        self.exp_episode_id = {}
        self.recent_rewards = {}
        self.cur_max_episode_num = 0
        self.initialize_dict()

    def avg_rewards(self, rsu_id):
        if rsu_id in self.reward:
            return np.sum(self.reward[rsu_id])
        else:
            return 0
    
    def clear_episode_memory(self):
        '''[已失效]清空之前episode的memory，仅保留最近n个记忆'''
        self.clear_all_memory()
        return 
        del_epi_id = self.cur_max_episode_num - self.episode_num - 1
        # 删除所有小于del_epi_id的episode
        for key in self.obs.keys():
            while len(self.exp_episode_id[key]) > 0 and self.exp_episode_id[key][0] < del_epi_id:
                self.obs[key].pop(0)
                self.action[key].pop(0)
                self.other_agents_obs[key].pop(0)
                self.reward[key].pop(0)
                self.done[key].pop(0)
                self.exp_agent_id[key].pop(0)
                self.exp_episode_id[key].pop(0)
                
        
    def add_experience(self, rsu_id, agent_id, epi_id, obs, action, other_agents_obs, reward, done):
        '''添加的每一条经验都是以rsu区域划分的，也就是说，都是按照区域内的task_V数量差异。因此，要加一个agent_id来标识'''
        # other_agents_obs: [n_agents, state_dim]。固定长度为5，多了就截断，少了就补0
        other_agents_obs = np.array(other_agents_obs)
        if other_agents_obs.shape[0] == 0:
            other_agents_obs = np.zeros((1, 128))
        if other_agents_obs.shape[0] < 15:
            pad = np.zeros((15 - other_agents_obs.shape[0], other_agents_obs.shape[1]))
            other_agents_obs = np.concatenate([other_agents_obs, pad], axis=0)
        else:
            other_agents_obs = other_agents_obs[:15]

        self.cur_max_episode_num = max(self.cur_max_episode_num, epi_id)
        if rsu_id in self.obs:
            self.obs[rsu_id].append(obs)
            self.action[rsu_id].append(action)
            self.other_agents_obs[rsu_id].append(other_agents_obs)
            self.reward[rsu_id].append(reward)
            self.done[rsu_id].append(done)
            self.exp_agent_id[rsu_id].append(agent_id)
            self.exp_episode_id[rsu_id].append(epi_id)
        else:
            self.obs[rsu_id] = deque([obs], maxlen=self.episode_size)
            self.action[rsu_id] = deque([action], maxlen=self.episode_size)
            self.other_agents_obs[rsu_id] = deque([other_agents_obs], maxlen=self.episode_size)
            self.reward[rsu_id] = deque([reward], maxlen=self.episode_size)
            self.done[rsu_id] = deque([done], maxlen=self.episode_size)
            self.exp_agent_id[rsu_id] = deque([agent_id], maxlen=self.episode_size)
            self.exp_episode_id[rsu_id] = deque([epi_id], maxlen=self.episode_size)
    

    
    def sample(self, rsu_id, n_episode=1):
        '''采样所有的经验, 是按照episode_id->agent_id排序的'''
        obs = np.array(self.obs[rsu_id])
        action = np.array(self.action[rsu_id])
        other_agents_obs = np.array(self.other_agents_obs[rsu_id])
        reward = np.array(self.reward[rsu_id])
        done = np.array(self.done[rsu_id])
        agent_id = np.array(self.exp_agent_id[rsu_id])
        episode_id = np.array(self.exp_episode_id[rsu_id])
        # 先按照episode_id排序，然后按照agent_id排序。以上排序对所有的数据都进行
        sort_indices = np.lexsort((agent_id, episode_id))
        obs = obs[sort_indices]
        action = action[sort_indices]
        other_agents_obs = other_agents_obs[sort_indices]
        reward = reward[sort_indices]
        done = done[sort_indices]
        agent_id = agent_id[sort_indices]
        episode_id = episode_id[sort_indices]
        return obs, action, other_agents_obs, reward, done, agent_id, episode_id

class FL_PPO_Agent(Base_Algorithm_Module):
    '''根据RSU区域管理PPO agent。由于区域内使用的是同一个MARL模型，每个区域内直接以batch去训练,训练完记得clear memory；区域间共享模型，使用的是FedAvg的思路，所以是第二个过程，用exeFedAvg()来实现\n
    此外，内部管理的map都是按照网格划分的
    '''
    def __init__(self, env, args):
        super().__init__()
        self.args = args
        self.env = env
        self.update_cnt = 0
    
        self.n_hiddens = args.n_hidden
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.gamma = args.gamma
        self.lmbda = args.lmbda
        self.eps = args.eps
        self.device = args.device
        self.device = torch.device(self.device) if torch.cuda.is_available() else torch.device('cpu')
        self.n_RSU = env.n_RSU # 几个RSU就有几个区域的MAPPO模型
        self.state_dim = 86 # 自身的task_lambda属性，位置（xyz），速度，当前任务属性【datasize, cpu, ddl】(8个)，2）最近10个fog车辆+RSU的属性（speed, cpu, position, 当前任务等待时延, SINR）(7*6个)
        self.actions_dim = self.args.v_neighbor_Veh + 2 # 12,最近的10个fog节点+1个RSU+不卸载
        self.ppo_agents = [PPO_Actor(self.state_dim, self.n_hiddens, self.actions_dim, self.actor_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]
        self.ppo_critics = [PPO_Critic(self.state_dim, self.n_hiddens, self.critic_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]
        # 让每一个agent和critic网络初始化成一样的参数
        for i in range(1, self.n_RSU):
            self.ppo_agents[i].load_state_dict(self.ppo_agents[0].state_dict())
            self.ppo_critics[i].load_state_dict(self.ppo_critics[0].state_dict())
        self.replay_buffer = ReplayBuffer(args.max_steps*5, 1) # 仅保留最近的5个episode的经验
        self.grid_width = args.grid_width # 正方形网格，height就不需要了
        self.grid_num_x = int((env.max_range_x - env.min_range_x) // self.grid_width)
        self.grid_num_y = int((env.max_range_y - env.min_range_y) // self.grid_width)
        self.auto_encoder = AutoEncoder().to(self.device)
        # 加载
        # with torch.cuda.device(4): 
        self.auto_encoder.load_state_dict(torch.load('./models/autoencoder_99.pth', map_location='cpu'))

        # 以下属性为在线属性，随着环境变化而变化
        self.veh_pos_map_traces = deque(maxlen=args.max_steps)
        self.uav_pos_map_traces = deque(maxlen=args.max_steps)
        self.rsu_positions = None
        self.rsu_coverage_map = None
        self.length_map_grid = None
        self.max_speed_map_grid = None
        self.no_fly_zone = None
        self.task_num_of_rsu = [deque(maxlen=args.max_steps) for _ in range(self.n_RSU)]
        self.task_qos_of_rsu = [deque(maxlen=args.max_steps) for _ in range(self.n_RSU)]
        self.RSU_labels = range(self.n_RSU)
        
        with open('./dataset/max_speed_map_grid.pkl', 'rb') as f:
            self.max_speed_map_grid = pickle.load(f)
        # ./images/length_map_grid.pkl load map列表
        with open('./dataset/length_map_grid.pkl', 'rb') as f:
            self.length_map_grid = pickle.load(f)
        
        from torch.utils.tensorboard import SummaryWriter
        import os
        import shutil
        writer_path = f"{args.tensorboard_writer_file}"
        if os.path.exists(writer_path):
            shutil.rmtree(writer_path)
        self.writer = SummaryWriter(writer_path)
        self.save_best = False
        if self.args.load_model:
            self.load_agents()

    def load_agents(self):
        saved_path = self.args.saved_path
        for agent_id, agent in enumerate(self.ppo_agents):
            agent.load_agent(saved_path, agent_id)
        for agent_id, agent in enumerate(self.ppo_critics):
            agent.load_agent(saved_path, agent_id)
        print('Agents loaded successfully!')

    def reset_RSU_voronoi_and_no_fly(self, result_id):
        '''更新rsu_positions, rsu_coverage_map, no_fly_zone'''
        # 根据result_id，重置rsu的部署覆盖范围和禁飞区域
        rsu_positions_results = [[(112, 88), (141, 58), (244, 80), (369, 99), (124, 187), (189, 164), (297, 199), (320, 136), (74, 271), (207, 258), (254, 259), (334, 221)], [(131, 84), (187, 103), (281, 110), (369, 91), (107, 135), (211, 152), (263, 136), (378, 191), (128, 226), (179, 248), (256, 234), (322, 228)], [(112, 108), (204, 92), (306, 97), (391, 87), (57, 204), (177, 133), (261, 185), (338, 172), (121, 218), (186, 244), (263, 267), (364, 253)], [(89, 80), (158, 74), (247, 101), (355, 67), (78, 161), (175, 167), (269, 148), (341, 147), (100, 222), (202, 255), (276, 217), (344, 251)], [(108, 94), (213, 82), (299, 99), (357, 97), (95, 209), (175, 134), (270, 175), (379, 197), (63, 276), (177, 230), (309, 234), (374, 228)]]
        no_fly_zone_results = [[(934, 809, 263.3870071425481), (1981, 1283, 315.82764458833543), (1997, 527, 117.4017041877551)], [(1746, 850, 159.46079502521124), (424, 1099, 209.13879180961072), (1100, 349, 237.02084218709663)], [(1451, 1005, 345.25277160683123), (673, 947, 230.31241074672877), (378, 1327, 121.44902277887451)], [(1824, 1006, 268.0090998233277), (1316, 793, 200.63560396801844), (541, 1393, 225.55923380031683)], [(1029, 680, 299.0083464063226), (861, 1158, 168.17489462537907), (1405, 1222, 134.47772905244267)]]
        for i in range(len(no_fly_zone_results)):
            no_fly_zone_results[i] = [(x/5, y/5, r/5) for x, y, r in no_fly_zone_results[i]]
        radius = 60
        coverage_map = np.ones((400, 400)) * -1
        rsu_positions = rsu_positions_results[result_id]
        self.rsu_positions = rsu_positions
        # 对于每一个格点,判断最近的rsu,然后判断举例是否小于radius.如果小于radius,则认为是这个rsu的覆盖范围,coverage_map[y][x] = rsu_id
        for y in range(50, 300):
            for x in range(50, 400):
                min_dist = 100000
                rsu_id = -1
                for i, (rsu_x, rsu_y) in enumerate(rsu_positions):
                    dist = np.sqrt((rsu_x - x) ** 2 + (rsu_y - y) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        rsu_id = i
                if min_dist < radius:
                    coverage_map[y][x] = rsu_id
        self.rsu_coverage_map = coverage_map # 按照网格划分的
        self.no_fly_zone = no_fly_zone_results[result_id]
        self.replay_buffer.clear_all_memory()
        self.resetAllAgents()
            
    def resetAllAgents(self):
        self.ppo_agents = [PPO_Actor(self.state_dim, self.n_hiddens, self.actions_dim, self.actor_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]
        self.ppo_critics = [PPO_Critic(self.state_dim, self.n_hiddens, self.critic_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]
        # 让每一个agent和critic网络初始化成一样的参数
        for i in range(1, self.n_RSU):
            self.ppo_agents[i].load_state_dict(self.ppo_agents[0].state_dict())
            self.ppo_critics[i].load_state_dict(self.ppo_critics[0].state_dict())
    def calculate_reward(self):
        '''判断本轮的task完成多少了，然后计算reward，在每一个task的g_veh上添加reward'''
        # 1. 先把所有task_V和UAV的reward清零
        for veh in self.env.vehicle_by_index:
            veh.reward = 0
        for uav in self.env.UAVs:
            uav.reward = 0
        for rsu_id in range(self.n_RSU):
            self.task_num_of_rsu[rsu_id].append(0)
            self.task_qos_of_rsu[rsu_id].append(0)
        # 2. 遍历self.offloading_tasks
        all_task_num = max(1, len(self.offloading_tasks))
        all_task_qos = 0
        failed_task_cnts = [0 for _ in range(self.n_RSU)]
        offloaded_task_cnts = [0 for _ in range(self.n_RSU)]
        self.total_latencies = [0 for _ in range(self.n_RSU)]
        for task in self.offloading_tasks:
            gen_V = task.g_veh
            rsu_id = self.env.BSs.index(gen_V.nearest_BS)
            self.task_num_of_rsu[rsu_id][-1] += 1
            if task.is_finished:
                task_u = task.get_task_utility()
                gen_V.reward += task_u
                self.task_qos_of_rsu[rsu_id][-1] += task_u
                all_task_qos += task_u
                offloaded_task_cnts[rsu_id] += 1
                self.total_latencies[rsu_id] += task.service_delay
            else:
                self.total_latencies[rsu_id] += task.ddl
                if task.is_offloaded:
                    failed_task_cnts[rsu_id] += 1
                    gen_V.reward -= task.ini_cpu
                    self.task_qos_of_rsu[rsu_id][-1] -= task.ini_cpu
        for rsu_id in range(self.n_RSU):
            self.task_num_of_rsu[rsu_id][-1] /= all_task_num
            self.task_qos_of_rsu[rsu_id][-1] /= max(1e-5, self.task_num_of_rsu[rsu_id][-1])

        self.all_task_num = all_task_num
        self.failed_task_cnts = failed_task_cnts
        self.offloaded_task_cnts = offloaded_task_cnts

    def update_agents(self, cnt):
        '''先更新每个区域的MAPPO，然后再进行FL'''
        # 1. 更新MAPPO
        # if self.replay_buffer.cur_max_episode_num < self.replay_buffer.episode_num:
        #     # 如果episode数量不够，不进行训练
        #     return
        if cnt % 40 != 0 or cnt == 0:
            return
        if self.args.save_model and self.update_cnt > 1 and self.update_cnt % self.args.fre_to_save == 0:
            self.save_agents()
        self.update_cnt += 1
        for rsu_id in range(self.n_RSU):
            obs, action, other_agents_obs, reward, done, agent_id, episode_id = self.replay_buffer.sample(rsu_id, self.args.retreive_n_episode)
            batch_size = obs.shape[0]
            if batch_size <= 10:
                continue
            # next_obs和next_other_agents_obs是下一个状态的obs和other_agents_obs，所以直接copy 原先的obs和other_agents_obs，然后把每一个状态循环前移
            next_obs = np.roll(obs, -1, axis=0)
            next_other_agents_obs = np.roll(other_agents_obs, -1, axis=0)

            # 除了agent_id和episode_id，其他转为tensor
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            other_agents_obs = torch.tensor(other_agents_obs, dtype=torch.float).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device) * 100
            done = torch.tensor(done, dtype=torch.float).to(self.device) # 需要保证，最后一个step的每个agent保存的经验都是done在最后
            next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
            next_other_agents_obs = torch.tensor(next_other_agents_obs, dtype=torch.float).to(self.device)

            ppo_actor = self.ppo_agents[rsu_id]
            ppo_critic = self.ppo_critics[rsu_id]
            if 'MAPPO_Cen' in self.args.method:
                ppo_actor = self.ppo_agents[0]
                ppo_critic = self.ppo_critics[0]
            
            obs_encoded = ppo_actor.state_encoder(obs).detach()
            next_obs_encoded = ppo_actor.state_encoder(next_obs).detach()
            # 1.1 计算target_value
            next_q_value = ppo_critic(next_obs_encoded, next_other_agents_obs)
            target_value = reward.unsqueeze(-1) + self.gamma * next_q_value * (1 - done.unsqueeze(-1))
            value = ppo_critic(obs_encoded, other_agents_obs)
            advantages = np.zeros(batch_size)  # 初始化优势数组
            last_adv = 0  # 上一步的优势，用于GAE计算

            # 假设我们已经有了rewards, dones, values等数组，其中dones表示状态是否是episode的最后状态
            last_agent_id = -1
            agent_num = -1
            for t in reversed(range(len(reward))):
                if done[t]:
                    last_adv = 0  # 如果是episode结束，重置优势
                if last_agent_id != agent_id[t]: # 如果agent_id不同，重置优势，表示一个agent的经验结束
                    last_adv = 0
                    last_agent_id = agent_id[t]
                    done[t] = 1
                    agent_num += 1
                if t == len(reward) - 1:
                    delta = reward[t] - value[t]
                    last_adv = delta
                else:
                    delta = reward[t] + self.gamma * value[t + 1] * (1 - done[t]) - value[t]  # 计算TD残差
                    last_adv = delta + self.gamma * self.lmbda * last_adv * (1 - done[t])  # 计算优势
                advantages[t] = last_adv
            
            advantage = torch.tensor(advantages, dtype=torch.float).to(self.device).detach()
            old_action_log_probs = torch.log(1e-5+ppo_actor(obs, other_agents_obs).gather(1, action.unsqueeze(-1))).detach()
            train_actor_loss = 0
            train_critic_loss = 0
            entropy = 0
            for _ in range(self.args.train_n_episode):
                action_prob = ppo_actor(obs, other_agents_obs)
                obs_encoded = ppo_actor.state_encoder(obs).detach()
                log_probs = torch.log(action_prob.gather(1, action.unsqueeze(-1))+1e-5)
                tmp_entropy = -(action_prob * torch.log(action_prob+1e-5)).sum(dim=-1).mean()
                entropy += tmp_entropy
                ratio = torch.exp(log_probs - old_action_log_probs)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                actor_loss = torch.mean(-torch.min(surr1, surr2).mean())# / max(1, agent_num)
                critic_loss = torch.mean(F.mse_loss(ppo_critic(obs_encoded, other_agents_obs), target_value.detach())) #/ max(1, agent_num)
                ppo_actor.opt_zero_grad()
                actor_loss.backward()
                ppo_actor.opt_step()
                ppo_critic.opt_zero_grad()
                critic_loss.backward()
                ppo_critic.opt_step()
                train_actor_loss += actor_loss.item()
                train_critic_loss += critic_loss.item()
            if np.isnan(train_actor_loss) or np.isnan(train_critic_loss):
                print('nan')
                return
            self.writer.add_scalar(f'Loss/Actor_{rsu_id}', train_actor_loss/self.args.train_n_episode, self.update_cnt)
            self.writer.add_scalar(f'Loss/Critic_{rsu_id}', train_critic_loss/self.args.train_n_episode, self.update_cnt)
            self.writer.add_scalar(f'Loss/Entropy_{rsu_id}', entropy.item()/self.args.train_n_episode, self.update_cnt)
            
        self.replay_buffer.clear_episode_memory()
        # 2. 更新FL模型，每FL_fre次更新一次.FedAvg，根据agent的数量来加权
        if self.update_cnt % (10*self.args.FL_fre) == 0 and self.args.draw_env_figure: 
            self.draw_env_figure('./images/FL')
        if self.update_cnt % self.args.FL_fre == 0 and ('Fed' in self.args.method):
            # 2.1 计算各个区域的eff，是一个特征向量的形式，[n_RSU, 64]
            agents_eff = None
            if 'MAPPO_AggFed' in self.args.method:  
                agents_eff = self.calculate_eff()
            # 2.2 层次聚类，只有MAPPO_AggFed才需要
            if 'MAPPO_AggFed' in self.args.method:
                cluster_labels = self.cluster_agents(agents_eff, self.update_cnt // self.args.FL_fre)
            else:
                cluster_labels = np.zeros(self.n_RSU) # 直接FedAvg,不分类,但是在权重上下手
            # 2.3 FedAvg分配全局模型
            # if self.args.method != 'MAPPO_nFL':
            self.clusterFedAvg(cluster_labels)

    def clusterFedAvg(self, cluster_labels):
        '''根据聚类结果，分配全局模型到各个代理，可以实现AggFed和FedAvg'''
        if 'MAPPO_AggFed' in self.args.method:
            # 对于AggFed使用np.array(self.aggFed_weights)的倒数,作为agent_num_in_RSUs,防止出现0
            tmp = np.array(self.aggFed_weights) # AggFed是本地化的算法，所以更新本地模型的时候是按照tmp对全局模型进行加权更新的。若tmp[i]=1，则全按照global model，如果tmp[i]=-1，则不更新，如果是0，则各自一半。所以本地权重是.tmp[-1,1]
            tmp_weight = (tmp+1)/2 # 0~1
            # tmp_weight = tmp # -1~1
            # agent_num_in_RSUs = np.array(self.task_qos_of_rsu).mean(axis=1).reshape(-1)
            agent_num_in_RSUs = np.array([rsu.task_v_num for rsu in self.env.BSs])
            # 按照tmp_weight最高的X个，设置为0
            # setzero_indices = np.argsort(tmp_weight)[self.args.max_FL_RSU:]
            setzero_indices = np.random.choice(range(self.n_RSU), self.n_RSU - self.args.max_FL_RSU, replace=False)
            agent_num_in_RSUs[setzero_indices] = 0
        elif 'MAPPO_FedAvg' in self.args.method:
            agent_num_in_RSUs = np.array(self.task_qos_of_rsu).mean(axis=1).reshape(-1) + np.min(self.task_qos_of_rsu)
            # 随机选取X个，其他的都设置为0
            setzero_indices = np.random.choice(range(self.n_RSU), self.n_RSU - self.args.max_FL_RSU, replace=False)
            agent_num_in_RSUs[setzero_indices] = 0
        elif 'Certain_FedAvg' in self.args.method:
            agent_num_in_RSUs = np.array([rsu.task_v_num for rsu in self.env.BSs])
            setzero_indices = np.random.choice(range(self.n_RSU), self.n_RSU-self.args.max_FL_RSU, replace=False)
            agent_num_in_RSUs[setzero_indices] = 0
        weights_of_RSUs = np.ones_like(agent_num_in_RSUs, dtype=np.float32)
        global_actor_networks = [PPO_Actor(self.state_dim, self.n_hiddens, self.actions_dim, self.actor_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]
        global_critic_networks = [PPO_Critic(self.state_dim, self.n_hiddens, self.critic_lr, self.device, useMA=self.args.useMA) for _ in range(self.n_RSU)]

        # 初始化全局模型参数集合
        global_actor_params = [{} for _ in range(self.n_RSU)]
        global_critic_params = [{} for _ in range(self.n_RSU)]

        for i in range(self.n_RSU):
            if i not in set(cluster_labels):
                continue
            cluster_indices = np.where(cluster_labels == i)[0]
            # weights_of_RSUs 根据当前的聚类结果来分配权重，和agent_num成正比
            weights_of_RSUs[cluster_indices] = agent_num_in_RSUs[cluster_indices] / max(1, np.sum(agent_num_in_RSUs[cluster_indices]))
            # 保证总和为1；如果全为0，就平均分配
            if np.sum(weights_of_RSUs[cluster_indices]) == 0:
                weights_of_RSUs[cluster_indices] = np.ones_like(cluster_indices) / len(cluster_indices)
            for j in cluster_indices:
                agent_params = self.ppo_agents[j].state_dict()
                critic_params = self.ppo_critics[j].state_dict()
                
                for name, param in agent_params.items():
                    if name not in global_actor_params[i]:
                        global_actor_params[i][name] = param.clone() * weights_of_RSUs[j]
                    else:
                        global_actor_params[i][name] += param * weights_of_RSUs[j]

                for name, param in critic_params.items():
                    if name not in global_critic_params[i]:
                        global_critic_params[i][name] = param.clone() * weights_of_RSUs[j]
                    else:
                        global_critic_params[i][name] += param * weights_of_RSUs[j]

            # 更新全局模型
            global_actor_networks[i].load_state_dict(global_actor_params[i])
            global_critic_networks[i].load_state_dict(global_critic_params[i])
            
        # 分配全局模型到各代理
        for i in range(self.n_RSU):
            if i not in set(cluster_labels):
                continue
            cluster_indices = np.where(cluster_labels == i)[0]
            for j in cluster_indices:
                if 'MAPPO_AggFed' in self.args.method and '_local' in self.args.method:
                    tmp_actor_params = global_actor_networks[i].state_dict()
                    tmp_critic_params = global_critic_networks[i].state_dict()
                    agent_params = self.ppo_agents[j].state_dict()
                    critic_params = self.ppo_critics[j].state_dict()
                    for name, param in agent_params.items():
                        agent_params[name] = tmp_actor_params[name] * (1-tmp_weight[j]) + param * ( tmp_weight[j])
                    for name, param in critic_params.items():
                        critic_params[name] = tmp_critic_params[name] * (1-tmp_weight[j]) + param * ( tmp_weight[j])
                    self.ppo_agents[j].load_state_dict(agent_params)
                    self.ppo_critics[j].load_state_dict(critic_params)
                else:
                    self.ppo_agents[j].load_state_dict(global_actor_networks[i].state_dict())
                    self.ppo_critics[j].load_state_dict(global_critic_networks[i].state_dict())


    def cluster_agents(self, agents_eff, FL_epoch):
        '''根据agents_eff进行层次聚类，返回聚类结果'''
        clustering = AgglomerativeClustering(affinity='cosine', n_clusters=self.args.n_cluster, linkage='complete')
        labels = clustering.fit_predict(agents_eff)
        self.RSU_labels = labels
        self.writer.add_histogram('Clustering/RSU_labels', labels, FL_epoch)
        # 按照cluster的结果，每个cluster里面把feature_vec平均，然后重新求一遍cosine距离，设置到self.aggFed_weights
        for i in range(np.max(labels)+1):
            cluster_indices = np.where(labels == i)[0]
            mean_feature_vec = np.zeros(66)
            for j in cluster_indices:
                mean_feature_vec += agents_eff[j] #/ self.eigenvalue[j]
            for j in cluster_indices:
                self.aggFed_weights[j] = np.dot(agents_eff[j], mean_feature_vec) / (np.linalg.norm(agents_eff[j]) * np.linalg.norm(mean_feature_vec))
        return labels
    def calculate_eff(self):
        '''基于各个RSU的veh/uav position信息，使用auto-encoder压缩为状态，然后计算每个区域的eff指数，并且返回'''
        step_length = 40 # len(self.veh_pos_map_traces)
        # 记录过去step_length中，每个区域的state
        state_feature_traces = np.zeros((self.n_RSU, step_length, 66)) # 这里的64 是auto-encoder的输出维度,2是任务属性
        step_cnt = 0
        total_task_qos = 0
        for step in range(self.args.max_steps-step_length, self.args.max_steps):
            rsu_state = []
            rsu_task_state = []
            for rsu_id in range(self.n_RSU):
                state = extract_RSU_state(self.veh_pos_map_traces[step], self.uav_pos_map_traces[step], self.no_fly_zone, self.rsu_positions, self.rsu_coverage_map, self.length_map_grid, self.max_speed_map_grid, rsu_id) # [7, 120, 120]
                rsu_state.append(state)
                rsu_task_state.append((self.task_num_of_rsu[rsu_id][step], self.task_qos_of_rsu[rsu_id][step]*self.task_num_of_rsu[rsu_id][step])) # [N, 2]
                total_task_qos += rsu_task_state[-1][1]
            rsu_state = np.array(rsu_state)
            rsu_state = torch.tensor(rsu_state, dtype=torch.float).to(self.device)
            state_feature = self.auto_encoder.encoder(rsu_state).cpu().detach().numpy() # [N, 64]
            state_feature_traces[:, step_cnt, :64] = state_feature / np.linalg.norm(state_feature)
            state_feature_traces[:, step_cnt, 64:] = np.array(rsu_task_state)
            state_feature_traces[:, step_cnt, 65] /= max(total_task_qos, 1e-5)
            # 把每个step_cnt每个RSU的state_feature_traces都到[0,1]之间
            for rsu_id in range(self.n_RSU):
                state_feature_traces[rsu_id, step_cnt, :] = state_feature_traces[rsu_id, step_cnt, :] / np.linalg.norm(state_feature_traces[rsu_id, step_cnt, :])
            step_cnt += 1
            total_task_qos = 0
        # 针对每一个RSU，遍历每个step，估算状态转移矩阵的特征值，存储特征向量
        feature_vec_for_rsus = np.zeros((self.n_RSU, 66))
        feature_value_for_rsus = -1000*np.ones(self.n_RSU)
        for rsu_id in range(self.n_RSU):
            for step in range(step_length-1):
                S_t = state_feature_traces[rsu_id, step]
                S_t1 = state_feature_traces[rsu_id, step+1]
                S_t /= np.linalg.norm(S_t)
                A = np.matmul(S_t.T, S_t1) # [1]
                if A.max() > feature_value_for_rsus[rsu_id]:
                    feature_value_for_rsus[rsu_id] = A.max()
                    feature_vec_for_rsus[rsu_id] = S_t
        # 计算所有特征向量的均值作为一个中间距离,然后计算每个向量到中间向量的余弦距离
        # 中间距离由self.task_qos_of_rsu的加权平均值来计算
        # sum_tmp = np.sum(feature_value_for_rsus)
        # if np.abs(sum_tmp) < 1e-3:
        #     sum_tmp = 1e-3
        weights = feature_value_for_rsus #/ sum_tmp
        # mean_feature_vec = feature_vec_for_rsus.mean(axis=0) # [64]
        mean_feature_vec = np.zeros(66)
        for rsu_id in range(self.n_RSU):
            # 保障weights在-1e-5, 1e-5之外，防止出现nan
            if weights[rsu_id] < 1e-5 and weights[rsu_id] > 0:
                weights[rsu_id] = 1e-5
            elif weights[rsu_id] > -1e-5 and weights[rsu_id] < 0:
                weights[rsu_id] = -1e-5
            mean_feature_vec += feature_vec_for_rsus[rsu_id] #/ weights[rsu_id]
        self.eigenvalue = weights
        cos_dis = np.zeros(self.n_RSU)
        for rsu_id in range(self.n_RSU):
            cos_dis[rsu_id] = np.dot(feature_vec_for_rsus[rsu_id], mean_feature_vec) / (np.linalg.norm(feature_vec_for_rsus[rsu_id]) * np.linalg.norm(mean_feature_vec)) # range [-1, 1]
        self.aggFed_weights = cos_dis
        # feature_vec_for_rsus[-1,:] = mean_feature_vec
        return feature_vec_for_rsus
    
    def save_agents(self, add_str=None, terminated = False):
        saved_path = self.args.saved_path
        if terminated:
            print('Agents terminated! Saving to terminated folder...')
            saved_path = os.path.join(saved_path, 'terminated')
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
        if add_str is not None:
            saved_path = os.path.join(saved_path, add_str)
            if not os.path.exists(saved_path):
                os.makedirs(saved_path)
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
        for agent_id, agent in enumerate(self.ppo_agents):
            agent.save_agent(saved_path, agent_id)
        for agent_id, agent in enumerate(self.ppo_critics):
            agent.save_agent(saved_path, agent_id)
        print(f'Agents saved to {saved_path} successfully!')

    def log_reward(self, step):
        # 通过replay_buffer计算reward，然后记录到tensorboard
        tot_reward = 0
        for rsu_id in range(self.n_RSU):
            avg_reward = self.task_qos_of_rsu[rsu_id][-1] * self.task_num_of_rsu[rsu_id][-1]
            if avg_reward is not None:
                self.writer.add_scalar(f'Reward/RSU_{rsu_id}', avg_reward, step)    
                tot_reward += avg_reward
            self.writer.add_scalar(f'Ratio/Failed_Task_Ratio_{rsu_id}', self.failed_task_cnts[rsu_id] / max(1, self.task_num_of_rsu[rsu_id][-1]*self.all_task_num), step)
            self.writer.add_scalar(f'Ratio/Offloaded_Succ_Ratio_{rsu_id}', self.offloaded_task_cnts[rsu_id] / max(1, self.task_num_of_rsu[rsu_id][-1]*self.all_task_num), step)
            self.writer.add_scalar(f'Num/Task_Num_{rsu_id}', self.task_num_of_rsu[rsu_id][-1]*self.all_task_num, step)
            self.writer.add_scalar(f'Latency/Avg_Latency_{rsu_id}', self.total_latencies[rsu_id] / max(1, self.task_num_of_rsu[rsu_id][-1]*self.all_task_num), step)
        self.writer.add_scalar('Reward/Total', tot_reward, step)
        # 添加total failed ratio, total offloaded ratio, total task num, total avg latency
        self.writer.add_scalar('Ratio/Total_Failed_Ratio', np.sum(self.failed_task_cnts) / max(1, self.all_task_num), step)
        self.writer.add_scalar('Ratio/Total_Offloaded_Ratio', np.sum(self.offloaded_task_cnts) / max(1, self.all_task_num), step)
        self.writer.add_scalar('Num/Total_Task_Num', self.all_task_num, step)
        self.writer.add_scalar('Latency/Total_Avg_Latency', np.sum(self.total_latencies) / max(1, self.all_task_num), step)

    def draw_env_figure(self, figure_path):
        env = self.env
        # 从env获取vehicle_by_index，里面各个vehicle的位置，根据vehicle.serving判断是任务车辆还是服务车辆
        # 从env获取UAVs和BSs的位置
        # 使用matplotlib画图，保存到tensorboard
        vehicle_by_index = env.vehicle_by_index
        uavs = env.UAVs
        bss = env.BSs
        # 画图
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(env.min_range_x, env.max_range_x)
        ax.set_ylim(env.min_range_y, env.max_range_y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Env Time: {self.env.cur_time}')
        # Create dummy plots for the legend
        ax.scatter([], [], c='r', marker='o', label='serving')
        ax.scatter([], [], c='b', marker='o', label='task')
        ax.scatter([], [], c='g', marker='^', label='uav')
        ax.scatter([], [], c='y', marker='^', label='bs')

        # Plot the points
        for vehicle in vehicle_by_index:
            if vehicle.serving:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='r', marker='o')
            else:
                ax.scatter(vehicle.position[0], vehicle.position[1], c='b', marker='o')
        for bs_id, bs in enumerate(bss):
            ax.scatter(bs.position[0], bs.position[1], c='y', marker='^')
            # 同时绘制出FL的cluster分类结果,self.RSU_labels
            ax.text(bs.position[0], bs.position[1], f'FL_cluster: {self.RSU_labels[bs_id]}', fontsize=10)
        for uav in uavs:
            ax.scatter(uav.position[0], uav.position[1], c='g', marker='D')

        ax.legend()
        plt.savefig(figure_path+f'/{self.update_cnt}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def store_experience(self, veh_positions, uav_positions, epi_id, isDone):
        # 存储veh和UAV position map
        veh_pos_maps = project_veh_position_to_map(veh_positions) # [400, 400], 每个网格内的veh数量
        uav_pos_maps = project_uav_position_to_map(uav_positions) # [400, 400], 每个网格内的uav数量
        self.veh_pos_map_traces.append(veh_pos_maps)
        self.uav_pos_map_traces.append(uav_pos_maps)

        other_agents_obs_for_RSUs = []
        for rsu in self.env.BSs:
            other_agents_obs_for_RSUs.append(rsu.all_agents_obs)
        tvcnt_for_RSU = [0 for _ in range(self.env.n_RSU)]
        for idx, veh in enumerate(self.env.vehicle_by_index):
            if veh.serving:
                continue
            rsu_id = self.env.BSs.index(veh.nearest_BS)
            obs = veh.get_std_observation(self.env)
            reward = self.task_qos_of_rsu[rsu_id][-1] #/ max(1, self.task_num_of_rsu[rsu_id][-1])
            if 'self' in self.args.method:
                reward = veh.reward
            done = isDone
            agent_id = veh.id
            other_agents_obs = other_agents_obs_for_RSUs[rsu_id].copy()
            # remove掉自己
            # other_agents_obs.pop(tvcnt_for_RSU[rsu_id])
            other_agents_obs = np.array(other_agents_obs)
            self.replay_buffer.add_experience(rsu_id, agent_id, epi_id, obs, veh.action, other_agents_obs, reward, done)
            tvcnt_for_RSU[rsu_id] += 1

        # 针对UAV
        for idx, uav in enumerate(self.env.UAVs):
            rsu_id = self.env.BSs.index(uav.nearest_BS)
            obs = uav.get_std_observation(self.env)
            reward = self.task_qos_of_rsu[rsu_id][-1] #/ max(1, self.task_num_of_rsu[rsu_id][-1])
            if 'self' in self.args.method:
                reward = uav.reward
            done = isDone
            agent_id = uav.id + 1000 # 为了和veh区分
            other_agents_obs = other_agents_obs_for_RSUs[rsu_id].copy()
            # other_agents_obs.pop(tvcnt_for_RSU[rsu_id])
            other_agents_obs = np.array(other_agents_obs)
            self.replay_buffer.add_experience(rsu_id, agent_id, epi_id, obs, uav.action, other_agents_obs, reward, done)
            tvcnt_for_RSU[rsu_id] += 1
        
    def act_CPU_allocation(self, env):
        cpu_allocation_for_fog_nodes = []
        serving_vehicles = env.serving_vehicles.values()
        bss = env.BSs
        devices = list(serving_vehicles) + list(bss)
        for device in devices:
            task_len = len(device.task_queue)
            if task_len > 0:
                info_dict = {}
                cheat_or_not = np.zeros(shape=(task_len), dtype='bool')
                info_dict['device'] = device
                cpu_alloc = np.zeros(shape=(task_len), dtype='float')
                cpu_alloc[0] = 1
                info_dict['CPU_allocation'] = cpu_alloc
                info_dict['is_to_cheat'] = cheat_or_not
                cpu_allocation_for_fog_nodes.append(info_dict)
        return cpu_allocation_for_fog_nodes

    
    def act_RB_allocation(self, env):
        activated_offloading_tasks_with_RB_Nos = np.zeros((len(env.offloading_tasks), env.n_RB), dtype='bool')
        # 每个RSU覆盖范围复用20个RB;均分
        # 1. 先把offloading_tasks按照RSU划分
        offloading_task_by_RSU = [[] for _ in range(env.n_RSU)]
        for i, task_info in enumerate(env.offloading_tasks):
            device = task_info['task'].g_veh
            rsu_id = env.BSs.index(device.nearest_BS)
            offloading_task_by_RSU[rsu_id].append(i)
        # 2. 对于每个RSU，按照时间顺序，分配RB: 如果大于20个任务，则只分配前20个任务；否则均分
        for rsu_id in range(env.n_RSU):
            task_num = len(offloading_task_by_RSU[rsu_id])
            can_off_num = task_num
            if task_num > 20:
                can_off_num = 20
            alloc_RB_num = max(1, 20 // max(1, can_off_num)) # 每个任务分配的RB数量
            for i in range(can_off_num):
                activated_offloading_tasks_with_RB_Nos[offloading_task_by_RSU[rsu_id][i], i*alloc_RB_num:(i+1)*alloc_RB_num] = True
        return activated_offloading_tasks_with_RB_Nos

    def act_offloading(self, env):
        '''遍历每一个taskV和UAV，输入observation和other_agents_observation，输出action，并且记录在env中'''
        # 1. 先遍历每个veh和uav的obs，存储在每个RSU中
        for bs in self.env.BSs:
            bs.all_agents_obs = []
            bs.served_agents = []
        for veh in self.env.vehicle_by_index:
            if veh.serving:
                continue
            obs = veh.get_std_observation(self.env)
            rsu_id = self.env.BSs.index(veh.nearest_BS)
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
            if 'MAPPO_Cen' in self.args.method:
                veh.nearest_BS.all_agents_obs.append(self.ppo_agents[0].state_encoder(obs_tensor).cpu().detach().numpy())
            else:
                veh.nearest_BS.all_agents_obs.append(self.ppo_agents[rsu_id].state_encoder(obs_tensor).cpu().detach().numpy())
        for uav in self.env.UAVs:
            obs = uav.get_std_observation(self.env)
            rsu_id = self.env.BSs.index(uav.nearest_BS)
            obs_tensor = torch.tensor(obs, dtype=torch.float).to(self.device)
            if 'MAPPO_Cen' in self.args.method:
                uav.nearest_BS.all_agents_obs.append(self.ppo_agents[0].state_encoder(obs_tensor).cpu().detach().numpy())
            else:
                uav.nearest_BS.all_agents_obs.append(self.ppo_agents[rsu_id].state_encoder(obs_tensor).cpu().detach().numpy())
        # 2. 然后对每一个taskV和UAV，根据所在的rsu_id，获取action
        tvcnt_for_RSU = [0 for _ in range(self.env.n_RSU)]
        for veh in self.env.vehicle_by_index:
            if veh.serving:
                veh.is_offloaded = None
                continue
            rsu_id = self.env.BSs.index(veh.nearest_BS)
            obs = veh.get_std_observation(self.env)
            other_agents_obs = veh.nearest_BS.all_agents_obs.copy()
            # 去除自己的部分
            # other_agents_obs.pop(tvcnt_for_RSU[rsu_id])
            if 'MAPPO_Cen' in self.args.method:
                action = self.ppo_agents[0].take_action(obs, other_agents_obs)
            elif 'greedy_notLearn' in self.args.method:
                device1 = veh.neighbor_vehicles[0] if len(veh.neighbor_vehicles) > 0 else veh.nearest_BS
                device2 = veh.nearest_BS
                # 判断哪个距离veh更近，就选择哪个（0或者5）
                if np.linalg.norm(np.array(veh.position) - np.array(device1.position)) < np.linalg.norm(np.array(veh.position) - np.array(device2.position)):
                    action = 0
                else:
                    action = self.args.v_neighbor_Veh
            else:
                action = self.ppo_agents[rsu_id].take_action(obs, other_agents_obs)
            veh.action = action
            tvcnt_for_RSU[rsu_id] += 1

        for idx, uav in enumerate(self.env.UAVs):
            rsu_id = self.env.BSs.index(uav.nearest_BS)
            obs = uav.get_std_observation(self.env)
            other_agents_obs = uav.nearest_BS.all_agents_obs.copy()
            # other_agents_obs.pop(tvcnt_for_RSU[rsu_id])
            if 'MAPPO_Cen' in self.args.method:
                action = self.ppo_agents[0].take_action(obs, other_agents_obs)
            elif 'greedy_notLearn' in self.args.method:
                device1 = uav.neighbor_vehicles[0] if len(uav.neighbor_vehicles) > 0 else uav.nearest_BS
                device2 = uav.nearest_BS
                # 判断哪个距离uav更近，就选择哪个（0或者5）
                if np.linalg.norm(np.array(uav.position) - np.array(device1.position)) < np.linalg.norm(np.array(uav.position) - np.array(device2.position)):
                    action = 0
                else:
                    action = self.args.v_neighbor_Veh
            else:
                action = self.ppo_agents[rsu_id].take_action(obs, other_agents_obs)
            uav.action = action
            tvcnt_for_RSU[rsu_id] += 1

        # 3. 转为offloading dict。全部任务都卸载
        self.offloading_tasks = []
        task_path_dict_list = []
        for task_idx, task in enumerate(env.to_offload_tasks):
            self.offloading_tasks.append(task) # 便于本轮统计reward
            gen_V = task.g_veh
            to_device = None
            if gen_V.action < len(gen_V.neighbor_vehicles): # 卸载给邻居车辆
                to_device = gen_V.neighbor_vehicles[gen_V.action]
            elif gen_V.action == self.args.v_neighbor_Veh: # 卸载给RSU
                to_device = gen_V.nearest_BS
                # gen_V.action = self.args.v_neighbor_Veh
            else: # 不卸载
                # gen_V.action = self.args.v_neighbor_Veh + 1
                task.is_offloaded = False
                continue
                
            # 1) 如果gen_V是vehicle，to_device也是vehicle，那么要求两者距离在self.args.V2V_communication_range范围内
            can_offload = True
            
            if isinstance(gen_V, Vehicle) and isinstance(to_device, Vehicle):
                if np.linalg.norm(np.array(gen_V.position) - np.array(to_device.position)) > self.args.V2V_communication_range:
                    can_offload = False
            # 2) 如果gen_V是UAV，to_device是vehicle，那么要求两者距离在self.args.UAV_communication_range范围内
            if isinstance(gen_V, UAV) and isinstance(to_device, Vehicle):
                # 除了2d的position，还要考虑uav的height
                uav_pos = [gen_V.position[0], gen_V.position[1], gen_V.height]
                tov_pos = [to_device.position[0], to_device.position[1], 0]
                if np.linalg.norm(np.array(uav_pos) - np.array(tov_pos)) > self.args.UAV_communication_range:
                    can_offload = False
            if 'free' not in self.args.method:
                if isinstance(to_device, Vehicle): 
                    if to_device.is_offloaded is None:
                        to_device.is_offloaded = gen_V
                    elif (to_device.is_offloaded != gen_V or to_device.nearest_BS != gen_V.nearest_BS):
                        can_offload = False
                elif isinstance(to_device, BS):
                    if gen_V not in to_device.served_agents:
                        if len(to_device.served_agents) >= 5:
                            can_offload = False
                        else:
                            to_device.served_agents.append(gen_V)
            else:
                if isinstance(to_device, Vehicle): 
                    if to_device.is_offloaded is None:
                        to_device.is_offloaded = [gen_V]
                    elif len(to_device.is_offloaded) < 3: # 每个fog-V最多offload 3个
                        if (to_device.nearest_BS != gen_V.nearest_BS):
                            can_offload = False
                        else:
                            to_device.is_offloaded.append(gen_V)
                    else:
                        if gen_V not in to_device.is_offloaded:
                            can_offload = False
                elif isinstance(to_device, BS):
                    if gen_V not in to_device.served_agents:
                        if len(to_device.served_agents) >= 5:
                            can_offload = False
                        else:
                            to_device.served_agents.append(gen_V)

            if can_offload:
                task.is_offloaded = True
                offload_path = [{
                    'X_device': to_device
                }]    
                
                task_path_dict_list.append({
                    'task':task,
                    'offload_path': offload_path,
                    'task_type':'offload'
                })
            else:
                task.is_offloaded = False
                # gen_V.action == self.args.v_neighbor_Veh + 1
        return task_path_dict_list
    # act_mining_and_pay, act_mobility, act_pay_and_punish, act_verification
    def act_mining_and_pay(self, env):
        pass
    def act_mobility(self, env):
        pass
    def act_pay_and_punish(self, env):
        pass
    def act_verification(self, env):
        pass