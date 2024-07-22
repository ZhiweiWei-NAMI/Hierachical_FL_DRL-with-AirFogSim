import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.hidden_layer = nn.Linear(state_size, state_size // 2)
        self.output_layer = nn.Linear(state_size // 2, action_size)

    def forward(self, state):
        x = nn.LeakyReLU()(self.hidden_layer(state))
        output = self.output_layer(x)
        # output = nn.Softmax()(output)
        return output

# Define the VDN Network
class VDN(nn.Module):
    def __init__(self, num_agents, state_size, action_size):
        super(VDN, self).__init__()

        self.q_networks = nn.ModuleList([QNetwork(state_size, action_size) for _ in range(num_agents)])

    def forward(self, states):
        q_values = [q_network(state) for q_network, state in zip(self.q_networks, states)]
        return sum(q_values)

class VDN_Agent:
    def __init__(self, device, num_agents, state_size, action_size, epsilon=0.1, gamma=0.99, learning_rate=0.001):
        self.vdn_network = VDN(num_agents, state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.vdn_network.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma
        self.train_num = 0
        self.device = device

    def update(self, transition_dicts, batch_idx):
        self.train_num += 1
        if self.epsilon > 0.1:
            self.epsilon *= 0.99
        states = [torch.tensor(np.array(transition_dict['states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict in transition_dicts]
        actions = [torch.tensor(np.array(transition_dict['actions_weight'])[batch_idx], dtype=torch.int64).to(self.device) for transition_dict in transition_dicts]
        next_states = [torch.tensor(np.array(transition_dict['next_states'])[batch_idx], dtype=torch.float).to(self.device) for transition_dict in transition_dicts]
        rewards = torch.tensor(np.array([np.array(transition_dict['rewards'])[batch_idx] for transition_dict in transition_dicts]), dtype=torch.float).to(self.device)

        # Zero the gradients
        self.optimizer.zero_grad()
        tot_q = torch.zeros_like(actions[0], dtype=torch.float).to(self.device).view(-1, 1)
        for idx, action in enumerate(actions):
            q_values = self.vdn_network.q_networks[idx](states[idx])
            action = action.view(-1,1)
            tot_q += q_values.gather(1, action)
        # expected_q_values = self.vdn_network(next_states).detach()

        # Compute the loss
        loss = ((torch.sum(rewards, dim=0) - tot_q) ** 2).mean()

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def select_actions(self, states):
        actions = []
        for idx, state in enumerate(states):
            # epsilon-greedy policy
            if np.random.rand() < self.epsilon:
                # Random action
                action = np.random.randint(0, self.vdn_network.q_networks[0].output_layer.out_features)
            else:
                # Greedy action
                with torch.no_grad():
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # add batch dimension
                    q_values = self.vdn_network.q_networks[idx](state)  # use the first Q-Network as the policy
                    action = q_values.argmax(dim=1).item()
            actions.append(action)

        return actions
    def save_agents(self, model_path):
        torch.save(self.vdn_network.state_dict(), model_path+'/weight_agent.pth')

    def load_agents(self, model_path, device):
        self.vdn_network.load_state_dict(torch.load(model_path+'/weight_agent.pth', map_location=device))