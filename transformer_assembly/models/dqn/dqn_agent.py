import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNetwork, self).__init__()
        # 定义隐藏层结构 (256, 128, 128, 64)
        self.fc1 = nn.Linear(state_size, 256)  
        self.fc2 = nn.Linear(256, 128)         
        self.fc3 = nn.Linear(128, 128)        
        self.fc4 = nn.Linear(128, 64)         
        self.fc5 = nn.Linear(64, action_size) 

    def forward(self, x):
        # 依次通过隐藏层，激活函数使用 ReLU
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)  # 输出动作值

class DQNAgent:
    def __init__(self, state_size, action_size, n_steps, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=1e-4, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_steps = n_steps
        self.n_step_buffer = deque(maxlen=n_steps)  # N-step buffer

        # Create the Q-network and target network
        self.policy_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Initialize target network weights to be the same as policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

    def remember(self, state, action, reward, next_state, done):
        """
        Store transitions with average reward using N-step buffer.
        """
        # Add current transition to the N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # When the buffer is full, calculate average reward and append each step
        if len(self.n_step_buffer) == self.n_steps:
            # Compute total discounted reward
            total_discounted_reward = sum([
                self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_steps)
            ])
            # Calculate average reward
            average_reward = total_discounted_reward / self.n_steps

            # Append each step in the buffer to memory with the average reward
            for i in range(self.n_steps):
                step_state, step_action, _, step_next_state, step_done = self.n_step_buffer[i]
                self.memory.append((step_state, step_action, average_reward, step_next_state, step_done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            act_values = self.policy_network(state)
        return np.argmax(act_values.cpu().data.numpy())

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Remove unnecessary dimension from states
        states = torch.FloatTensor(states).to(self.device).squeeze(1)  # Now shape should be (batch_size, state_size)
        actions = torch.LongTensor(actions).to(self.device)  # Ensure actions are long type
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device).squeeze(1)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get the current Q-values for the selected actions
        q_values = self.policy_network(states)  # Shape should be (batch_size, action_size)
        
        # Ensure actions have correct shape and gather the Q-values for the selected actions
        actions = actions.unsqueeze(1)  # Change actions shape to (batch_size, 1)
        
        # Gather the Q-values for the selected actions
        q_values = q_values.gather(1, actions).squeeze()  # Should gather the Q-value for each (state, action)
        
        # Compute the target Q-values
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def update_target_network(self):
        """ Copy weights from policy network to target network """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def load(self, name):
        self.policy_network.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.policy_network.state_dict(), name)
