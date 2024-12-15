import torch
import torch.optim as optim
import torch.nn as nn
from transformer_assembly.models.hybrid_reinforce.attention_model_with_encoder import AttentionModel
from transformer_assembly.env.attention_assembly_env import process_state
from transformer_assembly.models.evaluate import hybrid_evaluate_model
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from scipy.stats import ttest_ind
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import os
import pandas as pd

class REINFORCE(nn.Module):
    def __init__(self, env, base_env, greedy_env, base_policy, embedding_dim=128, num_heads=8, lr=1e-4, gamma=0.8, max_time_steps=100, improvement_threshold=0.01, T_max=100):
        super(REINFORCE, self).__init__()
        self.env = env
        self.base_env = base_env
        self.greedy_env =greedy_env

        self.action_not_allowed_penalty = -10

        self.base_policy = base_policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.gamma = gamma
        self.max_time_steps = max_time_steps
        self.improvement_threshold = improvement_threshold
        self.base_stock_loss_history = []
        self.greedy_loss_history = []

       # 初始化模型
        self.attention_model = AttentionModel(
            env.num_of_nodes, embedding_dim, num_heads, env.action_space.n, env.max_lead_time
        ).to(self.device)
        
        self.baseline_model = AttentionModel(
            env.num_of_nodes, embedding_dim, num_heads, env.action_space.n, env.max_lead_time
        ).to(self.device)

        # 初始化优化器
        self.optimizer = optim.Adam(self.attention_model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=1e-6)


    def forward(self, state_tensor, use_baseline=False):
        num_nodes = self.env.num_of_nodes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        capacities = torch.tensor(list(self.env.assembly_tree.capacities.values()), dtype=torch.float32).to(self.device)
        holding_costs = torch.tensor(list(self.env.assembly_tree.holdingCosts.values()), dtype=torch.float32).to(self.device)
        lead_times = torch.tensor(list(self.env.assembly_tree.leadTimes.values()), dtype=torch.float32).to(self.device)
        inventory = state_tensor[:num_nodes].to(self.device)
        pipeline = state_tensor[num_nodes:num_nodes * self.env.max_lead_time + num_nodes].view(num_nodes, -1).to(self.device)

        current_node = state_tensor[-1].unsqueeze(0)  # Current node as a single element tensor

        model = self.baseline_model if use_baseline else self.attention_model
        q_values = model.forward(capacities, holding_costs, lead_times, inventory, pipeline, current_node)
        if use_baseline:
            q_values = q_values.clone().detach()
        return q_values


    def _train_based_policy(self, num_episodes, max_time_steps, save_path, batch_size = 4):
        
        for episode in range(0, num_episodes, batch_size):
            all_log_probs, all_rewards = [], []
             # 批量采样多个轨迹
            for batch_idx in range(batch_size):
                log_probs, rewards = [], []
                state = self.env.reset()
                base_state = self.base_env.reset() 
 
                for time_step in range(max_time_steps):
                    for _ in range(self.env.num_of_nodes + 1):   
                        demand = self.env.generate_random_demand() if self.env.state['state_category'] == 'AwaitEvent' else None
                        base_action = self.base_policy.get_action(base_state) + self.env.demand_offset
                        action, log_prob = self.select_action(state)
                        action = action + self.env.demand_offset

                        while not self.base_env.isAllowedAction(base_action):
                            base_action -= 1
                        next_base_state, base_reward, _, _ = self.base_env.step(base_action, demand)
                        base_state = next_base_state
                        penalty = 0
                        while not self.env.isAllowedAction(action):
                            penalty += self.action_not_allowed_penalty
                            action -= 1
                        next_state, reward, done, _ = self.env.step(action, demand)
                        reward += penalty
                        state = next_state

                        advantage = reward - base_reward
                        log_probs.append(log_prob)
                        rewards.append(advantage)
                        if done:
                            break
            
                all_log_probs.append(log_probs)
                all_rewards.append(rewards)
            
            loss = self.update_model(all_log_probs, all_rewards)
            if (episode) % 3*batch_size == 0:
                print('model state: ', state)
                print('base_state: ', base_state)
                print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")


    def _train_based_greedy(self, save_path, num_episodes, max_time_steps):
        for episode in range(num_episodes):
            state = self.env.reset()
            greedy_state = self.greedy_env.reset()

            model_rewards, baseline_rewards = [], []
            log_probs, rewards = [], []
            for time_step in range(max_time_steps):
                for _ in range(self.env.num_of_nodes + 1):
                    demand = self.env.generate_random_demand() if self.env.state['state_category'] == 'AwaitEvent' else None
                    greedy_action = self.greedy_action(greedy_state, use_baseline=True) + self.env.demand_offset
                    action, log_prob = self.select_action(state) 
                    action = action + self.env.demand_offset

                    while not self.greedy_env.isAllowedAction(greedy_action):
                        greedy_action = greedy_action -1

                    while not self.env.isAllowedAction(action):
                        action = action - 1
                    next_state, reward, done, _ = self.env.step(action, demand)
                    next_greedy_state, base_reward, _, _ = self.greedy_env.step(greedy_action, demand)

                    model_rewards.append(reward)
                    baseline_rewards.append(base_reward)
                    advantage = reward - base_reward
                    log_probs.append(log_prob)
                    rewards.append(advantage)
                    if done:
                        break
                    state = next_state
                    greedy_state = next_greedy_state
            
            loss = self.update_model(log_probs, rewards, if_base_stock=False, batch_size=1)
            if (episode + 1) % 200 == 0:
                self.save_model(save_path)          
                
                if (episode + 1) % 200 == 0:
                    self.check_model_update()
                    model_rewards_eval, baseline_rewards_eval = hybrid_evaluate_model(self, self.env, self.greedy_env, num_episodes=100)
        
                    # Perform one-sided t-test to check if model's rewards are significantly higher
                    t_stat, p_value = ttest_ind(model_rewards_eval, baseline_rewards_eval, alternative='greater')
                    if p_value < 0.05:  # Significant at alpha = 0.05
                        # Update baseline model if statistically significant
                        self.update_baseline_model()
                        print(f"Updated baseline model at episode {episode + 1} due to statistically significant performance improvement.")
                       
                    else:
                        pass
                        # print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item():.4f}")
                    self.save_model(save_path)

        
    def train(self, save_path, load_path=None, num_episodes=1000, max_time_steps=50, batch_size = 4):
        
        if load_path:
            self.load_model(load_path=load_path)
            self._train_based_greedy(save_path, num_episodes, max_time_steps)
        else:
            print('load path is none!')
            self._train_based_policy(num_episodes, max_time_steps, save_path, batch_size)

        if save_path:
            self.save_model(save_path)

    def update_model(self, log_probs, rewards, if_base_stock = True, batch_size = 4):
        batch_loss = 0
        if self.env.num_of_nodes == 5:
                    offset = 10000
        elif self.env.num_of_nodes == 10:
            offset = 10000
        elif self.env.num_of_nodes == 20: 
            offset = 10000
        elif self.env.num_of_nodes == 2: 
            offset = 10000
        if batch_size == 1:
            discounted_rewards = self.compute_discounted_rewards(rewards)
            batch_loss = batch_loss + sum(-log_prob * G_t for log_prob, G_t in zip(log_probs, discounted_rewards)) / offset
        else:
            for i in range(batch_size):
                discounted_rewards = self.compute_discounted_rewards(rewards[i])
                batch_loss = batch_loss + sum(-log_prob * G_t for log_prob, G_t in zip(log_probs[i], discounted_rewards)) / offset
            batch_loss =  batch_loss / batch_size
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        if if_base_stock:
            self.base_stock_loss_history.append(batch_loss.item())
        else :
            self.greedy_loss_history.append(batch_loss.item())
        return batch_loss

    def compute_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards

    def update_baseline_model(self):
        self.baseline_model.load_state_dict(self.attention_model.state_dict())

    def select_action(self, state):
        epsilon = 1e-8
        max_pipeline_length = max(len(pipe) for pipe in state['matrix_of_pipeline'])
        state_tensor = process_state(state, max_pipeline_length).to(self.device)
        q_values = self.forward(state_tensor, use_baseline=False)
        q_values = (q_values - q_values.mean()) / (q_values.std() + epsilon) # 避免log1 = 0
        
        probabilities = torch.softmax(q_values, dim=-1)
        action = torch.multinomial(probabilities, 1).item()
        return action, torch.log(probabilities[action] + epsilon)
    
    def select_action_evaluate(self, state):
        epsilon = 1e-8
        max_pipeline_length = max(len(pipe) for pipe in state['matrix_of_pipeline'])
        state_tensor = process_state(state, max_pipeline_length).to(self.device)
        q_values = self.forward(state_tensor, use_baseline=True)
        q_values = (q_values - q_values.mean()) / (q_values.std() + epsilon) # 避免log1 = 0
        
        probabilities = torch.softmax(q_values, dim=-1)
        action = torch.multinomial(probabilities, 1).item()
        return action

    def greedy_action(self, state, use_baseline):
        max_pipeline_length = max(len(pipe) for pipe in state['matrix_of_pipeline'])
        state_tensor = process_state(state, max_pipeline_length).to(self.device)
    
        q_values = self.forward(state_tensor, use_baseline=use_baseline)
        action = torch.argmax(q_values).item()
        return action

    def save_model(self, save_path):
        """
        保存模型权重。
        """
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_path):
        """
        加载模型权重。
        """
        self.load_state_dict(torch.load(load_path, map_location=self.device), strict=False)
        # 将当前 attention_model 的参数复制到 baseline_model
        self.baseline_model.load_state_dict(self.attention_model.state_dict())
        # 检查加载后两个模型的参数是否相同
        same_params = all(torch.equal(param1, param2) 
                        for param1, param2 in zip(self.attention_model.parameters(), self.baseline_model.parameters()))
        if same_params:
            print("Warning: Loaded AttentionModel and BaselineModel have identical parameters!")
        else:
            print("Loaded models have different parameters.")

        print(f"Model loaded from {load_path}")
    
    def load_model_evaluate(self, load_path):
        """
        加载模型权重。
        """
        self.load_state_dict(torch.load(load_path, map_location=self.device), strict=True)
        same_params = all(torch.equal(param1, param2) 
                        for param1, param2 in zip(self.attention_model.parameters(), self.baseline_model.parameters()))
        if same_params:
            print("Warning: Loaded AttentionModel and BaselineModel have identical parameters!")
        else:
            print("Loaded models have different parameters.")

        print(f"Model loaded from {load_path}")

    def plot_loss_curve(self, save_path, if_greedy):
        plt.figure(figsize=(10, 5))

        if if_greedy:
            plt.plot(self.greedy_loss_history, label="Training Loss")
        else: 
            plt.plot(self.base_stock_loss_history, label="Training Loss")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Training Loss over Episodes")
        plt.legend()
        plt.savefig(save_path)
    
    def check_model_update(self):
        """
        检查 AttentionModel 和 BaselineModel 的权重是否相同。
        """
        same_params = all(torch.equal(param1, param2) 
                        for param1, param2 in zip(self.attention_model.parameters(), self.baseline_model.parameters()))
        if same_params:
            print("AttentionModel and BaselineModel have identical parameters!")
        else:
            print("AttentionModel and BaselineModel have different parameters.")
