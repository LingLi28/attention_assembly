import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import torch
from transformer_assembly.env.attention_assembly_env import AssemblyTreeEnv
from transformer_assembly.models.base_stock.base_stock_policy import BaseStockPolicy
from transformer_assembly.models.hybrid_reinforce.reinforce import REINFORCE

def evaluate_from_pth(pth_file_path, env_config_path, policy_config_path, num_episodes=1000, max_time_steps=100, use_baseline=True, save_results=True, result_prefix="hybrid_RL"):
    # Load environment configuration
    with open(env_config_path, 'r') as f:
        env_config = json.load(f)

    # Load environment and policy
    env = AssemblyTreeEnv(env_config)
    with open(policy_config_path, 'r') as f:
        policy_config = json.load(f)
    base_policy = BaseStockPolicy(env, policy_config)
    num_nodes = env.num_of_nodes
    embedding_dim, num_heads = {
            5: (128, 8),
            10: (128, 8),
            20: (256, 8)
        }.get(num_nodes, (128, 8)) 

    # Initialize model and load weights
    agent = REINFORCE(env, env, env, base_policy, embedding_dim=embedding_dim, num_heads=num_heads,lr=1e-4)
    agent.load_model_evaluate(pth_file_path)

    # Adjust max_time_steps for specific configurations
    if env.num_of_nodes == 2:
        max_time_steps = 50

    rewards = []
    holding_cost_all_episodes = []
    backorder_cost_all_episodes = []
    inventory_levels = []
    lost_sales_rates = []

    for ep in range(num_episodes):
        episode_rewards = []
        timestep_holding_cost = []
        timestep_backorder_cost = []
        state = env.reset()
        episode_lost_sales = 0
        episode_total_demand = 0

        for ts in range(max_time_steps):
            demand = None
            if env.state['state_category'] == 'AwaitEvent':
                demand = env.generate_random_demand()
                next_state, reward, done, info = env.step(0, demand)
            else:
                action = agent.greedy_action(state, use_baseline=use_baseline)
                action += env.demand_offset
                while not env.isAllowedAction(action):
                    action -= 1
                next_state, reward, done, _ = env.step(action, demand)

            state = next_state

            if demand is not None and env.state['vector_of_inventory'][0] - env.I_offset < 0:
                episode_lost_sales += min(abs(env.state['vector_of_inventory'][0] - env.I_offset), demand)

            if demand is not None:
                episode_total_demand += demand

            if reward < 0:
                episode_rewards.append(reward)
                timestep_holding_cost.append(info['holding_cost'])
                timestep_backorder_cost.append(info['backorder_cost'])

            inventory = np.array(env.state['vector_of_inventory'])
            inventory[0] -= env.I_offset
            inventory_levels.append(inventory)

        if episode_total_demand > 0:
            lost_sales_rate = episode_lost_sales / episode_total_demand
            lost_sales_rates.append(lost_sales_rate)

        rewards.append(np.mean(episode_rewards))
        holding_cost_all_episodes.append(np.mean(timestep_holding_cost))
        backorder_cost_all_episodes.append(np.mean(timestep_backorder_cost))

    avg_inventory_per_node = np.mean(inventory_levels, axis=0)
    std_inventory_per_node = np.std(inventory_levels, axis=0)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_holding_cost = np.mean(holding_cost_all_episodes)
    std_holding_cost = np.std(holding_cost_all_episodes)
    avg_backorder_cost = np.mean(backorder_cost_all_episodes)
    std_backorder_cost = np.std(backorder_cost_all_episodes)
    lost_sales_rate_mean = np.mean(lost_sales_rates)
    lost_sales_rate_std = np.std(lost_sales_rates)

    print(f"Evaluation Results:")
    print(f"Average Reward: {avg_reward}")
    print(f"Reward Std Dev: {std_reward}")
    print(f"Lost Sales Rate Mean: {lost_sales_rate_mean}")
    print(f"Lost Sales Rate Std Dev: {lost_sales_rate_std}")
    print(f"Average Inventory per Node: {avg_inventory_per_node}")
    print(f"Inventory Std Dev per Node: {std_inventory_per_node}")

    if save_results:
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(parent_dir, 'final_results', 'evaluation_results', 'att_RL', result_prefix)
        os.makedirs(results_dir, exist_ok=True)

        result_data = {
            "Metric": ["Average Reward", "Reward Std Dev","Average holding cost", "Holding Cost Std Dev",
                       "Average backorder cost", "Backorder cost Std Dev",
                       "Lost Sales Rate Mean", "Lost Sales Rate Std Dev"] +
                      [f"Node {i} Avg Inventory" for i in range(env.num_of_nodes)] +
                      [f"Node {i} Inventory Std Dev" for i in range(env.num_of_nodes)],
            "Value": [avg_reward, std_reward, avg_holding_cost, std_holding_cost, 
                      avg_backorder_cost, std_backorder_cost,
                      lost_sales_rate_mean, lost_sales_rate_std] +
                     list(avg_inventory_per_node) + list(std_inventory_per_node)
        }
        result_df = pd.DataFrame(result_data)

        excel_file_path = os.path.join(
            results_dir, f'summary_hybrid_numOfNodes{env.num_of_nodes}_p{env.back_order_cost}_ep{num_episodes}.xlsx'
        )
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Results')

        print(f"Results saved to {excel_file_path}")

    return avg_reward, std_reward, lost_sales_rate_mean, lost_sales_rate_std

if __name__ == "__main__":
    # Paths to files
    pth_file_paths = ["./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_2_p50.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_2_p100.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_5_p50.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_5_p100.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_10_p50.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_10_p100.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_20_p50.json_ep_10000.pth",
                      "./transformer_assembly/results/models/attention_RL/hybrid_RL/hybrid_RL_config_20_p100.json_ep_10000.pth"]
    
    env_config_paths = ["./transformer_assembly/configs/env_configs/config_2_p50.json",
                        "./transformer_assembly/configs/env_configs/config_2_p100.json",
                        "./transformer_assembly/configs/env_configs/config_5_p50.json",
                        "./transformer_assembly/configs/env_configs/config_5_p100.json",
                        "./transformer_assembly/configs/env_configs/config_10_p50.json",
                        "./transformer_assembly/configs/env_configs/config_10_p100.json",
                        "./transformer_assembly/configs/env_configs/config_20_p50.json",
                        "./transformer_assembly/configs/env_configs/config_20_p100.json"]
    
    policy_config_paths = ["./transformer_assembly/configs/policy_configs/config_2_p50.json",
                           "./transformer_assembly/configs/policy_configs/config_2_p100.json",
                           "./transformer_assembly/configs/policy_configs/config_5_p50.json",
                           "./transformer_assembly/configs/policy_configs/config_5_p100.json",
                           "./transformer_assembly/configs/policy_configs/config_10_p50.json",
                           "./transformer_assembly/configs/policy_configs/config_10_p100.json",
                           "./transformer_assembly/configs/policy_configs/config_20_p50.json",
                           "./transformer_assembly/configs/policy_configs/config_20_p100.json"]

    # Evaluation parameters
    num_episodes = 10000
    max_time_steps = 100

    # Evaluate the model
    avg_reward, std_reward, lost_sales_rate_mean, lost_sales_rate_std = evaluate_from_pth(
        pth_file_paths[7],
        env_config_paths[7],
        policy_config_paths[7],
        num_episodes=num_episodes,
        max_time_steps=max_time_steps
    )
