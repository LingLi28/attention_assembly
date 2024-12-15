import os
import json
import numpy as np
import pandas as pd
import gc
from transformer_assembly.env.attention_assembly_env import AssemblyTreeEnv
from transformer_assembly.models.base_stock.base_stock_policy import BaseStockPolicy


parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
config_file_names = ['config_2_p50.json', 'config_2_p100.json',
                     'config_5_p50.json', 'config_5_p100.json',
                     'config_10_p50.json', 'config_10_p100.json',
                     'config_20_p50.json', 'config_20_p100.json']

config_file_names = ['config_20_p50.json', 'config_20_p100.json']
num_episodes = 10000  
warmup_timesteps = 0  
num_timesteps = 100 


results_dir = os.path.join(parent_dir, 'evaluation_results','base_stock_new')
os.makedirs(results_dir, exist_ok=True)


for config_id, config_name in  enumerate(config_file_names):
    json_path = os.path.join(parent_dir, 'configs', 'env_configs', config_name)
    with open(json_path, 'r') as f:
        config = json.load(f)

    if_short_falls = [False]

    for short_fall_flag in if_short_falls: 
        print("short_fall_flag:", short_fall_flag)   
        env = AssemblyTreeEnv(config, seed=42)
        print(env.assembly_tree)
        policy_config_path = os.path.join(parent_dir, 'configs', 'policy_configs', config_name)
        with open(policy_config_path, 'r') as f:
            policy_config = json.load(f)
            print(policy_config)
        policy = BaseStockPolicy(env, policy_config, short_fall_flag)
        
        num_nodes = env.num_of_nodes  
        batch_size = num_nodes + 1
        
        if env.num_of_nodes == 2:
            num_timesteps = 50
      
        rewards_all_episodes = []
        holding_cost_all_episodes = []
        backorder_cost_all_episodes = []
        
        lost_sales_rates = []
        inventory_all_episodes = np.zeros((num_episodes, num_nodes))

        for episode in range(num_episodes):
            state = env.reset()
            timestep_rewards = []
            inventory_levels = []
            episode_lost_sales = 0
            episode_total_demand = 0
            timestep_holding_cost = []
            timestep_backorder_cost = []

            for t in range(num_timesteps):
                demand = None
                if state['state_category'] == 'AwaitEvent':
                    demand = env.generate_random_demand()
                    # print('demand,', demand)
                action = policy.get_action(state) + env.demand_offset
                while not env.isAllowedAction(action):
                    action = action - 1

                observation, reward, done, info = env.step(action, demand)
                state = observation
                if demand is not None and state['vector_of_inventory'][0] - env.I_offset < 0:
                    episode_lost_sales += min(abs(state['vector_of_inventory'][0] - env.I_offset), demand)

                if demand is not None:
                    episode_total_demand += demand
                if reward < 0:
                    timestep_rewards.append(reward)
                    timestep_holding_cost.append(info['holding_cost'])
                    timestep_backorder_cost.append(info['backorder_cost'])
                    
                inventory = np.array(state['vector_of_inventory'])
                inventory[0] -= env.I_offset  # 调整 node_id = 0
                inventory_levels.append(inventory)

            if episode_total_demand > 0:
                lost_sales_rate = episode_lost_sales / episode_total_demand
                lost_sales_rates.append(lost_sales_rate)
            inventory_all_episodes[episode, :] = np.mean(inventory_levels, axis=0)
            rewards_all_episodes.append(np.mean(timestep_rewards))
            holding_cost_all_episodes.append(np.mean(timestep_holding_cost))
            backorder_cost_all_episodes.append(np.mean(timestep_backorder_cost))

        # 计算奖励和库存的总体均值和方差
        avg_reward = np.mean(rewards_all_episodes)
        std_reward = np.std(rewards_all_episodes)
        avg_holding_cost = np.mean(holding_cost_all_episodes)
        std_holding_cost = np.std(holding_cost_all_episodes)
        avg_backorder_cost = np.mean(backorder_cost_all_episodes)
        std_backorder_cost = np.std(backorder_cost_all_episodes)
        
        lost_sales_rate_mean = np.mean(lost_sales_rates)
        lost_sales_rate_std = np.std(lost_sales_rates)
        print("lost_sales_rate_mean", lost_sales_rate_mean)
        print("lost_sales_rate_std", lost_sales_rate_std)


        avg_inventory_per_node = np.mean(inventory_all_episodes, axis=0)  # 每个节点的库存均值
        std_inventory_per_node = np.std(inventory_all_episodes, axis=0)  # 每个节点的库存方差

        
        # 保存统计结果至 CSV 文件
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

        # 指定文件保存路径
        excel_file_path = os.path.join(results_dir, f'summary_{config_name}_ep{num_episodes}.xlsx')

        # 保存为 Excel 文件
        with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False, sheet_name='Results')

        print(f"Results for {config_name} if_shortfall_{short_fall_flag} saved to {excel_file_path}")
        del policy, env, rewards_all_episodes, inventory_all_episodes
        gc.collect()
