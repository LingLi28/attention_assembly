import numpy as np
from transformer_assembly.models.dqn.assembly_env import AssemblyTreeEnv
from transformer_assembly.models.dqn.dqn_agent import DQNAgent
import json
import torch
import os
import pandas as pd

def evaluate(env, agent, config_name, num_episodes=1000, max_time_steps=100, trainep=20):
    """Evaluate the trained agent."""
    rewards = []
    inventory_levels = []
    lost_sales_rates = []
    
    if env.num_of_nodes == 2:
        max_time_steps = 50
    
    for ep in range(num_episodes):
        episode_rewards = []
        episode_lost_sales = 0
        episode_total_demand = 0
        state = env.reset()
        for ts in range(max_time_steps):
            demand = None
            if env.state['state_category'] == 'AwaitEvent':
                    demand = env.generate_random_demand()
            action = agent.act(state) + env.demand_offset
            # Verify the action using isAllowedAction()
            while not env.isAllowedAction(action):
                action = action - 1
            # Proceed with the step, state here is observation
            next_state, reward, done, _ = env.step(action, demand)
            state = next_state
            # 累计 lost sales 仅针对 Node 0
            if demand is not None and (state[0] - env.I_offset) < 0:
                episode_lost_sales += min(abs(state[0] - env.I_offset), demand)

            if demand is not None:
                episode_total_demand += demand
            
            if reward < 0:  # Only append negative rewards for penalty analysis
                episode_rewards.append(reward)
            inventory = state[0:env.num_of_nodes]
            inventory[0] -= env.I_offset  # Adjust node_id = 0
            inventory_levels.append(inventory)
            
        if episode_total_demand > 0:
            lost_sales_rate = episode_lost_sales / episode_total_demand
            lost_sales_rates.append(lost_sales_rate)

        rewards.append(np.mean(episode_rewards))

    # Calculate statistics
    avg_inventory_per_node = np.mean(inventory_levels, axis=0)
    std_inventory_per_node = np.std(inventory_levels, axis=0)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    lost_sales_rate_mean = np.mean(lost_sales_rates)
    lost_sales_rate_std = np.std(lost_sales_rates)


    print('Evaluation Results:')
    print('Average Inventory Per Node:', avg_inventory_per_node)
    print('Inventory Std Per Node:', std_inventory_per_node)
    print('Average Reward:', avg_reward)
    print('Reward Std Dev:', std_reward)
    print("lost_sales_rate_mean", lost_sales_rate_mean)
    print("lost_sales_rate_std", lost_sales_rate_std)


    # Save results to CSV or Excel
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(parent_dir, 'experiment_results', 'dqn')
    os.makedirs(results_dir, exist_ok=True)

    result_data = {
        "Metric": ["Average Reward", "Reward Std Dev", "Lost Sales Rate Mean", "Lost Sales Rate Std Dev"] +
                    [f"Node {i} Avg Inventory" for i in range(env.num_of_nodes)] +
                    [f"Node {i} Inventory Std Dev" for i in range(env.num_of_nodes)],
        "Value": [avg_reward, std_reward, lost_sales_rate_mean, lost_sales_rate_std] +
                    list(avg_inventory_per_node) + list(std_inventory_per_node)
    }
    result_df = pd.DataFrame(result_data)

    # Save as Excel
    excel_file_path = os.path.join(results_dir, f'evaluation_results_{config_name}_ts{max_time_steps}_trainep{trainep}.xlsx')
    with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Results')

    print(f"Evaluation results saved to {excel_file_path}")


def main():
    # Load configuration
    config_file_names = [
        'config_2_p50.json',
        'config_2_p100.json',
        'config_5_p50.json',
        'config_5_p100.json',
        'config_10_p50.json',
        'config_10_p100.json',
        'config_20_p50.json',
        'config_20_p100.json'
    ]

    for config_id, config_name in enumerate(config_file_names):
        print(config_name)

        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(parent_dir, 'configs', 'env_configs', config_name)
        with open(json_path, 'r') as f:
            config = json.load(f)

        # Create environment with a fixed random seed
        random_seed = 42
        env = AssemblyTreeEnv(config, seed=random_seed)
        print(env.assembly_tree)

        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        num_nodes = env.num_of_nodes
        print('num of nodes: ', num_nodes)

        # Initialize the DQN agent
        agent = DQNAgent(state_size, action_size, n_steps = num_nodes + 1)

        # Training parameters
        episodes = 10000
        target_update_freq = 50

        # 设置 max_time_steps
        max_time_steps = 50 if num_nodes == 5 else 30 if num_nodes == 10 else 20
        if num_nodes == 2:
            max_time_steps = 50
            
        # Train the agent
        for e in range(episodes):
            
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            for time in range(max_time_steps*(num_nodes + 1)):
                demand = None
                if env.state['state_category'] == 'AwaitEvent':
                    demand = env.generate_random_demand()
                action = agent.act(state) + env.demand_offset
                # Verify the action using isAllowedAction()
                while not env.isAllowedAction(action):
                    action = action - 1
                # Proceed with the step, state here is observation
                next_state, reward, done, _ = env.step(action, demand)
                next_state = np.reshape(next_state, [1, state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                agent.replay()

            # Update target network every `target_update_freq` episodes
            if e % target_update_freq == 0:
                # print('Episode:', e)
                agent.update_target_network()

        # Save the trained model
        save_path = f"./transformer_assembly/pretrained/dqn/dqn_{config_name}_ep{episodes}.pth"
        agent.save(save_path)
        print(f"Model saved to {save_path}")

        # Evaluate the trained model
        print("Starting evaluation...")
        evaluate(env, agent, config_name, num_episodes=1000, max_time_steps=100, trainep=episodes)


if __name__ == "__main__":
    main()
