import numpy as np
from transformer_assembly.env.attention_assembly_env import process_state
import pdb

def hybrid_evaluate_model(agent, env, base_env, num_episodes=1000, max_time_steps=100):
    
    all_episode_rewards = []
    all_episode_base_rewards = []
    
    if env.num_of_nodes == 2:
            max_time_steps = 50

    for episode in range(num_episodes):
        state = env.reset()
        base_state = base_env.reset()
        episode_rewards = []
        episode_base_rewards = []

        for timestep in range(max_time_steps):
            demand = None
            if env.state['state_category'] == 'AwaitEvent':
                demand = env.generate_random_demand()

            action_model = agent.greedy_action(state, use_baseline=False) + env.demand_offset
            action_baseline = agent.greedy_action(base_state, use_baseline=True) + env.demand_offset
            
            while not env.isAllowedAction(action_model):
                action_model -= 1
            
            while not base_env.isAllowedAction(action_baseline):
                action_baseline -= 1

            next_state, reward, done, _ = env.step(action_model, demand)
            next_base_state, base_reward, done, _ = base_env.step(action_baseline, demand)

            if reward < 0:
                episode_rewards.append(reward)
            if base_reward < 0: 
                episode_base_rewards.append(base_reward)
            state = next_state
            base_state = next_base_state

        all_episode_rewards.append(np.mean(episode_rewards))
        all_episode_base_rewards.append(np.mean(episode_base_rewards))
 
    return all_episode_rewards, all_episode_base_rewards
    

def evaluate_model(agent, env, num_episodes=50, max_time_steps=100):
   
    rewards_all_episodes = []
    inventory_levels_all = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_inventory_levels = []
        episode_rewards = []

        for timestep in range(max_time_steps):
            demand = None
            if env.state['state_category'] == 'AwaitEvent':
                demand = env.generate_random_demand()

            action = agent.greedy_action(state)
            action = action + env.demand_offset

            while not env.isAllowedAction(action):
                action -= 1

            next_state, reward, done, _ = env.step(action, demand)

            if reward < 0:
                episode_rewards.append(reward)

            inventory = np.array(state['vector_of_inventory'])
            inventory[0] -= env.I_offset  
            episode_inventory_levels.append(inventory)

            state = next_state

            if done:
                break
        rewards_all_episodes.append(np.mean(episode_rewards))
        inventory_levels_all.append(np.mean(episode_inventory_levels, axis=0))

    avg_reward = np.mean(rewards_all_episodes)
    std_reward = np.std(rewards_all_episodes)

    avg_inventory_per_node = np.mean(inventory_levels_all, axis=0)
    std_inventory_per_node = np.std(inventory_levels_all, axis=0)

    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.4f}, Reward Std Dev: {std_reward:.4f}")
    for i, (avg_inv, std_inv) in enumerate(zip(avg_inventory_per_node, std_inventory_per_node)):
        print(f"Node {i} Avg Inventory: {avg_inv:.4f}, Std Dev: {std_inv:.4f}")

    
    return avg_reward, std_reward, avg_inventory_per_node, std_inventory_per_node

