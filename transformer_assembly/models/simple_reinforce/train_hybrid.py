import json
from transformer_assembly.models.simple_reinforce.assembly_env import AssemblyTreeEnv
from transformer_assembly.models.base_stock.base_stock_policy import BaseStockPolicy
from transformer_assembly.models.simple_reinforce.reinforce import REINFORCE
import torch
import time
import gc

if __name__ == "__main__":
    import json
    import os
    import numpy as np
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "CPU")

    # Load configuration
    config_file_names = ['config_2_p50.json', 'config_2_p100.json',
                         'config_5_p50.json', 'config_5_p100.json',
                         'config_10_p50.json', 'config_10_p100.json',
                         'config_20_p50.json', 'config_20_p100.json']
    
    
    for config_id, config_name in enumerate(config_file_names):
        print(config_name)
        start_time = time.time()

        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        json_path = os.path.join(parent_dir, 'configs', 'env_configs', config_name)
        with open(json_path, 'r') as f:
            config = json.load(f)

        # Create environment with a fixed random seed
        random_seed = 42

        attention_env = AssemblyTreeEnv(config, seed=random_seed)
        base_env = AssemblyTreeEnv(config, seed=random_seed)
        base_greedy_env = AssemblyTreeEnv(config, seed=random_seed)

    # Initialize the BaseStockPolicy
        path_dir = os.path.dirname(os.path.abspath(__file__))
        policy_config_path = os.path.join(parent_dir, 'configs', 'policy_configs', config_name)
        with open(policy_config_path, 'r') as f:
            policy_config = json.load(f)
        print(policy_config)
        base_policy = BaseStockPolicy(base_env, policy_config)
        print(base_policy.base_stock_levels)
        num_nodes = attention_env.num_of_nodes

        # 设置 max_time_steps
        max_time_steps = 50 if num_nodes == 5 else 30 if num_nodes == 10 else 20
        if num_nodes == 2:
            max_time_steps = 50

        num_episodes = 10000
        load_path_list = [
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_2_p50.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_2_p100.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_5_p50.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_5_p100.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_10_p50.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_10_p100.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_20_p50.json_ep_2000.pth",
            "./transformer_assembly/pretrained/simple_RL/base_RL/RL_config_20_p100.json_ep_2000.pth" 
             ]

        save_model_cat = "hybrid_RL"
        
        save_path = f"./transformer_assembly/pretrained/simple_RL/{save_model_cat}/hybrid_RL_{config_name}_ep_{num_episodes}.pth"
        
        fig_save_path = f"./transformer_assembly/img/simple_RL/{save_model_cat}/loss_hybrid_{config_name}_ep_{num_episodes}.png"

        reinforce_agent = REINFORCE(attention_env, base_env, base_greedy_env, base_policy, lr=1e-4)

        reinforce_agent.train(num_episodes=num_episodes, max_time_steps=max_time_steps, save_path=save_path, load_path=load_path_list[config_id])
        # reinforce_agent.plot_loss_curve(fig_save_path, if_greedy=True)

        print(f"Training for {config_name} completed. Model saved to {save_path}\n")
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_formatted = f"{hours}h {minutes}m {seconds}s"

        print(f"Training time for {config_name}: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.0f}s\n")
        reinforce_agent.evaluate(num_episodes=10000, max_time_steps=100, use_baseline=True, save_results=True, result_prefix="hybrid_RL")
        reinforce_agent = None  
        env = None  
        torch.cuda.empty_cache()  
        gc.collect()  