import json
from transformer_assembly.env.attention_assembly_env import AssemblyTreeEnv
from transformer_assembly.models.base_stock.base_stock_policy import BaseStockPolicy
from transformer_assembly.models.hybrid_reinforce.reinforce import REINFORCE
import torch
import os
import time
import gc
import multiprocessing as mp

def train_model_with_config(config_name, config_id, load_path, save_model_cat, num_episodes=2000, gpu_id=None):
    
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Process {config_id}: Using GPU {gpu_id}")

    print(f"Starting training for {config_name}")
    start_time = time.time()

    # Load config
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(parent_dir, 'configs', 'env_configs', config_name)
    with open(json_path, 'r') as f:
        config = json.load(f)

    random_seed = 42
    attention_env = AssemblyTreeEnv(config, seed=random_seed)
    base_env = AssemblyTreeEnv(config, seed=random_seed)
    base_greedy_env = AssemblyTreeEnv(config, seed=random_seed)

    # Initialize BaseStockPolicy
    policy_config_path = os.path.join(parent_dir, 'configs', 'policy_configs', config_name)
    with open(policy_config_path, 'r') as f:
        policy_config = json.load(f)
    base_policy = BaseStockPolicy(base_env, policy_config)

    # Determine max_time_steps
    num_nodes = attention_env.num_of_nodes
    max_time_steps = 50 if num_nodes == 5 else 30 if num_nodes == 10 else 20
    if num_nodes == 2:
        max_time_steps = 50

    # Set embedding_dim and num_heads based on num_nodes
    embedding_dim, num_heads = {
        5: (128, 8),
        10: (128, 8),
        20: (256, 8)
    }.get(num_nodes, (128, 8))

    save_path = f"./transformer_assembly/pretrained/hybrid_RL/{save_model_cat}/hybrid_RL_{config_name}_ep_{num_episodes}.pth"
    fig_save_path = f"./transformer_assembly/img/hybrid_RL/{save_model_cat}/loss_hybrid_{config_name}_ep_{num_episodes}.png"

    # Initialize REINFORCE agent
    reinforce_agent = REINFORCE(attention_env, base_env, base_greedy_env, base_policy, embedding_dim=embedding_dim, num_heads=num_heads, lr=1e-4)

    # Train model
    reinforce_agent.train(num_episodes=num_episodes, max_time_steps=max_time_steps, save_path=save_path, load_path=load_path)
    # reinforce_agent.plot_loss_curve(fig_save_path, if_greedy=True)

    # Print training time
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    time_formatted = f"{hours}h {minutes}m {seconds}s"
    print(f"Training for {config_name} completed in {time_formatted}.")

    # Clean up resources
    reinforce_agent = None
    attention_env = None
    base_env = None
    base_greedy_env = None
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    # Configuration files
    config_file_names = [
        'config_2_p50.json', 'config_2_p100.json',
        'config_5_p50.json', 'config_5_p100.json',
        'config_10_p50.json', 'config_10_p100.json',
        'config_20_p50.json', 'config_20_p100.json'
    ]

    load_path_list = [
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_2_p50.json_ep_1000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_2_p100.json_ep_1000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_5_p50.json_ep_1000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_5_p100.json_ep_1000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_10_p50.json_ep_1000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_10_p100.json_ep_10000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_20_p50.json_ep_4000.pth",
        "./transformer_assembly/pretrained/evaluation_model/att_RL/hybrid_RL_config_20_p100.json_ep_10000.pth"
    ]

    save_model_cat = "att_greedy_RL"

    processes = []
    num_gpus = torch.cuda.device_count()

    for config_id, config_name in enumerate(config_file_names):
        load_path = load_path_list[config_id]
        gpu_id = config_id % num_gpus if num_gpus > 0 else None
        p = mp.Process(target=train_model_with_config, args=(config_name, config_id, load_path, save_model_cat, 2000, gpu_id))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("All training processes completed.")
