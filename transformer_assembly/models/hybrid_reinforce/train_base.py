import os
import json
import time
import torch
from multiprocessing import Process, cpu_count
from transformer_assembly.env.attention_assembly_env import AssemblyTreeEnv
from transformer_assembly.models.base_stock.base_stock_policy import BaseStockPolicy
from transformer_assembly.models.hybrid_reinforce.reinforce import REINFORCE


def train_for_config(config_name, device_id):
    """
    Train a specific configuration on a designated device (GPU or CPU).
    """
    if torch.cuda.is_available() and isinstance(device_id, int):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        device = torch.device("cuda")
    else:
        device = torch.device(f"cpu:{device_id}" if isinstance(device_id, int) else "cpu")

    print(f"Process for {config_name} running on Device: {device}")

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    json_path = os.path.join(parent_dir, 'configs', 'env_configs', config_name)
    with open(json_path, 'r') as f:
        config = json.load(f)

    random_seed = 42
    attention_env = AssemblyTreeEnv(config, seed=random_seed)
    base_env = AssemblyTreeEnv(config, seed=random_seed)
    base_greedy_env = AssemblyTreeEnv(config, seed=random_seed)

    policy_config_path = os.path.join(parent_dir, 'configs', 'policy_configs', config_name)
    with open(policy_config_path, 'r') as f:
        policy_config = json.load(f)
    base_policy = BaseStockPolicy(base_env, policy_config)

    num_nodes = attention_env.num_of_nodes
    max_time_steps = 50 if num_nodes == 5 else 30 if num_nodes == 10 else 20
    if num_nodes == 2:
            max_time_steps = 50
    embedding_dim, num_heads = {
        5: (128, 8),
        10: (128, 8),
        20: (256, 8)
    }.get(num_nodes, (128, 8))
    
    num_episodes = 30

    save_model_cat = "hybrid_RL"
    save_path = f"./transformer_assembly/pretrained/{save_model_cat}/hybrid_RL_{config_name}_ep_{num_episodes}.pth"
    fig_save_path = f"./transformer_assembly/img/{save_model_cat}/loss_hybrid_{config_name}_ep_{num_episodes}.png"

    reinforce_agent = REINFORCE(attention_env, base_env, base_greedy_env, base_policy, 
                                embedding_dim=embedding_dim, num_heads=num_heads, lr=1e-4)

    start_time = time.time()
    reinforce_agent.train(num_episodes=num_episodes, max_time_steps=max_time_steps, save_path=save_path, batch_size = 8)
    # reinforce_agent.plot_loss_curve(fig_save_path, if_greedy=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Training for {config_name} completed on {device}. Elapsed time: {hours}h {minutes}m {seconds}s")


def main():
    """
    Main function to train multiple configurations in parallel using multiple devices (GPUs or CPUs).
    """
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

    available_gpus = torch.cuda.device_count()
    if available_gpus > 0:
        devices = list(range(available_gpus))  
        print(f"Available GPU Devices: {devices}")
    else:
        devices = list(range(cpu_count()))  
    

    processes = []
    for i, config_name in enumerate(config_file_names):
        device_id = devices[i % len(devices)]  
        p = Process(target=train_for_config, args=(config_name, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All configurations trained.")


if __name__ == "__main__":
    main()
