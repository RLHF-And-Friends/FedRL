import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def load_tensorboard_logs(log_dir, tag):
    print("Tag: ", tag, ", log_dir: ", log_dir)
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 5000000})
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag '{tag}' not found in logs at {log_dir}")
    events = event_acc.Scalars(tag)
    return np.array([event.value for event in events])

def average_logs_across_seeds(logs_dir, setup_id, exp_name, env_id, num_seeds, num_agents, tag):
    all_data = {agent: [] for agent in range(num_agents)}
    
    for seed in range(num_seeds):
        for agent in range(num_agents):
            # Ищем папку, которая начинается с паттерна и заканчивается на _agent_X
            log_dir = [
                d for d in os.listdir(logs_dir) 
                if d.startswith(f"{env_id}__{exp_name}__{setup_id}__seed_{seed}__") 
                and d.endswith(f"_agent_{agent}")
            ][0]
            full_log_dir = os.path.join(logs_dir, log_dir)
            try:
                data = load_tensorboard_logs(full_log_dir, tag)
                all_data[agent].append(data)
            except ValueError as e:
                print(f"Skipping {log_dir}: {e}")
    
    averaged_data = {}
    for agent, values_list in all_data.items():
        if values_list:  # Если есть данные для этого агента
            # Находим минимальную длину среди всех массивов
            min_length = min(len(arr) for arr in values_list)
            # Обрезаем все массивы до минимальной длины
            trimmed_values = [arr[:min_length] for arr in values_list]
            # Усредняем
            averaged_data[agent] = np.mean(trimmed_values, axis=0)
    
    return averaged_data

def save_averaged_logs(averaged_data, output_dir, tag):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for agent, values in averaged_data.items():
        df = pd.DataFrame(values, columns=[tag])
        df.to_csv(os.path.join(output_dir, f"agent_{agent}_{tag.replace('/', '_')}.csv"), index=False)

if __name__ == "__main__":
    logs_dir = "runs/setup_5_11_fixed_kl_div_weighted"  # Папка с логами
    setup_id = "setup_5_11_fixed_kl_div_weighted"
    exp_name = "GridSearch"
    env_id = "MiniGrid-CustomSimpleCrossingS9N2-v0"
    num_seeds = 8  # Количество сидов
    num_agents = 3  # Количество агентов
    tag = "charts/episodic_return"  # Тег для усреднения
    
    averaged_data = average_logs_across_seeds(logs_dir, setup_id, exp_name, env_id, num_seeds, num_agents, tag)
    save_averaged_logs(averaged_data, "averaged_logs/setup_5_11_fixed_kl_div_weighted", tag)