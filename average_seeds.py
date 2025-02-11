import os
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


def load_tensorboard_logs(log_dir, tag):
    print("Tag: ", tag, ", log_dir: ", log_dir)
    event_acc = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        raise ValueError(f"Tag '{tag}' not found in logs at {log_dir}")
    events = event_acc.Scalars(tag)
    steps = np.array([event.step for event in events])
    values = np.array([event.value for event in events])
    print("Length of events: ", len(events))

    return steps, values


def average_logs_interpolated(logs_dir, exp_name, num_seeds, num_agents, tag):
    all_data = {agent: [] for agent in range(num_agents)}
    logs_data = []  # Сохраняем данные для повторного использования

    max_step = 0  # Инициализация максимального шага

    # Загружаем данные один раз для каждого файла
    for seed in range(num_seeds):
        for agent in range(num_agents):
            log_dirs = [
                d for d in os.listdir(logs_dir)
                if d.startswith(f"{exp_name}__seed_{seed}__") and d.endswith(f"_agent_{agent}")
            ]
            if not log_dirs:
                continue

            full_log_dir = os.path.join(logs_dir, log_dirs[0])
            try:
                steps, values = load_tensorboard_logs(full_log_dir, tag)
                logs_data.append((agent, steps, values))  # Сохраняем загруженные данные
                max_step = max(max_step, steps.max())
            except ValueError as e:
                print(f"Skipping {log_dirs[0]}: {e}")

    print(f"Максимальный шаг: {max_step}")
    common_steps = np.linspace(0, max_step, max_step + 1)  # Сетка для интерполяции

    # Интерполяция значений, используя уже загруженные данные
    for agent, steps, values in logs_data:
        interpolated_values = np.interp(common_steps, steps, values)
        all_data[agent].append(interpolated_values)

    averaged_data = {}
    for agent, values_list in all_data.items():
        if values_list:
            averaged_data[agent] = np.mean(values_list, axis=0)

    return common_steps, averaged_data


def save_averaged_logs(steps, averaged_data, output_dir, tag):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for agent, values in averaged_data.items():
        df = pd.DataFrame({'step': steps, tag: values})
        df.to_csv(os.path.join(output_dir, f"agent_{agent}_{tag.replace('/', '_')}.csv"), index=False)


if __name__ == "__main__":
    logs_dir = "experiments/03_02_2025_1/setup_5_11_ppo"
    exp_name = "GridSearch"
    num_seeds = 5
    num_agents = 3
    tag = "charts/episodic_return"

    steps, averaged_data = average_logs_interpolated(logs_dir, exp_name, num_seeds, num_agents, tag)
    save_averaged_logs(steps, averaged_data, "experiments/03_02_2025_1/averaged_logs/setup_5_11_ppo", tag)
