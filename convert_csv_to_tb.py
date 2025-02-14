import os
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_csv(csv_file, csv_dir, log_dir):
    if csv_file.endswith(".csv"):
        csv_path = os.path.join(csv_dir, csv_file)
        df = pd.read_csv(csv_path)

        # Извлекаем имя агента из названия файла
        agent_name = "agent_" + csv_file.replace(".csv", "").split("_")[1]

        # Создаем отдельную папку для агента
        agent_log_dir = os.path.join(log_dir, agent_name)
        os.makedirs(agent_log_dir, exist_ok=True)

        print(f"Создание логов для агента {agent_name} в папке: {agent_log_dir}")

        # Записываем данные в папку агента
        writer = SummaryWriter(agent_log_dir)
        for step, value in zip(df['step'], df['charts/episodic_return']):
            if pd.notnull(value) and np.isfinite(value):  # Проверка на NaN и inf
                writer.add_scalar("charts/episodic_return", value, int(step))
        writer.close()

def csv_to_tensorboard(csv_dir, log_dir, max_workers=4):
    """
    Читает CSV-файлы и записывает данные в логи TensorBoard с использованием многопоточности.

    :param csv_dir: Папка с CSV-файлами.
    :param log_dir: Папка для сохранения логов TensorBoard.
    :param max_workers: Количество потоков для параллельной обработки.
    """
    os.makedirs(log_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_csv, csv_file, csv_dir, log_dir) for csv_file in csv_files]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Ошибка при обработке файла: {e}")
    except KeyboardInterrupt:
        print("\nОперация прервана пользователем. Завершаем выполнение...")

if __name__ == "__main__":
    csv_dir = "experiments/05_02_2025_2/averaged_logs/setup_ppo_baseline"  # Папка с CSV-файлами
    log_dir = "experiments/05_02_2025_2/averaged_logs/setup_ppo_baseline"  # Папка для логов TensorBoard
    csv_to_tensorboard(csv_dir, log_dir, max_workers=10)  # Используем 8 потоков
