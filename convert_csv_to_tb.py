import os
import pandas as pd
from tensorboardX import SummaryWriter

def csv_to_tensorboard(csv_dir, log_dir):
    """
    Читает CSV-файлы и записывает данные в логи TensorBoard.
    Для каждого агента создается отдельная папка.
    
    :param csv_dir: Папка с CSV-файлами.
    :param log_dir: Папка для сохранения логов TensorBoard.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Проходим по всем CSV-файлам в папке
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # Извлекаем имя агента из названия файла
            agent_name = "agent_" + csv_file.replace(".csv", "").split("_")[1]  # Например, "agent_0"
            
            # Создаем отдельную папку для агента
            agent_log_dir = os.path.join(log_dir, agent_name)
            if not os.path.exists(agent_log_dir):
                os.makedirs(agent_log_dir)

            print(f"Создание логов для агента {agent_name} в папке: {agent_log_dir}")

            # Записываем данные в папку агента
            writer = SummaryWriter(agent_log_dir)
            for step, value in enumerate(df.iloc[:, 0]):  # Первый столбец — значения
                writer.add_scalar("charts/episodic_return", value, step)
            writer.close()

if __name__ == "__main__":
    csv_dir = "averaged_logs/setup_5_11_fixed_kl_div_weighted"  # Папка с CSV-файлами
    log_dir = "runs/setup_5_11_fixed_kl_div_weighted/averaged"  # Папка для логов TensorBoard
    csv_to_tensorboard(csv_dir, log_dir)