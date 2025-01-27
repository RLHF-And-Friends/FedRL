import os
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

def csv_to_tensorboard(csv_dir, log_dir):
    """
    Читает CSV-файлы и записывает данные в логи TensorBoard.
    
    :param csv_dir: Папка с CSV-файлами.
    :param log_dir: Папка для сохранения логов TensorBoard.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Проходим по всем CSV-файлам в папке
    for csv_file in tqdm(os.listdir(csv_dir)):
        if csv_file.endswith(".csv"):
            csv_path = os.path.join(csv_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # Создаем writer для TensorBoard
            writer = SummaryWriter(os.path.join(log_dir, csv_file.replace(".csv", "")))
            
            # Записываем данные
            for step, value in enumerate(df.iloc[:, 0]):  # Первый столбец — значения
                writer.add_scalar(csv_file.replace(".csv", ""), value, step)
            
            writer.close()

if __name__ == "__main__":
    csv_dir = "averaged_logs"  # Папка с CSV-файлами
    log_dir = "runs"  # Папка для логов TensorBoard
    csv_to_tensorboard(csv_dir, log_dir)
