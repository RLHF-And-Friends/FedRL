#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPUS=(0 1)
MAX_JOBS=4

mkdir -p logs

# Запускаем все команды из файла commands.txt
cat commands.txt | while read -r cmd; do
    if [ -z "$cmd" ]; then
        continue  # Пропускаем пустые строки
    fi
    
    setup_id=$(echo "$cmd" | awk -F'--setup-id=setup_' '{print $2}' | awk '{print $1}')

    if [ -z "$setup_id" ]; then
        echo "Error: Could not extract setup-id from command: $cmd"
        continue
    fi

    GPU="${GPUS[$gpu_index]}"
    echo "Выбран GPU = $GPU для команды: $cmd"

    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))

    echo "Running: $cmd"  # Отладочный вывод команды
    logfile="logs/setup_${setup_id}.log"
    echo "Log file: $logfile"  # Показываем, куда пишется лог

    CUDA_VISIBLE_DEVICES="$GPU" \
    $cmd > "$logfile" 2>&1 &
    # sg mygroup -c "$cmd > \"$logfile\" 2>&1 &"

    # Ограничиваем количество одновременно запущенных процессов
    while [ $(jobs -r | wc -l) -ge "$MAX_JOBS" ]; do
        sleep 5
    done
done
