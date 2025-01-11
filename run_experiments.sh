#!/bin/bash

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

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

    echo "Running: $cmd"  # Отладочный вывод команды
    logfile="logs/setup_${setup_id}.log"
    echo "Log file: $logfile"  # Показываем, куда пишется лог
    
    # Запускаем команду и перенаправляем вывод в лог-файл
    $cmd > "$logfile" 2>&1 &

    # Ограничиваем количество одновременно запущенных процессов
    while [ $(jobs -r | wc -l) -ge 18 ]; do
        sleep 5
    done
done
