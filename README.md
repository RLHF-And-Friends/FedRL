# FedRL

## ppo.py

Решения возможных проблем при запуске программы:

- прописать в терминале ```export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python```
- при передаче флага ```--capture-video``` придётся запатчить [монитор](/home/smirnov/FedRL/patches/site-packages/wandb/integration/gym/__init__.py)
- при запуске тензорборды инструкцией ```tensorboard --logdir runs``` придётся запатчить следующий файл: 

<img src="img/tb_patch.png" width=50%>

### Детали реализации:

Видео [здесь](https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases)

В рамках одного "policy rollout" мы собираем num_envs * num_steps точек для обучения, a.k.a. args.batch_size.

### How to run

1. В первом окне терминала запускаем процесс обучения следующей командой
    ```
    python3 federated_ppo.py --total-timesteps=500000 --n-agents=2 --local-updates=16 --num-envs=4 --exp-description="n_agents = 2, local-updates=16" --comm-coeff=10
    ```

2. Во втором окне запускаем тензорборд для визуализации результатов обучения
    ```
    tensorboard --logdir runs
    ```

    **Пример:**

    <img src="img/tb_example.png" width=50%>