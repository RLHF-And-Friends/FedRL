# FedRL

## ppo.py

Решения возможных проблем при запуске программы:

- прописать в терминале ```export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python```
- при передаче флага ```--capture-video``` придётся запатчить [монитор](/home/smirnov/FedRL/patches/site-packages/wandb/integration/gym/__init__.py)
- при запуске тензорборды инструкцией ```tensorboard --logdir runs``` придётся запатчить следующий файл: 

<img src="img/tb_patch.png" width=40%>

### Детали реализации:

Видео [здесь](https://www.youtube.com/watch?v=MEt6rrxH8W4&ab_channel=Weights%26Biases)

Соответствующая [статья](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)


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

    <img src="img/tb_example.png" width=40%>


### Логирование статистик

Интересно было посмотреть на вклад каждого из слагаемых в итоговый лосс, на который обучаются агенты. Получилось, что для калибровки этих слагаемых до примерно одного порядка нужно задать параметр *vf-coef = 0.001* вместо 0.5 по умолчанию.

<img src="img/loss_fractions.png" width="40%">

Характер изменения перфоманса в процессе обучения также изменился:

1. До масштабирования слагаемых в лоссе:
    ```
    python3 federated_ppo.py --total-timesteps=1000000 --n-agents=4 --local-updates=16 --num-envs=4 --comm-matrix-config="comm_matrices/4_agents.json" --use-clipping=True
    ```

    <img src="img/perfomance_before_loss_scaling.png" width="40%">

2. После масштабирования слагаемых:
    ```
    python3 federated_ppo.py --total-timesteps=1000000 --n-agents=4 --local-updates=16 --num-envs=4 --comm-matrix-config="comm_matrices/4_agents.json" --use-clipping=True --vf-coef=0.001
    ```

    <img src="img/perfomance_after_loss_scaling.png" width="40%">

Таким образом, с правильно подобранными коэффициентами для каждого из слагаемых в лоссе мы получаем лучшие результаты. Объединённые выше графики:

<img src="img/before_and_after_loss_scaling.png" width="40%">

Причём у конфигурации с масштабированием даже без сглаживания график награды за эпизод проходит почти по нижней границе графика, когда сглаживание есть, в отличие от второго сетапа, у которого соответствующий график слишком шумный.

Полученные графики соответствуют по своему поведению и масштабу тем, что представлены в исходной статье по имплементации PPO: [classic control experiments](https://wandb.ai/vwxyzjn/ppo-details/reports/Matching-the-metrics--VmlldzoxMzY5OTMy).