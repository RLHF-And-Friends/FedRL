import itertools
import json

# Определение параметров сетки
grid_params = {
    "n-agents": [1],
    "num-envs": [4, 8, 16],
    "exp-name": ["GridSearch"],
    "use-clipping": [False, True],
    "use-mdpo": [False, True],
    "local-updates": [128, 512, 1024],
    "use-comm-penalty": [False],
    "average-weights":[False],
    "gym-id": ["MiniGrid-LavaCrossingS9N2-v0"],
    "learning-rate": [2.5e-4],
    "total-timesteps": [5000000],
    "anneal-lr": [True],
    "vf-coef": [0.001],
    "ent-coef": [0.01],
    "capture-video": [False],
    "clip-vloss": [False, True],
    "track": [False],
}

seeds = [0, 1, 2]  # Три разных сида

# Генерация всех комбинаций сетки
keys, values = zip(*grid_params.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Генерация команд
commands = []
setup_id = 1
for combo in combinations:
    for seed in seeds:
        cmd = (
            f"python3 -m federated_ppo.main "
            f"--setup-id=setup_{setup_id} "
            f"--n-agents={combo['n-agents']} "
            f"--num-envs={combo['num-envs']} "
            f"--exp-name={combo['exp-name']} "
            f"--use-clipping={str(combo['use-clipping']).lower()} "
            f"--use-mdpo={str(combo['use-mdpo']).lower()} "
            f"--clip-vloss={str(combo['clip-vloss']).lower()} "
            f"--use-comm-penalty={str(combo['use-comm-penalty']).lower()} "
            f"--average-weights={str(combo['average-weights']).lower()} "
            f"--anneal-lr={str(combo['anneal-lr']).lower()} "
            f"--local-updates={combo['local-updates']} "
            f"--gym-id={combo['gym-id']} "
            f"--learning-rate={combo['learning-rate']} "
            f"--total-timesteps={combo['total-timesteps']} "
            f"--seed={seed} "
            f"--vf-coef={combo['vf-coef']} "
            f"--ent-coef={combo['ent-coef']} "
            f"--capture-video={str(combo['capture-video']).lower()} "
            f"--track={str(combo['track']).lower()} "
        )
        commands.append(cmd)
    setup_id += 1

# Сохранение в файл
with open("commands.txt", "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")
