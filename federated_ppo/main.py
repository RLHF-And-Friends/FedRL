import random
import copy
import time

import gym
import minigrid
import numpy as np
import torch
import concurrent.futures
import torch.optim as optim

from .federated_environment import FederatedEnvironment
from .agent import Agent

from .utils import (
    parse_args,
    create_comm_matrix,
    compute_kl_divergence,
    make_env,
)


def average_weights(federated_envs) -> None:
    agents = []
    for env in federated_envs:
        agents.append(copy.deepcopy(env.agent))

    state_dict_keys = agents[0].state_dict().keys()
    # print("State dict keys: ", state_dict_keys)

    for i, env in enumerate(federated_envs):
        agent = env.agent
        averaged_weights = {key: torch.zeros_like(param) for key, param in agents[0].state_dict().items()}
        for key in state_dict_keys:
            denom = 0
            for j, neighbor_agent in enumerate(agents):
                neighbor_agent_weights = neighbor_agent.state_dict()
                averaged_weights[key] += env.comm_matrix[i, j] * neighbor_agent_weights[key]
                denom += env.comm_matrix[i, j]
            averaged_weights[key] /= denom

        agent.load_state_dict(averaged_weights)
    
        print("agent: ", i, ", averaged weights: ", averaged_weights)

def exchange_weights(federated_envs) -> None:
    agents = []
    for env in federated_envs:
        agents.append(copy.deepcopy(env.agent))

    for env in federated_envs:
        env.set_neighbors(agents)


def generate_federated_system(device, args, run_name):
    # env setup
    federated_envs = []

    for agent_idx in range(args.n_agents):
        envs = gym.vector.SyncVectorEnv(
            [make_env(args, args.env_parameters_config, args.gym_id, args.seed + i, i, args.capture_video, run_name=run_name) for i in range(args.num_envs)]
        )
        # print("generate_federated_system: Envs are created")
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs, args, is_grid=args.gym_id.startswith("MiniGrid")).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  

        federated_envs.append(FederatedEnvironment(device, args, run_name, envs, agent_idx, agent, optimizer))

    if args.use_comm_penalty or args.average_weights:
        comm_matrix = create_comm_matrix(n_agents=args.n_agents, comm_matrix_config=args.comm_matrix_config)

        for env in federated_envs:
            env.set_comm_matrix(comm_matrix)
    
    exchange_weights(federated_envs)

    return federated_envs


def local_update(federated_env, global_step) -> None:
    federated_env.local_update(global_step)


if __name__ == "__main__":
    args = parse_args()
    run_name = args.gym_id
    if args.exp_name != "":
        run_name += f"__{args.exp_name}"
        if args.setup_id != "":
            run_name += f"__{args.setup_id}"
        if args.seed != "":
            run_name += f"__seed_{args.seed}"
    run_name += f"__{int(time.time())}"

    # if args.track:
    #     import wandb

    #     wandb.init(
    #         project=args.wandb_project_name,
    #         entity=args.wandb_entity,
    #         sync_tensorboard=True,
    #         config=vars(args),
    #         name=run_name,
    #         monitor_gym=True,
    #         save_code=True,
    #     )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    federated_envs = generate_federated_system(device, args, run_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_agents) as executor:
        for global_step in range(0, args.global_updates):
            print("GLOBAL_STEP: ", global_step)
            futures = []
            for i in range(args.n_agents):
                futures.append(executor.submit(local_update, federated_envs[i], global_step))

            for future in futures:
                future.result()

            if args.average_weights:
                average_weights(federated_envs)

            exchange_weights(federated_envs)

    for env in federated_envs:
        env.close()
