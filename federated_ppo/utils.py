import os
import argparse
import numpy as np
import torch
import json
from distutils.util import strtobool
import gym
import torch.nn.functional as F
from custom_envs.classic_control.cartpole import CustomCartPoleEnv
from custom_envs.custom_minigrid.simple_env import SimpleEnv
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper



def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="",
        help="the name of this experiment")
    parser.add_argument("--setup-id", type=str, default="",
        help="the id of the setup")
    parser.add_argument("--exp-description", type=str, default="Empty description",
        help="Experiment description")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
    parser.add_argument("--use-custom-env", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="use custom environment or not")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--n-agents", type=int, default=2,
        help="number of agents")
    parser.add_argument("--comm-matrix-config", type=str, default=None, help="path to comm_matrix json-config")
    parser.add_argument("--env-parameters-config", type=str, default=None, help="path to cartpole environment json-config")
    parser.add_argument("--local-updates", type=int, default=16,
        help="parameter E from chinese article")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="T oggles advantages normalization")
    parser.add_argument("--use-mdpo", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
        help="Use MDPO instead of PPO")
    parser.add_argument("--use-clipping", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use clipping or KL penalty (adaptive loss)")
    parser.add_argument("--use-comm-penalty", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Penalize for kl divergence with neighbors or not") 
    parser.add_argument("--comm-penalty-coeff",  type=float, default=1.0,
        help="coefficient of the communication penalty") 
    parser.add_argument("--penalty-coeff", type=float, default=1.0,
        help="KL penalty coefficient")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.global_updates = int(int(args.total_timesteps // args.batch_size) // args.local_updates)
    # fmt: on
    return args


def create_comm_matrix(n_agents, comm_matrix_config):
    W = np.zeros((n_agents, n_agents))

    with open(comm_matrix_config, 'r') as file:
        data = json.load(file)
        for left, coeffs in data["comm_matrix"].items():
            for right, coef in coeffs.items():
                left_idx = int(left)
                right_idx = int(right)
                W[left_idx - 1][right_idx - 1] = W[right_idx - 1][left_idx - 1] = coef

    return torch.tensor(W, dtype=torch.float32)


def extract_env_parameters(env_parameters_config, agent_idx):
    with open(env_parameters_config, 'r') as file:
        data = json.load(file)
        
        return data["cartpole_parameters"][str(agent_idx + 1)] 


def compute_kl_divergence(p, q):
    p_log = F.log_softmax(p, dim=-1)
    q_log = F.log_softmax(q, dim=-1)
    kl_div = (p_log.exp() * (p_log - q_log)).sum()

    return kl_div


def make_env(args, env_parameters_config, gym_id, seed, idx, capture_video, run_name=None):
    def thunk():
        import minigrid

        if args.use_custom_env:
            if env_parameters_config is not None:
                env_parameters = extract_env_parameters(env_parameters_config, idx)
                env = CustomCartPoleEnv(render_mode="rgb_array", **env_parameters)
            else:
                env = SimpleEnv(render_mode="rgb_array", size=6)
                obs1 = env.reset() # obs: {'image': numpy.ndarray (7, 7, 3),'direction': ,'mission': ,}
                env = RGBImgPartialObsWrapper(env) # Get pixel observations
                obs2 = env.reset() # obs: {'mission': ,'image': numpy.ndarray (56, 56, 3)}
                env = ImgObsWrapper(env) # Get rid of the 'mission' field
                obs3 = env.reset() # obs: numpy.ndarray (56, 56, 3)
        else:
            env = gym.make(gym_id, render_mode="rgb_array")
            if gym_id.startswith("MiniGrid"):
                env = ImgObsWrapper(env)

        # print("utils: obs space before sample: ", env.observation_space)
        # print("utils: obs space before sample shape: ", env.observation_space.shape)
        # print("utils: obs space after sample: ", env.observation_space.sample())

        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/env_{idx}/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
