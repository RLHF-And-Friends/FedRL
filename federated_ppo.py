import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import concurrent.futures


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-description", type=str, default="Empty description",
        help="Experiment description")
    parser.add_argument("--gym-id", type=str, default="CartPole-v1",
        help="the id of the gym environment")
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
    parser.add_argument("--local-updates", type=int, default=16,
        help="parameter E from chinese article")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
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


def create_comm_matrix(n_agents):
    W = np.zeros((n_agents, n_agents))
    for i in range(n_agents - 1):
        W[i, i + 1] = W[i + 1, i] = 1
    return torch.tensor(W, dtype=torch.float32)


def compute_kl(agent1, agent2):
    # Рассчитываем KL-дивергенцию между двумя агентами
    # Это упрощенный расчет; в реальной задаче нужно будет реализовать точный расчет KL для распределений
    params1 = torch.cat([p.view(-1) for p in agent1.parameters()])
    params2 = torch.cat([p.view(-1) for p in agent2.parameters()])
    return torch.sum((params1 - params2) ** 2)



def make_env(gym_id, seed, idx, capture_video, run_name=None):
    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0:
        #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class FederatedEnvironment():
    def __init__(self, args, run_name, envs, agent_idx, agent, optimizer):
        self.envs = envs
        self.agent_idx = agent_idx
        self.agent = agent
        self.optimizer = optimizer
        self.comm_matrix = None
        self.neighbors = None
        self.args = args

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.writer = SummaryWriter(f"runs/{run_name}_agent_{agent_idx}", comment=args.exp_description)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.num_steps = 0
        self.start_time = time.time()
        self.next_obs = torch.Tensor(envs.reset(seed=[args.seed + args.num_envs * self.agent_idx + i for i in range(args.num_envs)])[0]).to(device)
        self.next_done = torch.zeros(args.num_envs).to(device)

    def set_neighbors(self, agents):
        self.neighbors = agents

    def set_comm_matrix(self, comm_matrix):
        self.comm_matrix = comm_matrix

    def local_update(self, global_step):
        # TRY NOT TO MODIFY: start the game
        for update in range(1, args.local_updates + 1):
            # Annealing the rate if instructed to do so.
            num_updates = global_step * args.local_updates + update
            if args.anneal_lr:
                frac = 1.0 - (num_updates - 1.0) / (args.local_updates * args.global_updates)
                lrnow = frac * args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                self.num_steps += 1 * args.num_envs
                self.obs[step] = self.next_obs
                self.dones[step] = self.next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(self.next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                # print(envs.step(action.cpu().numpy()))
                self.next_obs, reward, done, truncations, info = self.envs.step(action.cpu().numpy())
                # print(torch.tensor(reward).to(device))
                self.rewards[step] = torch.tensor(reward).to(device).view(-1)
                self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(device), torch.Tensor(done).to(device)

                # print("info: ", info)
                if len(info) > 0:
                    for i in range(args.num_envs):
                        if info['_final_observation'][i]:
                            item = info['final_info'][i]
                            if "episode" in item.keys():
                                print(f"global_step={self.num_steps}, episodic_return={item['episode']['r']}")
                                self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.num_steps)
                                self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.num_steps)
                                break

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(self.rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - self.next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            nextvalues = self.values[t + 1]
                        delta = self.rewards[t] + args.gamma * nextvalues * nextnonterminal - self.values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + self.values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - self.dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.num_steps)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), self.num_steps)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.num_steps)
            self.writer.add_scalar("losses/entropy", entropy_loss.item(), self.num_steps)
            self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), self.num_steps)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), self.num_steps)
            self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), self.num_steps)
            self.writer.add_scalar("losses/explained_variance", explained_var, self.num_steps)
            print("SPS:", int(self.num_steps / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(self.num_steps / (time.time() - self.start_time)), self.num_steps)

    def close(self):
        self.envs.close()
        self.writer.close()


def exchange_weights(federated_envs) -> None:
    pass
    # agents = []
    # for env in federated_envs:
    #     agents.append(self.agent.deepcopy())

    # for env in federated_envs:
    #     env.set_neighbors(agents)


def generate_federated_system(args, run_name):
    # env setup
    federated_envs = []

    for agent_idx in range(args.n_agents):
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.gym_id, args.seed + i, i, args.capture_video) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)  

        federated_envs.append(FederatedEnvironment(args, run_name, envs, agent_idx, agent, optimizer))

    comm_matrix = create_comm_matrix(n_agents=args.n_agents)
    
    for env in federated_envs:
        env.set_comm_matrix(comm_matrix)
    
    exchange_weights(federated_envs)

    return federated_envs


def local_update(federated_env, global_step) -> None:
    federated_env.local_update(global_step)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    federated_envs = generate_federated_system(args, run_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.n_agents) as executor:
        for global_step in range(1, args.global_updates + 1):
            futures = []
            for i in range(args.n_agents):
                futures.append(executor.submit(local_update, federated_envs[i], global_step))

            for future in futures:
                future.result()

            exchange_weights(federated_envs)

    for env in federated_envs:
        env.close()
