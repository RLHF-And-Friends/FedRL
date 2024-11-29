import copy
import time


import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .utils import compute_kl_divergence


class FederatedEnvironment():
    def __init__(self, device, args, run_name, envs, agent_idx, agent, optimizer):
        self.device = device
        self.envs = envs
        self.agent_idx = agent_idx
        self.agent = agent
        self.previous_version_of_agent = copy.deepcopy(self.agent)
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
        args = self.args
        # TRY NOT TO MODIFY: start the game
        for update in range(1, args.local_updates + 1):
            # Annealing the rate if instructed to do so.
            num_updates = global_step * args.local_updates + update
            if args.anneal_lr:
                frac = 1.0 - (num_updates - 1.0) / (args.local_updates * args.global_updates)
                assert frac > 0, "fraction for learning rate annealing must be positive"
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
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                self.next_obs, self.next_done = torch.Tensor(self.next_obs).to(self.device), torch.Tensor(done).to(self.device)

                # print("info: ", info)
                if len(info) > 0:
                    # for i in range(args.num_envs):
                    for i in range(1):
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
                    advantages = torch.zeros_like(self.rewards).to(self.device)
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
                    returns = torch.zeros_like(rewards).to(self.device)
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
                    if args.use_clipping:
                        pg_loss1 = -ratio * mb_advantages
                        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    if not args.use_clipping or args.use_mdpo:
                        '''
                        For first batch pg_loss = 0 since self.previous_version_of_agent is equal to self.agent
                        '''
                        # print("mb inds shape: ", mb_inds.shape)
                        pg_loss1 = -mb_advantages * ratio
                        _, old_b_logprobs, _, _ = self.previous_version_of_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                        _, current_b_logprobs, _, _ = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                        # print("old shape: ", old_b_logprobs.shape)
                        # print("current shape: ", current_b_logprobs.shape)
                        kl_penalty = compute_kl_divergence(old_b_logprobs, current_b_logprobs)
                        pg_loss2 = args.penalty_coeff * kl_penalty
                        self.writer.add_scalar(f"charts/kl_penalty_{self.agent_idx}", kl_penalty, self.num_steps)
                        pg_loss = (pg_loss1 + pg_loss2).mean()

                    if not args.use_mdpo:
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
                    
                    if args.use_mdpo:
                        loss = pg_loss
                    else:
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    if args.use_comm_penalty:
                        kl_sum_penalty = 0
                        for neighbor_agent_idx in range(args.n_agents):
                            if neighbor_agent_idx != self.agent_idx:
                                comm_coeff = self.comm_matrix[self.agent_idx][neighbor_agent_idx]
                                current_agent = self.neighbors[self.agent_idx]
                                # print("b_obs[mb_inds] shape: ", b_obs[mb_inds].shape)
                                # print("b_actions.long()[mb_inds] shape: ", b_actions.long()[mb_inds].shape)
                                _, current_b_logprobs, _, _ = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                                neighbor_agent = self.neighbors[neighbor_agent_idx]
                                _, neighbor_b_logprobs, _, _ = neighbor_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                                kl_div = compute_kl_divergence(current_b_logprobs, neighbor_b_logprobs)
                                kl_sum_penalty += comm_coeff * kl_div
                                self.writer.add_scalar(f"charts/kl_{self.agent_idx}_{neighbor_agent_idx}", kl_div, self.num_steps)

                        loss += args.comm_penalty_coeff * kl_sum_penalty

                    self.writer.add_scalar(f"charts/loss_fractions/pg_loss", abs(pg_loss / loss), self.num_steps)
                    if args.use_comm_penalty:
                        self.writer.add_scalar(f"charts/loss_fractions/kl_penalty_loss", abs(kl_sum_penalty / loss), self.num_steps)
                                        
                    if not args.use_mdpo:
                        self.writer.add_scalar(f"charts/loss_fractions/entropy_loss", abs(entropy_loss * args.ent_coef / loss), self.num_steps)
                        self.writer.add_scalar(f"charts/loss_fractions/value_loss", abs(v_loss * args.vf_coef / loss), self.num_steps)


                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                self.previous_version_of_agent = copy.deepcopy(self.agent)

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.num_steps)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), self.num_steps)
            if not args.use_mdpo:
                self.writer.add_scalar("losses/value_loss", v_loss.item(), self.num_steps)
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
