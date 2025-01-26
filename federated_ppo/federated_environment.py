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
        self.previous_version_of_agent = self._create_agent_without_gradients(agent)
        self.optimizer = optimizer
        self.comm_matrix = None
        self.neighbors = None
        self.args = args

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, device=device)
        self.actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, device=device)
        self.logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.dones = torch.zeros((args.num_steps, args.num_envs), device=device)
        self.values = torch.zeros((args.num_steps, args.num_envs), device=device)

        self.writer = SummaryWriter(f"runs/{run_name}_agent_{agent_idx}", comment=args.exp_description)
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        self.num_steps = 0
        self.start_time = time.time()
        self.next_obs = torch.tensor(
            envs.reset(seed=[args.seed + args.num_envs * self.agent_idx + i for i in range(args.num_envs)])[0],
            dtype=torch.float32,
            device=device
        )
        self.next_done = torch.zeros(args.num_envs, device=device)

        self.episodic_returns = {}

    def _create_agent_without_gradients(self, agent):
        with torch.no_grad():
            agent_copy = copy.deepcopy(agent)
            for param in agent_copy.parameters():
                param.requires_grad = False
        return agent_copy

    def set_neighbors(self, agents):
        self.neighbors = agents

    def set_comm_matrix(self, comm_matrix):
        self.comm_matrix = comm_matrix

    def local_update(self, number_of_communications):
        args = self.args
        # TRY NOT TO MODIFY: start the game
        for update in range(1, args.local_updates + 1):
            # if self.args.gym_id.startswith("MiniGrid") and self.agent_idx == 1:
            #     print(self.envs.envs[0].pprint_grid())
            # Annealing the rate if instructed to do so.
            num_updates = number_of_communications * args.local_updates + update
            if args.anneal_lr:
                frac = 1.0 - (num_updates - 1.0) / int(args.total_timesteps // args.batch_size)
                # note: denominator is not equal to (args.local_updates * args.global_updates)
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
                self.rewards[step] = torch.tensor(reward, device=self.device).view(-1)
                self.next_obs, self.next_done = torch.tensor(self.next_obs, dtype=torch.float32, device=self.device), torch.tensor(done, dtype=torch.float32, device=self.device)

                # print("info: ", info)
                if len(info) > 0:
                    for i in range(args.num_envs):
                        if info['_final_observation'][i]:
                            item = info['final_info'][i]
                            if "episode" in item.keys():
                                # print(f"global_step={self.num_steps - args.num_envs + i}, episodic_return={item['episode']['r']}")
                                self.writer.add_scalar("charts/episodic_return", item["episode"]["r"], self.num_steps - args.num_envs + i)
                                self.writer.add_scalar("charts/episodic_length", item["episode"]["l"], self.num_steps - args.num_envs + i)                                
                                if number_of_communications not in self.episodic_returns.keys():
                                    self.episodic_returns[number_of_communications] = []

                                self.episodic_returns[number_of_communications].append(item["episode"]["r"])

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(self.next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(self.rewards, device=self.device)
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
                    returns = torch.zeros_like(rewards, device=self.device)
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

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if not args.use_mdpo and args.clip_vloss:
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

                    if not args.use_mdpo:
                        entropy_loss = entropy.mean()
                    
                    if args.use_mdpo:
                        loss = pg_loss + v_loss * args.vf_coef
                        abs_loss = abs(pg_loss) + abs(v_loss * args.vf_coef) # for logging
                    else:
                        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                        abs_loss = abs(pg_loss) + abs(args.ent_coef * entropy_loss) + abs(v_loss * args.vf_coef) # for logging

                    if args.use_comm_penalty:
                        # log two options
                        sum_kl_penalty = 0
                        kl_div_weighted = None

                        _, current_b_logprobs, _, _ = self.agent.get_action_and_value(
                            b_obs[mb_inds], b_actions.long()[mb_inds]
                        )

                        sum_comm_weight = 0
                        weighted_neighbor_b_logprobs = torch.zeros_like(current_b_logprobs)

                        for neighbor_agent_idx in range(args.n_agents):
                            if neighbor_agent_idx != self.agent_idx:
                                comm_coeff = self.comm_matrix[self.agent_idx][neighbor_agent_idx]
                                if comm_coeff != 0:
                                    sum_comm_weight += comm_coeff

                                    neighbor_agent = self.neighbors[neighbor_agent_idx]
                                    _, neighbor_b_logprobs, _, _ = neighbor_agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                                    kl_div_with_neighbor = compute_kl_divergence(current_b_logprobs, neighbor_b_logprobs)
                                    self.writer.add_scalar(f"charts/kl_{self.agent_idx}_{neighbor_agent_idx}", kl_div_with_neighbor, self.num_steps)

                                    sum_kl_penalty += comm_coeff * kl_div_with_neighbor
                                    weighted_neighbor_b_logprobs += comm_coeff * neighbor_b_logprobs

                        if sum_comm_weight > 0:
                            sum_kl_penalty /= sum_comm_weight
                            weighted_neighbor_b_logprobs /= sum_comm_weight

                            kl_div_weighted = compute_kl_divergence(current_b_logprobs, weighted_neighbor_b_logprobs)
                            
                            self.writer.add_scalar(f"charts/sum_kl_{self.agent_idx}", sum_kl_penalty, self.num_steps)
                            self.writer.add_scalar(f"charts/weighted_kl_{self.agent_idx}", kl_div_weighted, self.num_steps)

                            if args.sum_kl_divergencies:
                                kl_penalty = sum_kl_penalty
                            else:
                                kl_penalty = kl_div_weighted
                        else:
                            # no communication with neighbors
                            kl_penalty = 0
                            pass

                        loss += args.comm_penalty_coeff * kl_penalty
                        abs_loss += abs(args.comm_penalty_coeff * kl_penalty) # for logging

                    self.writer.add_scalar(f"charts/loss_fractions/pg_loss", abs(pg_loss / abs_loss), self.num_steps)
                    if args.use_comm_penalty:
                        self.writer.add_scalar(f"charts/loss_fractions/kl_penalty_loss", abs(args.comm_penalty_coeff * kl_penalty / abs_loss), self.num_steps)
                                        
                    if not args.use_mdpo:
                        self.writer.add_scalar(f"charts/loss_fractions/entropy_loss", abs(entropy_loss * args.ent_coef / abs_loss), self.num_steps)
                        self.writer.add_scalar(f"charts/loss_fractions/value_loss", abs(v_loss * args.vf_coef / abs_loss), self.num_steps)


                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    self.optimizer.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            del self.previous_version_of_agent
            self.previous_version_of_agent = self._create_agent_without_gradients(self.agent)

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
            # print("SPS:", int(self.num_steps / (time.time() - self.start_time)))
            self.writer.add_scalar("charts/SPS", int(self.num_steps / (time.time() - self.start_time)), self.num_steps)

        average_episodic_return_between_communications = 0
        if number_of_communications in self.episodic_returns.keys():
            average_episodic_return_between_communications = np.mean(self.episodic_returns[number_of_communications])
            self.writer.add_scalar("charts/average_episodic_return_between_communications", average_episodic_return_between_communications, number_of_communications)

        print("agent_idx: ", self.agent_idx, ", average_episodic_return_between_communications: ", average_episodic_return_between_communications)

    def close(self):
        self.envs.close()
        self.writer.close()
