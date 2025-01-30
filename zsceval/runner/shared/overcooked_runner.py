import time
from collections import defaultdict

import numpy as np
import torch
import wandb
from icecream import ic
from loguru import logger

from zsceval.runner.separated.base_runner import Runner
from zsceval.utils.log_util import eta


def _t2n(x):
    return x.detach().cpu().numpy()


class OvercookedRunner(Runner):
    def __init__(self, config):
        super(OvercookedRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0

        for episode in range(episodes):
            time.time()
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)

                # Obser reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                obs = np.stack(obs)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor([total_num_steps] * self.n_rollout_threads)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)
            # e_time = time.time()
            # logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            # s_time = time.time()
            self.compute()
            train_infos = self.train(total_num_steps)
            # e_time = time.time()
            # logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            # post process
            # s_time = time.time()
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    self.save(total_num_steps)
                    # self.save(episode)
            elif episode < 100:
                if episode % 5 == 0:
                    self.save(total_num_steps)
                    # self.save(episode)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.save(total_num_steps)
                    # self.save(episode)

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )

                for a in range(self.num_agents):
                    train_infos[a]["average_episode_rewards"] = np.mean(self.buffer[a].rewards) * self.episode_length
                    logger.info(
                        "agent {} average episode rewards is {}".format(a, train_infos[a]["average_episode_rewards"])
                    )

                env_infos = defaultdict(list)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
                if self.env_name == "Overcooked":
                    if self.all_args.overcooked_version == "old":
                        from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )

                        shaped_info_keys = SHAPED_INFOS
                    else:
                        from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                            SHAPED_INFOS,
                        )

                        shaped_info_keys = SHAPED_INFOS

                    for info in infos:
                        for a in range(self.num_agents):
                            env_infos[f"ep_sparse_r_by_agent{a}"].append(info["episode"]["ep_sparse_r_by_agent"][a])
                            env_infos[f"ep_shaped_r_by_agent{a}"].append(info["episode"]["ep_shaped_r_by_agent"][a])
                            env_infos[f"ep_utility_r_by_agent{a}"].append(info["episode"]["ep_utility_r_by_agent"][a])
                            env_infos[f"ep_hidden_r_by_agent{a}"].append(info["episode"]["ep_hidden_r_by_agent"][a])
                            for i, k in enumerate(shaped_info_keys):
                                env_infos[f"ep_{k}_by_agent{a}"].append(info["episode"]["ep_category_r_by_agent"][a][i])
                        env_infos["ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        env_infos["ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                        env_infos["ep_utility_r"].append(info["episode"]["ep_utility_r"])
                        env_infos["ep_hidden_r"].append(info["episode"]["ep_hidden_r"])

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                logger.info(f'average sparse rewards is {np.mean(env_infos["ep_sparse_r"]):.3f}')

            # eval
            if episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                self.eval(total_num_steps)
            # e_time = time.time()
            # logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        obs = np.stack(obs)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
                self.buffer[agent_id].available_actions[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)

            actions.append(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        bad_masks = np.array([[[0.0] if info["bad_transition"] else [1.0]] * self.num_agents for info in infos])

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = obs

            self.buffer[agent_id].insert(
                share_obs[:, agent_id],
                obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                bad_masks=bad_masks[:, agent_id],
                available_actions=available_actions[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_env_infos = defaultdict(list)
        if self.env_name == "Overcooked":
            if self.all_args.overcooked_version == "old":
                from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
            else:
                from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                    SHAPED_INFOS,
                )

                shaped_info_keys = SHAPED_INFOS
        eval_episode_rewards = []
        eval_obs, _, eval_available_actions = self.eval_envs.reset()
        eval_obs = np.stack(eval_obs)

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for _ in range(self.episode_length):
            eval_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id],
                    deterministic=not self.all_args.eval_stochastic,
                )

                eval_action = _t2n(eval_action)
                eval_actions.append(eval_action)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            eval_actions = np.stack(eval_actions).transpose(1, 0, 2)
            # logger.debug(f"eval_actions {eval_actions.shape}")
            # Obser reward and next obs
            (
                eval_obs,
                _,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_obs = np.stack(eval_obs)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        for eval_info in eval_infos:
            for a in range(self.num_agents):
                eval_env_infos[f"eval_ep_sparse_r_by_agent{a}"].append(eval_info["episode"]["ep_sparse_r_by_agent"][a])
                eval_env_infos[f"eval_ep_shaped_r_by_agent{a}"].append(eval_info["episode"]["ep_shaped_r_by_agent"][a])
                eval_env_infos[f"eval_ep_utility_r_by_agent{a}"].append(eval_info["episode"]["ep_utility_r_by_agent"][a])
                eval_env_infos[f"eval_ep_hidden_r_by_agent{a}"].append(eval_info["episode"]["ep_hidden_r_by_agent"][a])
                for i, k in enumerate(shaped_info_keys):
                    eval_env_infos[f"eval_ep_{k}_by_agent{a}"].append(
                        eval_info["episode"]["ep_category_r_by_agent"][a][i]
                    )
            eval_env_infos["eval_ep_sparse_r"].append(eval_info["episode"]["ep_sparse_r"])
            eval_env_infos["eval_ep_shaped_r"].append(eval_info["episode"]["ep_shaped_r"])
            eval_env_infos["eval_ep_utility_r"].append(eval_info["episode"]["ep_utility_r"])
            eval_env_infos["eval_ep_hidden_r"].append(eval_info["episode"]["ep_hidden_r"])

        eval_env_infos["eval_average_episode_rewards"] = np.sum(eval_episode_rewards, axis=0)
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_sparse_r"]):.3f} {len(eval_env_infos["eval_ep_sparse_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )
        logger.success(
            f'eval average sparse rewards {np.mean(eval_env_infos["eval_ep_shaped_r"]):.3f} {len(eval_env_infos["eval_ep_shaped_r"])} episodes, total num timesteps {total_num_steps}/{self.num_env_steps}'
        )

        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        envs = self.envs
        obs, share_obs, available_actions = envs.reset()
        obs = np.stack(obs)

        for episode in range(self.all_args.render_episodes):
            episode_rewards = []

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                time.time()
                actions = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(np.array(obs)[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(obs)[:, agent_id],
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    actions.append(action[0])
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # Obser reward and next obs
                print("action:", actions)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step([actions])
                obs = np.stack(obs)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

            for info in infos:
                for a in range(self.num_agents):
                    ic(info["episode"]["ep_sparse_r_by_agent"][a])
                    ic(info["episode"]["ep_shaped_r_by_agent"][a])
                ic(info["episode"]["ep_sparse_r"])
                ic(info["episode"]["ep_shaped_r"])

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))
            # print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

    def save(self, step, save_critic: bool = False):
        # logger.info(f"save hsp periodic_{step}.pt")
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(
                    policy_model.state_dict(),
                    str(self.save_dir) + f"/model_agent{agent_id}_periodic_{step}.pt",
                )
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(
                    policy_actor.state_dict(),
                    str(self.save_dir) + f"/actor_agent{agent_id}_periodic_{step}.pt",
                )
                if save_critic:
                    policy_critic = self.trainer[agent_id].policy.critic
                    torch.save(
                        policy_critic.state_dict(),
                        str(self.save_dir) + f"/critic_agent{agent_id}_periodic_{step}.pt",
                    )

<<<<<<< HEAD
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)
            
            # eval
            if episode > 0 and episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                if self.all_args.use_opponent_utility:
                    eval_info = self.evaluate_opp_utility_policy_with_multi_policy()
                else:
                    eval_info = self.evaluate_with_multi_policy()
                # logger.debug("eval_info: {}".format(pprint.pformat(eval_info)))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)
            
            e_time = time.time()
            logger.trace(f"Post update models time: {e_time - s_time:.3f}s")


    def train_fcp(self):
        raise NotImplementedError

    def train_mep(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(f"population_size: {self.all_args.population_size}, {self.population}")

        if self.all_args.stage == 1:
            # Stage 1: train a maximum entropy population
            if self.use_eval:
                assert self.n_eval_rollout_threads % self.population_size == 0
                self.all_args.eval_episodes *= self.population_size
                map_ea2p = {
                    (e, a): self.population[e % self.population_size]
                    for e in range(self.n_eval_rollout_threads)
                    for a in range(self.num_agents)
                }
                self.policy.set_map_ea2p(map_ea2p)

            def pbt_reset_map_ea2t_fn(episode):
                # Round robin trainer
                trainer_name = self.population[episode % self.population_size]
                map_ea2t = {(e, a): trainer_name for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
                return map_ea2t

            # MARK: *self.population_size
            self.num_env_steps *= self.population_size
            self.save_interval *= self.population_size
            self.log_interval *= self.population_size
            self.eval_interval *= self.population_size

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=pbt_reset_map_ea2t_fn)

            if self.use_eval:
                self.all_args.eval_episodes /= self.population_size
            self.num_env_steps /= self.population_size
            self.save_interval /= self.population_size
            self.log_interval /= self.population_size
            self.eval_interval /= self.population_size
        else:
            # Stage 2: train an agent against population with prioritized sampling
            agent_name = self.trainer.agent_name
            assert self.use_eval
            assert (
                self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
                and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
            )
            assert self.n_rollout_threads % self.all_args.train_env_batch == 0
            self.all_args.eval_episodes = (
                self.all_args.eval_episodes * self.population_size // self.all_args.eval_env_batch
            )
            self.eval_idx = 0

            if self.all_args.fixed_index == None:
                all_agent_pairs = list(itertools.product(self.population, [agent_name]))\
                + list(itertools.product([agent_name], self.population))
                logger.info(f"all agent pairs: {all_agent_pairs}")

                population_size = self.population_size * 2
            else:
                if self.all_args.fixed_index == 1:
                    all_agent_pairs = list(itertools.product(self.population, [agent_name]))
                else:
                    all_agent_pairs = list(itertools.product([agent_name], self.population))
                population_size = self.population_size

            running_avg_r = -np.ones((population_size,), dtype=np.float32) * 1e9

            def mep_reset_map_ea2t_fn(episode):
                # Randomly select agents from population to be trained
                # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
                if self.all_args.fixed_index == None:
                    sampling_prob_np = np.ones((population_size,)) / self.population_size / 2
                else:
                    sampling_prob_np = np.ones((population_size,)) / self.population_size

                if self.all_args.use_advantage_prioritized_sampling: # Default:False
                    # logger.debug("use advantage prioritized sampling")
                    if episode > 0:
                        metric_np = np.array([self.avg_adv[agent_pair] for agent_pair in all_agent_pairs])
                        # TODO: retry this
                        sampling_rank_np = rankdata(metric_np, method="dense")
                        sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                        sampling_prob_np /= sampling_prob_np.sum()
                        maxv = 1.0 / (population_size) * 10
                        while sampling_prob_np.max() > maxv + 1e-6:
                            sampling_prob_np = sampling_prob_np.clip(max=maxv)
                            sampling_prob_np /= sampling_prob_np.sum()
                elif self.all_args.mep_use_prioritized_sampling:   # Default:False
                    metric_np = np.zeros((population_size,))
                    for i, agent_pair in enumerate(all_agent_pairs):
                        if self.all_args.use_primitive_hsp:
                            train_r = np.mean(self.env_info.get(f"{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r", -1e9))
                        else:
                            train_r = np.mean(self.env_info.get(f"{agent_pair[0]}-{agent_pair[1]}-ep_shaped_r", -1e9))
                        eval_r = np.mean(
                            self.eval_info.get(
                                f"{agent_pair[0]}-{agent_pair[1]}-eval_ep_shaped_r",
                                -1e9,
                            )
                        )

                        avg_r = 0.0
                        cnt_r = 0
                        if train_r > -1e9:
                            avg_r += train_r * (self.n_rollout_threads // self.all_args.train_env_batch)
                            cnt_r += self.n_rollout_threads // self.all_args.train_env_batch
                        if eval_r > -1e9:
                            avg_r += eval_r * (
                                self.all_args.eval_episodes
                                // (self.n_eval_rollout_threads // self.all_args.eval_env_batch)
                            )
                            cnt_r += self.all_args.eval_episodes // (
                                self.n_eval_rollout_threads // self.all_args.eval_env_batch
                            )
                        if cnt_r > 0:
                            avg_r /= cnt_r
                        else:
                            avg_r = -1e9
                        if running_avg_r[i] == -1e9:
                            running_avg_r[i] = avg_r
                        else:
                            # running average
                            running_avg_r[i] = running_avg_r[i] * 0.95 + avg_r * 0.05
                        metric_np[i] = running_avg_r[i]
                    running_avg_r_dict = {}
                    for i, agent_pair in enumerate(all_agent_pairs):
                        running_avg_r_dict[f"running_average_return/{agent_pair[0]}-{agent_pair[1]}"] = np.mean(
                            running_avg_r[i]
                        )
                    if self.use_wandb:
                        for k, v in running_avg_r_dict.items():
                            if v > -1e9:
                                wandb.log({k: v}, step=self.total_num_steps)
                    running_avg_r_dict = {
                        f"running_average_return/{agent_pair[0]}-{agent_pair[1]}": f"{running_avg_r[i]:.3f}"
                        for i, agent_pair in enumerate(all_agent_pairs)
                    }
                    logger.trace(f"running avg_r\n{pprint.pformat(running_avg_r_dict, width=600, compact=True)}")
                    if (metric_np > -1e9).astype(np.int32).sum() > 0:
                        avg_metric = metric_np[metric_np > -1e9].mean()
                    else:
                        # uniform
                        avg_metric = 1.0
                    metric_np[metric_np == -1e9] = avg_metric

                    # reversed return
                    sampling_rank_np = rankdata(1.0 / (metric_np + 1e-6), method="dense")
                    sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                    sampling_prob_np = sampling_prob_np**self.all_args.mep_prioritized_alpha
                    sampling_prob_np /= sampling_prob_np.sum()
                assert abs(sampling_prob_np.sum() - 1) < 1e-3

                # log sampling prob
                sampling_prob_dict = {}
                for i, agent_pair in enumerate(all_agent_pairs):
                    sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = sampling_prob_np[i]
                if self.use_wandb:
                    wandb.log(sampling_prob_dict, step=self.total_num_steps)

                n_selected = self.n_rollout_threads // self.all_args.train_env_batch
                pair_idx = np.random.choice(population_size, size=(n_selected,), p=sampling_prob_np)

                if self.all_args.uniform_sampling_repeat > 0:  # Default:0
                    assert n_selected >= population_size * self.all_args.uniform_sampling_repeat
                    i = 0
                    for r in range(self.all_args.uniform_sampling_repeat):
                        for x in range(population_size):
                            pair_idx[i] = x
                            i += 1
                map_ea2t = {
                    (e, a): all_agent_pairs[pair_idx[e % n_selected]][a]
                    for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))
                }

                return map_ea2t

            def mep_reset_map_ea2p_fn(episode):
                if self.all_args.eval_policy != "":
                    map_ea2p = {
                        (e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2]
                        for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                    }
                else:
                    map_ea2p = {
                        (e, a): all_agent_pairs[
                            (self.eval_idx + e // self.all_args.eval_env_batch) % (population_size)
                        ][a]
                        for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                    }
                    self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                    self.eval_idx %= population_size
                featurize_type = [
                    [self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]
                    for e in range(self.n_eval_rollout_threads)
                ]
                self.eval_envs.reset_featurize_type(featurize_type)
                return map_ea2p

            self.naive_train_with_multi_policy(
                reset_map_ea2t_fn=mep_reset_map_ea2t_fn,
                reset_map_ea2p_fn=mep_reset_map_ea2p_fn,
            )

    def train_traj(self):
        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(f"population_size: {self.all_args.population_size}, {self.population}")

        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        assert self.all_args.stage == 1
        if self.all_args.stage == 1:
            if self.use_eval:
                assert self.n_eval_rollout_threads % self.population_size == 0
                self.all_args.eval_episodes *= self.population_size
                map_ea2p = {
                    (e, a): self.population[e % self.population_size]
                    for e in range(self.n_eval_rollout_threads)
                    for a in range(self.num_agents)
                }
                self.policy.set_map_ea2p(map_ea2p)

            def pbt_reset_map_ea2t_fn(episode):
                # Round robin trainer
                map_ea2t = {
                    (e, a): self.population[(e + episode * self.n_rollout_threads) % self.population_size]
                    for e in range(self.n_rollout_threads)
                    for a in range(self.num_agents)
                }
                return map_ea2t

            # MARK: *self.population_size
            self.num_env_steps *= self.population_size
            self.save_interval *= self.population_size
            self.log_interval *= self.population_size
            self.eval_interval *= self.population_size

            self.naive_train_with_multi_policy(reset_map_ea2t_fn=pbt_reset_map_ea2t_fn)

            if self.use_eval:
                self.all_args.eval_episodes /= self.population_size
            self.num_env_steps /= self.population_size
            self.save_interval /= self.population_size
            self.log_interval /= self.population_size
            self.eval_interval /= self.population_size

    def train_cole(self):
        
        assert self.all_args.stage == 2
        assert self.use_eval
        assert (
            self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
            and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
        )
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args._eval_episodes = self.all_args.eval_episodes

        """
           p1 p2 p3 ...
        p1
        p2
        p3
        ...
        agent_name
        """
        self.u_matrix = defaultdict(dict)
        self.generation_interval = self.all_args.generation_interval
        self.num_generation = self.all_args.num_generation
        self.population_play_ratio = self.all_args.population_play_ratio
        assert self.all_args.population_size == len(self.trainer.population)
        self.max_population_size = self.all_args.population_size
        self.population = list(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents
        logger.info(f"total population {self.population}")
        # self.population_size = self.population_play_ratio
        self.generated_population_names = self.trainer.generated_population_names
        self.population_size = len(self.generated_population_names)
        self.generation = self.population_size
        logger.info(f"population {self.generated_population_names}")

        self.eval_idx = 0
        self.n_generation_try = 0

        # self.all_args.eval_episodes = (
        #     self.all_args.eval_episodes
        #     * self.population_size
        #     // self.all_args.eval_env_batch
        # )

        agent_name = self.trainer.agent_name

        # init u_matrix
        for p in self.generated_population_names:
            for o_p in self.generated_population_names:
                self.u_matrix[p][o_p] = 0.0
            self.u_matrix[agent_name][p] = 0.0

        def cole_reset_map_ea2t_fn(episode):
            if episode > 0:
                # update u_matrix
                for p in self.generated_population_names:
                    eval_r = []
                    for log_name in [f"{p}-{agent_name}", f"{agent_name}-{p}"]:
                        if f"{log_name}-ep_shaped_r" in self.env_info:
                            eval_r.append(np.mean(self.env_info[f"{log_name}-ep_shaped_r"]))
                    if len(eval_r) > 0:
                        self.u_matrix[agent_name][p] = (self.u_matrix[agent_name][p] + np.mean(eval_r)) / 2

            if episode > self.generation_interval and episode % self.generation_interval == 1:
                # generate a new partner
                model_path = self.trainer.save_actor(agent_name, self.generation + 1)
                self.generation += 1
                available_population = list(set(self.population).difference(set(self.generated_population_names)))
                if len(available_population) > 0:
                    percent = 0.9
                else:
                    percent = 0.8
                metric_np = [
                    np.mean([v for _, v in self.u_matrix[p_name].items()]) for p_name in self.generated_population_names
                ] + [np.mean([v for _, v in self.u_matrix[agent_name].items()])]
                rank = np.argsort(np.argsort(metric_np))[-1]
                if self.use_wandb:
                    wandb.log({"rank": rank}, step=self.total_num_steps)
                threshold = np.ceil(len(self.generated_population_names) * percent)
                if rank >= threshold or self.n_generation_try >= 2:
                    if len(available_population) > 0:
                        p_name = available_population[0]
                        self.trainer.policy_pool.update_policy(p_name, False, model_path={"actor": model_path})
                        logger.success(
                            f"add {model_path} with rank {rank}/{len(self.generated_population_names)} as {p_name}"
                        )
                        self.generated_population_names.append(p_name)
                        self.population_size += 1
                    else:
                        # replace old policy
                        p_name = np.random.choice(self.generated_population_names[:10])
                        self.trainer.policy_pool.update_policy(p_name, False, model_path={"actor": model_path})
                        logger.success(
                            f"replace {model_path} with rank {rank}/{len(self.generated_population_names)} as {p_name}"
                        )
                    # update u_matrix
                    self.u_matrix[p_name] = copy.deepcopy(self.u_matrix[agent_name])
                    for p, v in self.u_matrix[p_name].items():
                        self.u_matrix[p][p_name] = v
                    sp_v = np.mean(self.env_info[f"{agent_name}-{agent_name}-ep_shaped_r"])
                    self.u_matrix[p_name][p_name] = sp_v
                    self.u_matrix[agent_name][p_name] = sp_v

                    self.n_generation_try = 0
                    population_str = zip(
                        self.generated_population_names,
                        [
                            osp.basename(self.trainer.policy_pool.policy_info[a_n][1]["model_path"]["actor"])
                            for a_n in self.generated_population_names
                        ],
                    )
                    logger.success(f"population: size {len(self.generated_population_names)}, {list(population_str)}")

                    metric_np = [
                        np.mean([v for _, v in self.u_matrix[p_name].items()])
                        for p_name in self.generated_population_names
                    ] + [np.mean([v for _, v in self.u_matrix[agent_name].items()])]
                    ranks = np.argsort(np.argsort(metric_np)) + 1
                    logger.success(f"utility matrix sum\n{[round(m, 3) for m in metric_np]}")
                    logger.success(f"ranks\n{ranks}")
                else:
                    self.n_generation_try += 1
                    logger.warning(f"Failed to generate a new partner, try {self.n_generation_try} / 3 times")
                    logger.warning(
                        f"""population metric: {[round(m,3) for m in metric_np[:-1]]}, ego agent metric: {round(metric_np[-1], 3)} rank {rank}/{len(self.generated_population_names)}, need to rank >= {threshold}"""
                    )

            all_agent_pairs = list(itertools.product(self.generated_population_names, [agent_name])) + list(
                itertools.product([agent_name], self.generated_population_names)
            )
            rollout_block_size = self.n_rollout_threads // (self.population_play_ratio + 1)
            map_ea2t = {
                (e, a): agent_name for e, a in itertools.product(range(rollout_block_size), range(self.num_agents))
            }
            metric_np = []
            for p_name in self.generated_population_names:
                metric = np.mean([v for _, v in self.u_matrix[p_name].items()])
                metric_np.append(metric)
            metric_np = np.array(metric_np)
            if metric_np.sum() > 0:
                metric_np = metric_np / metric_np.sum()
            metric_np = 1 - metric_np
            metric_np /= metric_np.sum()

            sampling_prob_np = metric_np

            sampling_prob_np = rankdata(sampling_prob_np, method="dense")
            sampling_prob_np = sampling_prob_np / sampling_prob_np.sum()
            sampling_prob_np = sampling_prob_np**self.all_args.prioritized_alpha
            sampling_prob_np /= sampling_prob_np.sum()
            # logger.info(f"cole sampling prob {sampling_prob_np}")
            # log sampling prob
            sampling_prob_dict = {}
            for i, p_name in enumerate(self.generated_population_names):
                sampling_prob_dict[f"sampling_prob/{p_name}"] = sampling_prob_np[i]
            if self.use_wandb:
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

            sampling_prob_np = np.repeat(sampling_prob_np, 2) / 2
            n_selected = self.n_rollout_threads - rollout_block_size
            pair_idx = np.random.choice(2 * self.population_size, size=(n_selected,), p=sampling_prob_np)
            for i in range(rollout_block_size, self.n_rollout_threads):
                map_ea2t[(i, 0)] = all_agent_pairs[pair_idx[i - rollout_block_size]][0]
                map_ea2t[(i, 1)] = all_agent_pairs[pair_idx[i - rollout_block_size]][1]
            return map_ea2t

        def cole_reset_map_ea2p_fn(episode):
            self.all_args.eval_episodes = (
                self.all_args._eval_episodes * (self.population_size * 2 + 1) // self.all_args.eval_env_batch
            )
            all_agent_pairs = (
                list(itertools.product(self.generated_population_names, [agent_name]))
                + list(itertools.product([agent_name], self.generated_population_names))
                + [(agent_name, agent_name)]
            )
            map_ea2p = {
                (e, a): all_agent_pairs[(self.eval_idx + e) % (self.population_size * 2 + 1)][a]
                for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
            }
            self.eval_idx += self.n_eval_rollout_threads
            self.eval_idx %= self.population_size * 2 + 1

            featurize_type = [
                [self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]
                for e in range(self.n_eval_rollout_threads)
            ]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.naive_train_with_multi_policy(
            reset_map_ea2t_fn=cole_reset_map_ea2t_fn,
            reset_map_ea2p_fn=cole_reset_map_ea2p_fn,
        )


    def biased_train_with_multi_policy(self, reset_map_ea2t_fn=None, reset_map_ea2p_fn=None):
        """This is a naive training loop using TrainerPool and PolicyPool.

        To use PolicyPool and TrainerPool, you should first initialize population in policy_pool, with either:
        >>> self.policy.load_population(population_yaml_path)
        >>> self.trainer.init_population()
        or:
        >>> # mannually register policies
        >>> self.policy.register_policy(policy_name="ppo1", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.policy.register_policy(policy_name="ppo2", policy=rMAPPOpolicy(args, obs_space, share_obs_space, act_space), policy_config=(args, obs_space, share_obs_space, act_space), policy_train=True)
        >>> self.trainer.init_population()

        To bind (env_id, agent_id) to different trainers and policies:
        >>> map_ea2t = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_rollout_threads) for a in range(self.num_agents)}
        # Qs: 2p? n_eval_rollout_threads?
        >>> map_ea2p = {(e, a): "ppo1" if a == 0 else "ppo2" for e in range(self.n_eval_rollout_threads) for a in range(self.num_agents)}
        >>> self.trainer.set_map_ea2t(map_ea2t)
        >>> self.policy.set_map_ea2p(map_ea2p)

        # MARK
        Note that map_ea2t is for training while map_ea2p is for policy evaluations

        WARNING: Currently do not support changing map_ea2t and map_ea2p when training. To implement this, we should take the first obs of next episode in the previous buffers and feed into the next buffers.
        """

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        total_num_steps = 0
        env_infos = defaultdict(list)
        self.eval_info = dict()
        self.env_info = dict()

        for episode in range(0, episodes):
            self.total_num_steps = total_num_steps
            if self.use_linear_lr_decay:
                self.trainer.lr_decay(episode, episodes)

            # reset env agents
            if reset_map_ea2t_fn is not None:
                map_ea2t = reset_map_ea2t_fn(episode)
                self.trainer.reset(
                    map_ea2t,
                    self.n_rollout_threads,
                    self.num_agents,
                    load_unused_to_cpu=True,
                )
                if self.all_args.use_policy_in_env:
                    load_policy_cfg = np.full((self.n_rollout_threads, self.num_agents), fill_value=None).tolist()
                    for e in range(self.n_rollout_threads):
                        for a in range(self.num_agents):
                            trainer_name = map_ea2t[(e, a)]
                            if trainer_name not in self.trainer.on_training:
                                load_policy_cfg[e][a] = self.trainer.policy_pool.policy_info[trainer_name]
                    self.envs.load_policy(load_policy_cfg)

            # init env
            obs, share_obs, available_actions = self.envs.reset()

            # replay buffer
            if self.use_centralized_V:
                share_obs = share_obs
            else:
                share_obs = obs

            s_time = time.time()
            self.trainer.init_first_step(share_obs, obs, available_actions)

            

            for step in range(self.episode_length):
                # Sample actions
                actions = self.trainer.step(step)

                # Observe reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                total_num_steps += self.n_rollout_threads
                self.envs.anneal_reward_shaping_factor(self.trainer.reward_shaping_steps())

                

                bad_masks = np.array([[[0.0] if info["bad_transition"] else [1.0]] * self.num_agents for info in infos])

                self.trainer.insert_data(
                    share_obs,
                    obs,
                    rewards,
                    dones,
                    bad_masks=bad_masks,
                    infos=infos,
                    available_actions=available_actions,
                )

            # update env infos
            episode_env_infos = defaultdict(list)
            ep_returns_per_trainer = defaultdict(lambda: [[] for _ in range(self.num_agents)])
            e2ta = dict()
            if self.env_name == "Overcooked":
                if self.all_args.overcooked_version == "old":
                    from zsceval.envs.overcooked.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                else:
                    from zsceval.envs.overcooked_new.src.overcooked_ai_py.mdp.overcooked_mdp import (
                        SHAPED_INFOS,
                    )

                    shaped_info_keys = SHAPED_INFOS
                for e, info in enumerate(infos):
                    agent0_trainer = self.trainer.map_ea2t[(e, 0)]
                    agent1_trainer = self.trainer.map_ea2t[(e, 1)]
                    for log_name in [
                        f"{agent0_trainer}-{agent1_trainer}",
                    ]:
                        episode_env_infos[f"{log_name}-ep_sparse_r"].append(info["episode"]["ep_sparse_r"])
                        episode_env_infos[f"{log_name}-ep_shaped_r"].append(info["episode"]["ep_shaped_r"])
                        for a in range(self.num_agents):
                            # if getattr(self.all_args, "stage", 1) == 1 or not self.all_args.use_wandb:
                            for i, k in enumerate(shaped_info_keys):
                                episode_env_infos[f"{log_name}-ep_{k}_by_agent{a}"].append(
                                    info["episode"]["ep_category_r_by_agent"][a][i]
                                )
                            
                            episode_env_infos[f"{log_name}-ep_sparse_r_by_agent{a}"].append(
                                info["episode"]["ep_sparse_r_by_agent"][a]
                            )
                            episode_env_infos[f"{log_name}-ep_shaped_r_by_agent{a}"].append(
                                info["episode"]["ep_shaped_r_by_agent"][a]
                            )
                    for k in ["ep_sparse_r", "ep_shaped_r"]:
                        for log_name in [
                            f"either-{agent0_trainer}-{k}",
                            f"either-{agent0_trainer}-{k}-as_agent_0",
                            f"either-{agent1_trainer}-{k}",
                            f"either-{agent1_trainer}-{k}-as_agent_1",
                        ]:
                            episode_env_infos[log_name].append(info["episode"][k])
                    if agent0_trainer != self.trainer.agent_name:
                        # suitable for both stage 1 and stage 2
                        if self.all_args.use_primitive_hsp:
                            ep_returns_per_trainer[agent1_trainer][1].append(info["episode"]["ep_sparse_r"])
                        else:
                            ep_returns_per_trainer[agent1_trainer][1].append(info["episode"]["ep_shaped_r"])
                        e2ta[e] = (agent1_trainer, 1)
                    elif agent1_trainer != self.trainer.agent_name:
                        if self.all_args.use_primitive_hsp:
                            ep_returns_per_trainer[agent1_trainer][1].append(info["episode"]["ep_sparse_r"])
                        else:
                            ep_returns_per_trainer[agent0_trainer][0].append(info["episode"]["ep_shaped_r"])
                        e2ta[e] = (agent0_trainer, 0)
                logger.debug(episode_env_infos)
                env_infos.update(episode_env_infos)
            max_ep_shaped_r_dict = defaultdict(lambda: [0, 0])

            self.env_info.update(env_infos)
            e_time = time.time()
            logger.trace(f"Rollout time: {e_time - s_time:.3f}s")

            # compute return and update network
            s_time = time.time()

            self.trainer.adapt_entropy_coef(total_num_steps)
            train_infos = self.trainer.train(sp_size=getattr(self, "n_repeats", 0) * self.num_agents)
            
            e_time = time.time()
            
            logger.trace(f"Update models time: {e_time - s_time:.3f}s")

            s_time = time.time()

            if self.all_args.use_advantage_prioritized_sampling:
                if not hasattr(self, "avg_adv"):
                    self.avg_adv = defaultdict(float)
                adv = self.trainer.compute_advantages()
                for (agent0, agent1, a), vs in adv.items():
                    agent_pair = (agent0, agent1)
                    for v in vs:
                        if agent_pair not in self.avg_adv.keys():
                            self.avg_adv[agent_pair] = v
                        else:
                            self.avg_adv[agent_pair] = self.avg_adv[agent_pair] * 0.99 + v * 0.01

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode < 50:
                if episode % 2 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            elif episode < 100:
                if episode % 5 == 0:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)
            else:
                if episode % self.save_interval == 0 or episode == episodes - 1:
                    self.trainer.save(total_num_steps, save_dir=self.save_dir)
                    # self.trainer.save(episode, save_dir=self.save_dir)

            self.trainer.update_best_r(
                {
                    trainer_name: np.mean(self.env_info.get(f"either-{trainer_name}-ep_shaped_r", -1e9))
                    for trainer_name in self.trainer.active_trainers
                },
                save_dir=self.save_dir,
            )

            # log information
            if episode % self.log_interval == 0 or episode == episodes - 1:
                end = time.time()
                eta_t = eta(start, end, self.num_env_steps, total_num_steps)
                logger.info(
                    "Layout {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}, ETA {}.".format(
                        self.all_args.layout_name,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                        eta_t,
                    )
                )
                average_ep_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(train_infos[k]):.3f}"
                    for k in train_infos.keys()
                    if "average_episode_rewards" in k and "either" not in k
                }
                logger.info(f"average episode rewards is\n{pprint.pformat(average_ep_rew_dict, width=600)}")
                average_ep_shaped_rew_dict = {
                    k[: k.rfind("-")]: f"{np.mean(env_infos[k]):.3f}"
                    for k in env_infos.keys()
                    if k.endswith("ep_shaped_r") and "either" not in k
                }
                logger.info(
                    f"average shaped episode rewards is\n{pprint.pformat(average_ep_shaped_rew_dict, width=600, compact=True)}"
                )
                if self.all_args.algorithm_name == "traj":
                    if self.all_args.stage == 1:
                        logger.debug(f'jsd is {train_infos["average_jsd"]}')
                        logger.debug(f'jsd loss is {train_infos["average_jsd_loss"]}')

                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)
                if self.use_wandb:
                    wandb.log({"train/ETA": eta_t}, step=total_num_steps)

            # eval
            if episode > 0 and episode % self.eval_interval == 0 and self.use_eval or episode == episodes - 1:
                if reset_map_ea2p_fn is not None:
                    map_ea2p = reset_map_ea2p_fn(episode)
                    self.policy.set_map_ea2p(map_ea2p, load_unused_to_cpu=True)
                eval_info = self.evaluate_with_multi_policy()
                # logger.debug("eval_info: {}".format(pprint.pformat(eval_info)))
                self.log_env(eval_info, total_num_steps)
                self.eval_info.update(eval_info)

            e_time = time.time()
            logger.trace(f"Post update models time: {e_time - s_time:.3f}s")

    def train_adaptive_population(self):

        assert self.all_args.population_size == len(self.trainer.population)
        self.population_size = self.all_args.population_size
        self.population = sorted(
            self.trainer.population.keys()
        )  # Note index and trainer name would not match when there are >= 10 agents

        logger.info(f"population_size: {self.all_args.population_size}, {self.population}")

        # Stage 2: train an agent against population with prioritized sampling
        agent_name = self.trainer.agent_name
        assert self.use_eval
        assert (
            self.n_eval_rollout_threads % self.all_args.eval_env_batch == 0
            and self.all_args.eval_episodes % self.all_args.eval_env_batch == 0
        )
        assert self.n_rollout_threads % self.all_args.train_env_batch == 0
        self.all_args.eval_episodes = (
            self.all_args.eval_episodes * self.population_size // self.all_args.eval_env_batch
        )
        self.eval_idx = 0
        all_agent_pairs = list(itertools.product(self.population, [agent_name])) + list(
            itertools.product([agent_name], self.population)
        )
        logger.info(f"all agent pairs: {all_agent_pairs}")

        running_avg_r = -np.ones((self.population_size * 2,), dtype=np.float32) * 1e9

        def mep_reset_map_ea2t_fn(episode):
            # Randomly select agents from population to be trained
            # 1) consistent with MEP to train against one agent each episode 2) sample different agents to train against
            sampling_prob_np = np.ones((self.population_size * 2,)) / self.population_size / 2


            if self.all_args.use_advantage_prioritized_sampling:
                # logger.debug("use advantage prioritized sampling")
                if episode > 0:
                    metric_np = np.array([self.avg_adv[agent_pair] for agent_pair in all_agent_pairs])
                    # TODO: retry this
                    sampling_rank_np = rankdata(metric_np, method="dense")
                    sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                    sampling_prob_np /= sampling_prob_np.sum()
                    maxv = 1.0 / (self.population_size * 2) * 10
                    while sampling_prob_np.max() > maxv + 1e-6:
                        sampling_prob_np = sampling_prob_np.clip(max=maxv)
                        sampling_prob_np /= sampling_prob_np.sum()

            elif self.all_args.mep_use_prioritized_sampling:
                metric_np = np.zeros((self.population_size * 2,))
                for i, agent_pair in enumerate(all_agent_pairs):
                    train_r = np.mean(self.env_info.get(f"{agent_pair[0]}-{agent_pair[1]}-ep_sparse_r", -1e9))
                    eval_r = np.mean(
                        self.eval_info.get(
                            f"{agent_pair[0]}-{agent_pair[1]}-eval_ep_shaped_r",
                            -1e9,
                        )
                    )

                    avg_r = 0.0
                    cnt_r = 0
                    if train_r > -1e9:
                        avg_r += train_r * (self.n_rollout_threads // self.all_args.train_env_batch)
                        cnt_r += self.n_rollout_threads // self.all_args.train_env_batch
                    if eval_r > -1e9:
                        avg_r += eval_r * (
                            self.all_args.eval_episodes
                            // (self.n_eval_rollout_threads // self.all_args.eval_env_batch)
                        )
                        cnt_r += self.all_args.eval_episodes // (
                            self.n_eval_rollout_threads // self.all_args.eval_env_batch
                        )
                    if cnt_r > 0:
                        avg_r /= cnt_r
                    else:
                        avg_r = -1e9
                    if running_avg_r[i] == -1e9:
                        running_avg_r[i] = avg_r
                    else:
                        # running average
                        running_avg_r[i] = running_avg_r[i] * 0.95 + avg_r * 0.05
                    metric_np[i] = running_avg_r[i]
                running_avg_r_dict = {}
                for i, agent_pair in enumerate(all_agent_pairs):
                    running_avg_r_dict[f"running_average_return/{agent_pair[0]}-{agent_pair[1]}"] = np.mean(
                        running_avg_r[i]
                    )
                if self.use_wandb:
                    for k, v in running_avg_r_dict.items():
                        if v > -1e9:
                            wandb.log({k: v}, step=self.total_num_steps)
                running_avg_r_dict = {
                    f"running_average_return/{agent_pair[0]}-{agent_pair[1]}": f"{running_avg_r[i]:.3f}"
                    for i, agent_pair in enumerate(all_agent_pairs)
                }
                logger.trace(f"running avg_r\n{pprint.pformat(running_avg_r_dict, width=600, compact=True)}")
                if (metric_np > -1e9).astype(np.int32).sum() > 0:
                    avg_metric = metric_np[metric_np > -1e9].mean()
                else:
                    # uniform
                    avg_metric = 1.0
                metric_np[metric_np == -1e9] = avg_metric

                # reversed return
                sampling_rank_np = rankdata(1.0 / (metric_np + 1e-6), method="dense")
                sampling_prob_np = sampling_rank_np / sampling_rank_np.sum()
                sampling_prob_np = sampling_prob_np**self.all_args.mep_prioritized_alpha
                sampling_prob_np /= sampling_prob_np.sum()


            assert abs(sampling_prob_np.sum() - 1) < 1e-3

            # log sampling prob
            sampling_prob_dict = {}
            for i, agent_pair in enumerate(all_agent_pairs):
                sampling_prob_dict[f"sampling_prob/{agent_pair[0]}-{agent_pair[1]}"] = sampling_prob_np[i]
            if self.use_wandb:
                wandb.log(sampling_prob_dict, step=self.total_num_steps)

            n_selected = self.n_rollout_threads // self.all_args.train_env_batch
            pair_idx = np.random.choice(2 * self.population_size, size=(n_selected,), p=sampling_prob_np)
            if self.all_args.uniform_sampling_repeat > 0:
                assert n_selected >= 2 * self.population_size * self.all_args.uniform_sampling_repeat
                i = 0
                for r in range(self.all_args.uniform_sampling_repeat):
                    for x in range(2 * self.population_size):
                        pair_idx[i] = x
                        i += 1
            map_ea2t = {
                (e, a): all_agent_pairs[pair_idx[e % n_selected]][a]
                for e, a in itertools.product(range(self.n_rollout_threads), range(self.num_agents))
            }

            return map_ea2t

        def mep_reset_map_ea2p_fn(episode):
            if self.all_args.eval_policy != "":
                map_ea2p = {
                    (e, a): [self.all_args.eval_policy, agent_name][(e + a) % 2]
                    for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                }
            else:
                map_ea2p = {
                    (e, a): all_agent_pairs[
                        (self.eval_idx + e // self.all_args.eval_env_batch) % (self.population_size * 2)
                    ][a]
                    for e, a in itertools.product(range(self.n_eval_rollout_threads), range(self.num_agents))
                }
                self.eval_idx += self.n_eval_rollout_threads // self.all_args.eval_env_batch
                self.eval_idx %= self.population_size * 2
            featurize_type = [
                [self.policy.featurize_type[map_ea2p[(e, a)]] for a in range(self.num_agents)]
                for e in range(self.n_eval_rollout_threads)
            ]
            self.eval_envs.reset_featurize_type(featurize_type)
            return map_ea2p

        self.biased_train_with_multi_policy(
            reset_map_ea2t_fn=mep_reset_map_ea2t_fn,
            reset_map_ea2p_fn=mep_reset_map_ea2p_fn,
        )