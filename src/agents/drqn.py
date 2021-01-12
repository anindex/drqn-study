from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import torch
import numpy as np

from src.agents.base import Agent, adjust_learning_rate


class DRQNAgent(Agent):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        super(DRQNAgent, self).__init__(env_prototype, model_prototype, memory_prototype, **kwargs)
        self.logger.info('<===================================> DRQNAgent')
        # DRQN agent params
        self.drqn_n_step = kwargs.get('drqn_n_step', 2)
        # DRQN loggings
        self.log_lstm_grad = kwargs.get('log_lstm_grad', False)

    def _reset_training_loggings(self):
        super(DRQNAgent, self)._reset_training_loggings()
        self.grad_mean_ih = [0]
        self.grad_mean_hh = [0]
        self.grad_max_ih = [0]
        self.grad_max_hh = [0]

    def _log_lstm_grad(self, named_parameters):
        ave_grads_ih = []
        max_grads_ih = []
        ave_grads_hh = []
        max_grads_hh = []
        for n, p in self.model.named_parameters():
            if(p.requires_grad) and ("bias" not in n) and ("lstm" in n):
                if 'ih' in n:
                    ave_grads_ih.append(p.grad.abs().mean().item())
                    max_grads_ih.append(p.grad.abs().max().item())
                elif 'hh' in n:
                    ave_grads_hh.append(p.grad.abs().mean().item())
                    max_grads_hh.append(p.grad.abs().max().item())
        self.grad_mean_ih.append(np.mean(ave_grads_ih))
        self.grad_mean_hh.append(np.mean(ave_grads_hh))
        self.grad_max_ih.append(np.mean(max_grads_ih))
        self.grad_max_hh.append(np.mean(max_grads_hh))

    def _process_episode_batch(self, episodes):
        batch_size, eps_len = len(episodes), len(episodes[0])
        s0_eps_batch = np.array([[episodes[i][j].s0 for j in range(eps_len)] for i in range(batch_size)])
        a_eps_batch = np.array([[episodes[i][j].a for j in range(eps_len)] for i in range(batch_size)])
        r_eps_batch = np.array([[episodes[i][j].r for j in range(eps_len)] for i in range(batch_size)])
        s1_eps_batch = np.array([[episodes[i][j].s1 for j in range(eps_len)] for i in range(batch_size)])
        t1_eps_batch = np.array([[float(not episodes[i][j].t1) for j in range(eps_len)] for i in range(batch_size)])
        return s0_eps_batch, a_eps_batch, r_eps_batch, s1_eps_batch, t1_eps_batch

    def _get_n_step_q_update(self, s0_batch, a_batch, r_batch, s1_batch, t1_batch, current_hidden, next_current_hidden, target_hidden):
        s0_batch = torch.from_numpy(s0_batch).type(self.dtype).to(self.device)
        a_batch = torch.from_numpy(a_batch[:, 0]).long().to(self.device)  # first action in the window
        r_batch = torch.from_numpy(r_batch.sum(axis=1)).type(self.dtype).to(self.device)  # accumulate rewards over temporal dim
        s1_batch = torch.from_numpy(s1_batch).type(self.dtype).to(self.device)
        t1_batch = torch.from_numpy(t1_batch[:, -1]).type(self.dtype).to(self.device)
        # Compute target Q values for mini-batch update.
        if self.bootstrap_type == 'double_q':
            q_values, new_next_current_hidden = self.model(s1_batch, next_current_hidden)   # Detach this variable from the current graph since we don't want gradients to propagate
            q_values = q_values[:, -1, :].squeeze(1).detach()
            _, q_max_actions = q_values.max(dim=1, keepdim=True)
            next_max_q_values, new_target_hidden = self.target_model(s1_batch, target_hidden)
            next_max_q_values = next_max_q_values[:, -1, :].squeeze(1).detach()
            next_max_q_values = next_max_q_values.gather(1, q_max_actions)
        elif self.bootstrap_type == 'target_q':
            new_next_current_hidden = next_current_hidden  # do not use double Q
            next_max_q_values, new_target_hidden = self.target_model(s1_batch, target_hidden)
            next_max_q_values = next_max_q_values[:, -1, :].detach().squeeze(1)
            next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        elif self.bootstrap_type == 'learn_q':
            new_target_hidden = target_hidden  # do not use target hidden
            next_max_q_values, new_next_current_hidden = self.model(s1_batch, next_current_hidden)
            next_max_q_values = next_max_q_values[:, -1, :].detach().squeeze(1)
            next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        else:
            raise ValueError('Input bootstrapping type is not supported!')
        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
        current_q_values, new_current_hidden = self.model(s0_batch, current_hidden)
        current_q_values = current_q_values[:, 0, :].squeeze(1).gather(1, a_batch.unsqueeze(1)).squeeze()
        # Set discounted reward to zero for all states that were terminal.
        next_max_q_values = (next_max_q_values * t1_batch.unsqueeze(1)).squeeze()
        expected_q_values = (r_batch + self.gamma * next_max_q_values).squeeze()
        td_error = self.value_criteria(current_q_values, expected_q_values)
        if self.step != 0 and self.step % self.log_step_interval == 0:  # logging
            self.window_max_abs_q.append(np.mean(np.abs(next_max_q_values.tolist())))
            self.max_abs_q_log.append(np.max(self.window_max_abs_q))
            self.tderr_log.append(td_error.item())
            self.step_log.append(self.step)
        return td_error, new_current_hidden, new_next_current_hidden, new_target_hidden

    def _forward(self, states, hidden):
        states = torch.from_numpy(states).unsqueeze(0).unsqueeze(1).type(self.dtype).to(self.device)
        q_values, new_hidden = self.model(states, hidden)
        action = self._epsilon_greedy(q_values.detach())
        return action, new_hidden

    def _backward(self, episodes=None):
        # Train the network on a single stochastic batch.
        if self.step % self.train_interval == 0:
            if episodes is None:
                episodes = self.memory.sample_batch(self.batch_size)
            batch_size = len(episodes)
            s0_eps_batch, a_eps_batch, r_eps_batch, s1_eps_batch, t1_eps_batch = self._process_episode_batch(episodes)
            # reset hidden every training
            current_hidden = self._create_zero_lstm_hidden(batch_size=batch_size)
            next_current_hidden = self._create_zero_lstm_hidden(batch_size=batch_size)
            target_hidden = self._create_zero_lstm_hidden(batch_size=batch_size)
            for i in range(0, len(episodes[0]), self.drqn_n_step):
                td_error, current_hidden, next_current_hidden, target_hidden = self._get_n_step_q_update(s0_eps_batch[:, i:i + self.drqn_n_step],
                                                                                                         a_eps_batch[:, i:i + self.drqn_n_step],
                                                                                                         r_eps_batch[:, i:i + self.drqn_n_step],
                                                                                                         s1_eps_batch[:, i:i + self.drqn_n_step],
                                                                                                         t1_eps_batch[:, i:i + self.drqn_n_step],
                                                                                                         current_hidden, next_current_hidden, target_hidden)
                self.optimizer.zero_grad()
                td_error.backward(retain_graph=True)
                if self.log_lstm_grad and self.step != 0 and self.step % self.log_step_interval == 0:
                    self._log_lstm_grad(self.model.named_parameters())
                for param in self.model.parameters():
                    param.grad.data.clamp_(-self.clip_grad, self.clip_grad)
                self.optimizer.step()
        # adjust learning rate if enabled
        if self.lr_decay:
            self.lr_adjusted = max(self.lr * (self.steps - self.step) / self.steps, 1e-32)
            adjust_learning_rate(self.optimizer, self.lr_adjusted)
        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self._update_target_model_hard()    # Hard update every `target_model_update` steps.
        if self.target_model_update < 1.:
            self._update_target_model_soft()    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.

    def _random_initialization(self):
        self.logger.info('<===================================> Random policy initialization for %d eps' % self.random_eps)
        for e in range(self.random_eps):
            self.experience = self.env.reset()
            episode = []
            while not self.experience.t1:
                action = self.env.sample_random_action()
                self.experience = self.env.step(action)
                episode.append(self.experience)
            self._backward([episode])

    def fit_model(self):
        self._init_model(training=True)
        self.eps = self.eps_start
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_adjusted = self.lr
        self._reset_training_loggings()
        self.start_time = time.time()
        self.step = 0
        nepisodes = 0
        total_reward = 0.
        self._random_initialization()
        episode_steps, episode_reward = 0, 0
        self.experience = self.env.reset()
        hidden = self._create_zero_lstm_hidden()
        self.logger.info('<===================================> Training ...')
        while self.step < self.steps:
            action, hidden = self._forward(self.experience.s1, hidden)
            self.experience = self.env.step(action)
            self._store_experience(self.experience)
            self._visualize(visualize=self.train_visualize)
            if self.step > self.learn_start:
                self._backward()
            episode_steps += 1
            episode_reward += self.experience.r
            self.step += 1
            if self.experience.t1:
                self.experience = self.env.reset()
                self.window_scores.append(episode_reward)
                run_avg_reward = np.mean(self.window_scores)
                total_reward += episode_reward
                nepisodes += 1
                total_avg_reward = total_reward / nepisodes if nepisodes > 0 else 0
                if nepisodes % self.log_episode_interval == 0:
                    if self.use_tensorboard:
                        self.writer.add_scalar('run_avg_reward/eps', run_avg_reward, nepisodes)
                    self.run_avg_score_log.append(run_avg_reward)
                    self.total_avg_score_log.append(total_avg_reward)
                    self.eps_log.append(nepisodes)
                if run_avg_reward > self.env.solved_criteria:
                    self._save_model(self.step, episode_reward)
                    if self.solved_stop:
                        break
                hidden = self._create_zero_lstm_hidden()
                episode_steps, episode_reward = 0, 0
            if self.step % self.prog_freq == 0:
                self.logger.info('Reporting at %d step | Elapsed Time: %s' % (self.step, str(time.time() - self.start_time)))
                self.logger.info('Training Stats: lr: %f  epsilon: %f nepisodes: %d ' % (self.lr_adjusted, self.eps, nepisodes))
                self.logger.info('Training Stats: total_reward: %f  total_avg_reward: %f run_avg_reward: %f ' % (total_reward, total_avg_reward, run_avg_reward))

    def test_model(self):
        if not self.model:
            self._init_model(training=False)
        for episode in range(self.test_nepisodes):
            episode_reward = 0
            episode_steps = 0
            hidden = self._create_zero_lstm_hidden()
            self.experience = self.env.reset()
            while not self.experience.t1:
                action, hidden = self._forward(self.experience.s1, hidden)
                self.experience = self.env.step(action)
                self._visualize(visualize=True)
                episode_steps += 1
                episode_reward += self.experience.r
                if episode_reward > self.env.solved_criteria:
                    self.logger.info('Test episode %d at %d step with reward %f ' % (episode, episode_steps, episode_reward))
