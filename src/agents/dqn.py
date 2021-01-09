from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import torch
import numpy as np
from collections import deque

from src.agents.base import Agent, adjust_learning_rate


class DQNAgent(Agent):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        super(DQNAgent, self).__init__(env_prototype, model_prototype, memory_prototype, **kwargs)
        self.logger.info('<===================================> DQNAgent')

    def _get_q_update(self, experiences):
        batch_size = len(experiences)
        s0_batch = torch.from_numpy(np.array([experiences[i].s0 for i in range(batch_size)])).type(self.dtype).to(self.device)
        a_batch = torch.from_numpy(np.array([experiences[i].a for i in range(batch_size)])).long().to(self.device)
        r_batch = torch.from_numpy(np.array([experiences[i].r for i in range(batch_size)])).type(self.dtype).to(self.device)
        s1_batch = torch.from_numpy(np.array([experiences[i].s1 for i in range(batch_size)])).type(self.dtype).to(self.device)
        t1_batch = torch.from_numpy(np.array([float(not experiences[i].t1) for i in range(batch_size)])).type(self.dtype).to(self.device)
        # Compute target Q values for mini-batch update.
        if self.enable_double_dqn:
            q_values = self.model(s1_batch).detach()    # Detach this variable from the current graph since we don't want gradients to propagate
            _, q_max_actions = q_values.max(dim=1, keepdim=True)
            next_max_q_values = self.target_model(s1_batch).detach()
            next_max_q_values = next_max_q_values.gather(1, q_max_actions)
        else:
            next_max_q_values = self.target_model(s1_batch).detach()
            next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
        current_q_values = self.model(s0_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        # Set discounted reward to zero for all states that were terminal.
        next_max_q_values = next_max_q_values * t1_batch.unsqueeze(1)
        expected_q_values = (r_batch + self.gamma * next_max_q_values.squeeze()).squeeze()
        td_error = self.value_criteria(current_q_values, expected_q_values)
        return td_error

    def _forward(self, states):
        states = torch.from_numpy(states).unsqueeze(0).type(self.dtype).to(self.device)
        q_values = self.model(states).detach()
        action = self._epsilon_greedy(q_values)
        return action

    def _backward(self, experiences=None):
        # Train the network on a single stochastic batch.
        if self.step % self.train_interval == 0:
            if experiences is None:
                experiences = self.memory.sample_batch(self.batch_size)
            td_error = self._get_q_update(experiences)
            self.optimizer.zero_grad()
            td_error.backward()
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
            while not self.experience.t1:
                action = self.env.sample_random_action()
                self.experience = self.env.step(action)
                self._backward([self.experience])

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
        window_rewards = deque(maxlen=self.run_avg_nepisodes)
        episode_steps, episode_reward = 0, 0
        self._random_initialization()
        self.logger.info('<===================================> Training ...')
        self.experience = self.env.reset()
        while self.step < self.steps:
            action = self._forward(self.experience.s1)
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
                window_rewards.append(episode_reward)
                run_avg_reward = sum(window_rewards) / len(window_rewards)
                total_reward += episode_reward
                nepisodes += 1
                if nepisodes % self.log_episode_interval == 0:
                    self.writer.add_scalar('run_avg_reward/eps', run_avg_reward, nepisodes)
                if run_avg_reward > self.env.solved_criteria:
                    self._save_model(self.step, episode_reward)
                    if self.solved_stop:
                        break
                episode_steps, episode_reward = 0, 0
            # report training stats
            # if self.step % self.log_step_interval == 0:
            #    self.writer.add_scalar('run_avg_reward/steps', run_avg_reward, self.step)
            if self.step % self.prog_freq == 0:
                self.logger.info('Reporting at %d step | Elapsed Time: %s' % (self.step, str(time.time() - self.start_time)))
                self.logger.info('Training Stats: lr: %f  epsilon: %f nepisodes: %d ' % (self.lr_adjusted, self.eps, nepisodes))
                self.logger.info('Training Stats: total_reward: %f  total_avg_reward: %f run_avg_reward: %f ' % (total_reward, total_reward/nepisodes if nepisodes > 0 else 0, run_avg_reward))

    def test_model(self):
        if not self.model:
            self._init_model(training=False)
        for episode in range(self.test_nepisodes):
            episode_reward = 0
            episode_steps = 0
            self.experience = self.env.reset()
            while not self.experience.t1:
                action = self._forward(self.experience.s1)
                self.experience = self.env.step(action)
                self._visualize(visualize=True)
                episode_steps += 1
                episode_reward += self.experience.r
                if episode_reward > self.env.solved_criteria:
                    self.logger.info('Test episode %d at %d step with reward %f ' % (episode, episode_steps, episode_reward))
