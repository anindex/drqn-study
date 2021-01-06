from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import torch
from collections import deque

from src.agents.base import Agent, adjust_learning_rate


class DRQNAgent(Agent):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        super(DRQNAgent, self).__init__(env_prototype, model_prototype, memory_prototype, **kwargs)
        self.logger.info('<===================================> DRQNAgent')
        # env
        self.env = self.env_prototype(**self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim = self.env.action_dim
        # cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_params['use_cuda'] = torch.cuda.is_available()
        # model
        self.model_params['state_shape'] = self.state_shape
        self.model_params['action_dim'] = self.action_dim
        self.model_params['seq_len'] = self.env_params['seq_len']

    def _forward(self, states):
        states = torch.from_numpy(states).unsqueeze(0).type(self.dtype).to(self.device)
        q_values_ts = self.model(states).detach()
        action = self._epsilon_greedy(q_values_ts)
        return action

    def _backward(self, experience):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(experience.s0, experience.a, experience.r, experience.s1, experience.t1)
        # Train the network on a single stochastic batch.
        if self.step > self.learn_start and self.step % self.train_interval == 0:
            experiences = self.memory.sample_batch(self.batch_size)
            td_error_vb = self._get_q_update(experiences)
            self.optimizer.zero_grad()
            td_error_vb.backward()
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

    def fit_model(self):
        self._init_model(training=True)
        self.eps = self.eps_start
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_adjusted = self.lr
        self.logger.info('<===================================> Training ...')
        self._reset_training_loggings()
        self.start_time = time.time()
        self.step = 0
        nepisodes = 0
        total_reward = 0.
        window_rewards = deque(maxlen=self.run_avg_nepisodes)
        new_eps = True
        while self.step < self.steps:
            if new_eps:    # start of a new episode
                episode_steps = 0
                episode_reward = 0.
                self.experience = self.env.reset()
                self._visualize(visualize=self.train_visualize)
                new_eps = False  # reset flag
            action = self._forward(self.experience.s1)
            self.experience = self.env.step(action)
            self._visualize(visualize=self.train_visualize)
            new_eps = bool(self.experience.t1)
            self._backward(self.experience)
            episode_steps += 1
            episode_reward += self.experience.r
            self.step += 1
            if new_eps:
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
            # report training stats
            if self.step % self.log_step_interval == 0:
                self.writer.add_scalar('run_avg_reward/steps', run_avg_reward, self.step)
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
