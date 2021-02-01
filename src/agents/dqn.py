from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import torch
import numpy as np

from src.agents.base import Agent, adjust_learning_rate


class DQNAgent(Agent):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        super(DQNAgent, self).__init__(env_prototype, model_prototype, memory_prototype, **kwargs)
        self.logger.info('<===================================> DQNAgent')

    def _get_loss(self, experiences, logging=True):
        batch_size = len(experiences)
        s0_batch = torch.from_numpy(np.asarray([experiences[i][0] for i in range(batch_size)])).type(self.dtype).to(self.device)
        a_batch = torch.from_numpy(np.asarray([experiences[i][1] for i in range(batch_size)])).long().to(self.device)
        r_batch = torch.from_numpy(np.asarray([experiences[i][2] for i in range(batch_size)])).type(self.dtype).to(self.device)
        s1_batch = torch.from_numpy(np.asarray([experiences[i][3] for i in range(batch_size)])).type(self.dtype).to(self.device)
        t1_batch = torch.from_numpy(np.asarray([experiences[i][4] for i in range(batch_size)])).type(self.dtype).to(self.device)
        # Compute target Q values for mini-batch update.
        if self.bootstrap_type == 'double_q':
            q_values = self.model(s1_batch).detach()    # Detach this variable from the current graph since we don't want gradients to propagate
            _, q_max_actions = q_values.max(dim=1, keepdim=True)
            next_max_q_values = self.target_model(s1_batch).detach()
            next_max_q_values = next_max_q_values.gather(1, q_max_actions)
        elif self.bootstrap_type == 'target_q':
            next_max_q_values = self.target_model(s1_batch).detach()
            next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        elif self.bootstrap_type == 'learn_q':
            next_max_q_values = self.model(s1_batch).detach()
            next_max_q_values, _ = next_max_q_values.max(dim=1, keepdim=True)
        else:
            raise ValueError('Input bootstrapping type is not supported!')
        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
        current_q_values = self.model(s0_batch).gather(1, a_batch.unsqueeze(1)).squeeze()
        # Set discounted reward to zero for all states that were terminal.
        next_max_q_values = (next_max_q_values * t1_batch.unsqueeze(1)).squeeze()
        expected_q_values = (r_batch + self.gamma * next_max_q_values).squeeze()
        loss = self.value_criteria(current_q_values, expected_q_values)
        error = (current_q_values - next_max_q_values).tolist()
        if logging and self.step != 0 and self.step % self.log_step_interval == 0:  # logging
            self.window_max_abs_q.append(np.mean(np.abs(next_max_q_values.tolist())))
            self.max_abs_q_log.append(np.max(self.window_max_abs_q))
            self.loss_log.append(loss.item())
            self.step_log.append(self.step)
        return loss, error

    def _forward(self, states):
        states = torch.from_numpy(states).unsqueeze(0).type(self.dtype).to(self.device)
        q_values = self.model(states).detach()
        action = self._epsilon_greedy(q_values)
        return action

    def _backward(self, experiences=None, idxs=None, is_weights=None):
        # Train the network on a single stochastic batch.
        error = 0.
        if self.step % self.train_interval == 0:
            if experiences is None:
                experiences, idxs, is_weights = self.memory.sample_batch(self.batch_size)
            loss, error = self._get_loss(experiences)
            if idxs is not None:  # update priorities
                error = np.abs(error)  # abs error as priorities
                for i in range(self.batch_size):
                    self.memory.update(idxs[i], error[i])
            is_weights = np.ones(len(experiences)) if is_weights is None else is_weights
            loss = (torch.from_numpy(is_weights).type(self.dtype).to(self.device) * loss).mean()  # apply importance weights
            self.optimizer.zero_grad()
            loss.backward()
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
        return error

    def _random_initialization(self):
        self.logger.info('<===================================> Random policy initialization for %d eps' % self.random_eps)
        for e in range(self.random_eps):
            self.experience = self.env.reset()
            while not self.env.episode_ended:
                action = self.env.sample_random_action()
                self.experience = self.env.step(action)
                error = self._backward([self.experience])
                self._store_experience(self.experience, abs(error))

    def fit_model(self):
        self._init_model(training=True)
        self.eps = self.eps_start
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_adjusted = self.lr
        self._reset_training_loggings()
        self.start_time = time.time()
        self.step = 0
        total_reward = 0.
        self._random_initialization()
        self.logger.info('<===================================> Training ...')
        for self.episode in range(self.episodes):
            self.experience = self.env.reset()
            episode_steps, episode_reward = 0, 0
            while not self.env.episode_ended:
                action = self._forward(self.experience[3])
                self.experience = self.env.step(action)
                _, error = self._get_loss([self.experience], logging=False)  # compute priority
                self._store_experience(self.experience, abs(error))
                self._visualize(visualize=self.train_visualize)
                if self.step > self.learn_start:
                    self._backward()
                episode_steps += 1
                episode_reward += self.experience[2]
                self.step += 1
            self.window_scores.append(episode_reward)
            run_avg_reward = np.mean(self.window_scores)
            total_reward += episode_reward
            total_avg_reward = total_reward / self.episode if self.episode > 0 else 0
            if self.episode % self.log_episode_interval == 0:
                if self.use_tensorboard:
                    self.writer.add_scalar('run_avg_reward/episode', run_avg_reward, self.episode)
                self.run_avg_score_log.append(run_avg_reward)
                self.total_avg_score_log.append(total_avg_reward)
                self.eps_log.append(self.episode)
            if run_avg_reward > self.env.solved_criteria:
                self._save_model(self.step, episode_reward)
                if self.solved_stop:
                    break
            if self.episode % self.prog_freq == 0:
                self.logger.info('Reporting at episode %d | Elapsed Time: %s' % (self.episode, str(time.time() - self.start_time)))
                self.logger.info('Training Stats: lr: %f  epsilon: %f steps: %d ' % (self.lr_adjusted, self.eps, self.step))
                self.logger.info('Training Stats: total_reward: %f  total_avg_reward: %f run_avg_reward: %f ' % (total_reward, total_avg_reward, run_avg_reward))
            if self.step > self.steps:
                self.logger.info('Maximal steps reached. Training stop!')
                break
        self.logger.info('Saving model...')
        self._save_model(self.step, episode_reward)

    def test_model(self):
        if not self.model:
            self._init_model(training=False)
        for episode in range(self.test_nepisodes):
            episode_reward = 0
            episode_steps = 0
            self.experience = self.env.reset()
            while not self.env.episode_ended:
                action = self._forward(self.experience[3])
                self.experience = self.env.step(action)
                self._visualize(visualize=True)
                episode_steps += 1
                episode_reward += self.experience[2]
                if episode_reward > self.env.solved_criteria:
                    self.logger.info('Test episode %d at %d step with reward %f ' % (episode, episode_steps, episode_reward))
