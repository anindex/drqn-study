from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import time
import torch

from src.agents.base import Agent, adjust_learning_rate


class DQNAgent(Agent):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        super(DQNAgent, self).__init__(env_prototype, model_prototype, memory_prototype, **kwargs)
        self.logger.info('<===================================> DQNAgent')
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
        self.model = self.model_prototype(name='Current Model', **self.model_params).to(self.device)
        self._load_model(self.model_file)   # load pretrained model if provided
        # target_model
        self.target_model = self.model_prototype(name='Target Model', **self.model_params).to(self.device)
        self._update_target_model_hard()
        # memory
        if memory_prototype is not None:
            self.memory = self.memory_prototype(**self.memory_params)
        # experience & states
        self._reset_experiences()

    def _reset_training_loggings(self):
        self._reset_testing_loggings()
        # during the evaluation in training, we additionally log for
        # the predicted Q-values and TD-errors on validation data
        self.v_avg_log = []
        self.tderr_avg_log = []

    def _reset_testing_loggings(self):
        # setup logging for testing/evaluation stats
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []

    def _reset_experiences(self):
        self.env.reset()
        self.memory.reset()

    # Hard update every `target_model_update` steps.
    def _update_target_model_hard(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
    def _update_target_model_soft(self):
        for i, (key, target_weights) in enumerate(self.target_model.state_dict().iteritems()):
            target_weights += self.target_model_update * self.model.state_dict()[key]

    def _visualize(self):
        if self.visualize:
            self.env.visual()
            self.env.render()

    def _get_q_update(self, experiences):
        s0_batch_vb = torch.from_numpy(np.array([experiences[i].s0 for i in range(len(experiences))])).type(self.dtype).to(self.device)
        a_batch_vb = torch.from_numpy(np.array([experiences[i].a for i in range(len(experiences))])).long().to(self.device)
        r_batch_vb = torch.from_numpy(np.array([experiences[i].r for i in range(len(experiences))])).type(self.dtype).to(self.device)
        s1_batch_vb = torch.from_numpy(np.array([experiences[i].s1 for i in range(len(experiences))])).type(self.dtype).to(self.device)
        t1_batch_vb = torch.from_numpy(np.array([0. if experiences[i].t1 else 1. for i in range(len(experiences))])).type(self.dtype).to(self.device)
        # Compute target Q values for mini-batch update.
        if self.enable_double_dqn:
            q_values_vb = self.model(s1_batch_vb).detach()    # Detach this variable from the current graph since we don't want gradients to propagate
            _, q_max_actions_vb = q_values_vb.max(dim=1, keepdim=True)
            next_max_q_values_vb = self.target_model(s1_batch_vb).detach()
            next_max_q_values_vb = next_max_q_values_vb.gather(1, q_max_actions_vb)
        else:
            next_max_q_values_vb = self.target_model(s1_batch_vb).detach()
            next_max_q_values_vb, _ = next_max_q_values_vb.max(dim=1, keepdim=True)
        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the targets accordingly
        current_q_values_vb = self.model(s0_batch_vb).gather(1, a_batch_vb.unsqueeze(1)).squeeze()
        # Set discounted reward to zero for all states that were terminal.
        next_max_q_values_vb = next_max_q_values_vb * t1_batch_vb.unsqueeze(1)
        expected_q_values_vb = r_batch_vb + self.gamma * next_max_q_values_vb.squeeze()
        td_error_vb = self.value_criteria(current_q_values_vb, expected_q_values_vb)
        return td_error_vb

    def _epsilon_greedy(self, q_values_ts):
        self.eps = self.eps_end + max(0, (self.eps_start - self.eps_end) * (self.eps_decay - max(0, self.step - self.learn_start)) / self.eps_decay)
        # choose action
        if np.random.uniform() < self.eps:  # then we choose a random action
            action = random.randrange(self.action_dim)
        else:                               # then we choose the greedy action
            if self.model_params['use_cuda']:
                action = np.argmax(q_values_ts.cpu().numpy())
            else:
                action = np.argmax(q_values_ts.numpy())
        return action

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
        self.eps = self.eps_start
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_adjusted = self.lr
        self.logger.info('<===================================> Training ...')
        self._reset_training_loggings()
        self.start_time = time.time()
        self.step = 0
        nepisodes = 0
        nepisodes_solved = 0
        total_reward = 0.
        new_eps = True
        while self.step < self.steps:
            if new_eps:    # start of a new episode
                episode_steps = 0
                episode_reward = 0.
                self.experience = self.env.reset()
                self._visualize()
                new_eps = False  # reset flag
            action = self._forward(self.experience.s1)
            self.experience = self.env.step(action)
            self._visualize()
            new_eps = bool(self.experience.t1)
            self._backward(self.experience)
            episode_steps += 1
            episode_reward += self.experience.r
            self.step += 1
            if new_eps:
                total_reward += episode_reward
                nepisodes += 1
                if self.experience.t1 and episode_reward >= self.env.solved_criteria:
                    nepisodes_solved += 1
            # report training stats
            if self.step % self.prog_freq == 0:
                self.logger.info('Reporting at %d step | Elapsed Time: %s' % (self.step, str(time.time() - self.start_time)))
                self.logger.info('Training Stats: lr: %f  epsilon: %f  total_reward: %f  avg_reward: %f ' % (self.lr_adjusted, self.eps, total_reward, total_reward/nepisodes if nepisodes > 0 else 0.))
                self.logger.info('Training Stats: nepisodes: %d  nepisodes_solved: %d  repisodes_solved: %f ' % (nepisodes, nepisodes_solved, nepisodes_solved/nepisodes if nepisodes > 0 else 0.))

    @property
    def dtype(self):
        return self.model.dtype
