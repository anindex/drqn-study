from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import random
import torch
from os.path import exists
from torch.nn import MSELoss  # noqa
from torch.nn.functional import smooth_l1_loss  # noqa
from torch.optim import Adam, Adagrad  # noqa
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Agent(object):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        # logging
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        # prototypes for env & model & memory
        self.env_prototype = env_prototype
        self.env_params = kwargs.get('env')
        self.env = None
        self.model_prototype = model_prototype
        self.model_params = kwargs.get('model')
        self.model = None
        self.memory_prototype = memory_prototype
        self.memory_params = kwargs.get('memory')
        self.memory = None
        # logging
        self.model_file = self.model_params.get('model_file', None)
        self.log_folder = kwargs.get('log_folder', 'logs')
        self.writer = SummaryWriter(self.log_folder)
        self.log_step_interval = kwargs.get('log_step_interval', 100)
        self.log_episode_interval = kwargs.get('log_episode_interval', 10)
        self.train_visualize = kwargs.get('train_visualize', False)
        self.save_best = kwargs.get('save_best', True)
        if self.save_best:
            self.best_step = None  # NOTE: achieves best_reward at this step
            self.best_reward = None  # NOTE: only save a new model if achieves higher reward
        self.retrain = kwargs.get('retrain', True)
        self.solved_stop = kwargs.get('solved_stop', True)
        # agent_params
        # criteria and optimizer
        self.value_criteria = eval(kwargs.get('value_criteria', 'MSELoss'))()
        self.optimizer_class = eval(kwargs.get('optimizer', 'Adam'))
        # hyperparameters
        self.steps = kwargs.get('steps', 100000)
        self.learn_start = kwargs.get('learn_start', 1000)  # num steps using random policy
        self.gamma = kwargs.get('gamma', 0.99)
        self.clip_grad = kwargs.get('clip_grad', 1.0)
        self.lr = kwargs.get('lr', 0.0001)
        self.lr_decay = kwargs.get('lr_decay', False)
        self.weight_decay = kwargs.get('weight_decay', 0.0005)
        self.eps_start = kwargs.get('eps_start', 1.0)
        self.eps_decay = kwargs.get('eps_decay', 50000)  # num of decaying steps
        self.eps_end = kwargs.get('eps_end', 0.1)
        self.prog_freq = kwargs.get('prog_freq', 2500)
        self.train_interval = kwargs.get('train_interval', 1)
        self.memory_interval = kwargs.get('memory_interval', 1)
        self.action_repetition = kwargs.get('action_repetition', 1)
        self.test_nepisodes = kwargs.get('test_nepisodes', 1)
        self.run_avg_nepisodes = kwargs.get('run_avg_nepisodes', 100)
        self.target_model_update = kwargs.get('target_model_update', 1000)  # update every # steps
        self.batch_size = kwargs.get('batch_size', 32)
        self.enable_double_dqn = kwargs.get('enable_double_dqn', False)
        # count step
        self.step = 0

    def _load_model(self):
        if self.model_file is not None and exists(self.model_file):
            self.model.load_state_dict(torch.load(self.model_file))
            self.logger.info('Loaded Model: ' + self.model_file)
        else:
            self.logger.info('No pretrained Model. Will train from scratch.')

    def _save_model(self, step, curr_reward):
        if self.model is None:
            return
        if self.save_best:
            if self.best_step is None:
                self.best_step = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_file)
            self.logger.info('Saved model: %s at best steps: %d and best reward: %d '
                             % (self.model_file, self.best_step, self.best_reward))
        else:
            torch.save(self.model.state_dict(), self.model_file)
            self.logger.info('Saved model: %s after %d steps: ' % (self.model_file, step))

    def _visualize(self, visualize=True):
        if visualize:
            self.env.visual()
            self.env.render()

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

    def _reset_experiences(self):
        self.env.reset()
        if self.memory is not None:
            self.memory.reset()

    def _init_model(self, training=False):
        self.model = self.model_prototype(name='Current Model', **self.model_params).to(self.device)
        if not self.retrain:
            self._load_model()   # load pretrained model if provided
        self.model.train(mode=training)
        if training:
            # target_model
            self.target_model = self.model_prototype(name='Target Model', **self.model_params).to(self.device)
            self._update_target_model_hard()
            self.target_model.eval()
            # memory
            if self.memory_prototype is not None:
                self.memory = self.memory_prototype(**self.memory_params)
            # experience & states
            self._reset_experiences()

    # Hard update every `target_model_update` steps.
    def _update_target_model_hard(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
    def _update_target_model_soft(self):
        for i, (key, target_weights) in enumerate(self.target_model.state_dict().iteritems()):
            target_weights += self.target_model_update * self.model.state_dict()[key]

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

    def _forward(self, observation):
        raise NotImplementedError()

    def _backward(self, reward, terminal):
        raise NotImplementedError()

    def fit_model(self):    # training
        raise NotImplementedError()

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError()

    @property
    def dtype(self):
        return self.model.dtype
