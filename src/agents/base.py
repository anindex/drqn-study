from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import numpy as np
import random
from collections import deque
import torch
from os.path import exists
from torch.nn.functional import smooth_l1_loss, mse_loss  # noqa
from torch.optim import Adam, Adagrad  # noqa
from tensorboardX import SummaryWriter


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Agent(object):
    def __init__(self, env_prototype, model_prototype, memory_prototype=None, **kwargs):
        # env
        self.env_prototype = env_prototype
        self.env_params = kwargs.get('env')
        self.env = self.env_prototype(**self.env_params)
        self.state_shape = self.env.state_shape
        self.action_dim = self.env.action_dim
        # model
        self.model_prototype = model_prototype
        self.model_params = kwargs.get('model')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_params['use_cuda'] = torch.cuda.is_available()
        self.model_params['stack_len'] = self.env_params['stack_len']
        self.model_params['state_shape'] = (self.model_params['stack_len'] * self.state_shape[0], *self.state_shape[1:])  # state_shape in model: (stack_len * C, H, W)
        self.model_params['action_dim'] = self.action_dim
        self.model = None
        # memory
        self.memory_prototype = memory_prototype
        self.memory_type = kwargs.get('memory_type', 'random')
        self.memory_params = kwargs.get('memory')
        self.memory = None
        random.seed(self.env_params['seed'])
        # logging
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        self.model_file = self.model_params.get('model_file', None)
        self.log_folder = kwargs.get('log_folder', 'logs')
        self.use_tensorboard = kwargs.get('use_tensorboard', True)
        if self.use_tensorboard:
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
        self.log_window_size = kwargs.get('log_window_size', 100)
        self._reset_training_loggings()
        # agent_params
        # criteria and optimizer
        self.value_criteria = eval(kwargs.get('value_criteria', 'mse_loss'))
        self.optimizer_class = eval(kwargs.get('optimizer', 'Adam'))
        # hyperparameters
        self.steps = kwargs.get('steps', 100000)
        self.random_eps = kwargs.get('random_eps', 50)
        self.learn_start = kwargs.get('learn_start', 1000)  # num steps to fill the memory
        self.gamma = kwargs.get('gamma', 0.99)
        self.clip_grad = kwargs.get('clip_grad', float('inf'))
        self.lr = kwargs.get('lr', 0.0001)
        self.lr_decay = kwargs.get('lr_decay', False)
        self.weight_decay = kwargs.get('weight_decay', 0.)
        self.eps_start = kwargs.get('eps_start', 1.0)
        self.eps_decay = kwargs.get('eps_decay', 50000)  # num of decaying steps
        self.eps_end = kwargs.get('eps_end', 0.1)
        self.prog_freq = kwargs.get('prog_freq', 2500)
        self.train_interval = kwargs.get('train_interval', 1)
        self.memory_interval = kwargs.get('memory_interval', 1)
        self.action_repetition = kwargs.get('action_repetition', 1)
        self.test_nepisodes = kwargs.get('test_nepisodes', 1)
        self.target_model_update = kwargs.get('target_model_update', 1000)  # update every # steps
        self.batch_size = kwargs.get('batch_size', 32)
        self.bootstrap_type = kwargs.get('bootstrap_type', 'double_q')
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
        self.window_scores = deque(maxlen=self.log_window_size)
        self.window_max_abs_q = deque(maxlen=self.log_window_size)
        self.max_abs_q_log = [0]  # per step
        self.loss_log = [0]  # per step
        self.total_avg_score_log = [0]  # per eps
        self.run_avg_score_log = [0]  # per eps
        self.step_log = [0]
        self.eps_log = [0]

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

    def _store_episode(self, episode):
        if self.memory_type == 'episodic':
            self.memory.add_episode(episode)
        else:
            self.logger.warn('Only episodic memory can add episode!')

    def _store_experience(self, experience, error=0.):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            if self.memory_type == 'episodic':
                self.memory.add(experience)
            elif self.memory_type == 'random':
                self.memory.add(experience, error)
            else:
                raise ValueError('Memory type %s is unsupported!' % self.memory_type)

    # Hard update every `target_model_update` steps.
    def _update_target_model_hard(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
    def _update_target_model_soft(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.target_model_update * local_param.data + (1.0 - self.target_model_update) * target_param.data)

    def _epsilon_greedy(self, q_values):
        self.eps = self.eps_end + max(0, (self.eps_start - self.eps_end) * (self.eps_decay - max(0, self.step - self.learn_start)) / self.eps_decay)
        # choose action
        if np.random.uniform() < self.eps:  # then we choose a random action
            action = random.randrange(self.action_dim)
        else:                               # then we choose the greedy action
            if self.model_params['use_cuda']:
                action = np.argmax(q_values.cpu().numpy())
            else:
                action = np.argmax(q_values.numpy())
        return action

    def _create_zero_lstm_hidden(self, batch_size=1):
        return (torch.zeros(self.model.num_lstm_layer, batch_size, self.model.hidden_dim).type(self.dtype).to(self.device),
                torch.zeros(self.model.num_lstm_layer, batch_size, self.model.hidden_dim).type(self.dtype).to(self.device))

    def _get_loss(self, experiences):
        raise NotImplementedError()

    def _forward(self, states):
        raise NotImplementedError()

    def _backward(self, experience):
        raise NotImplementedError()

    def fit_model(self):    # training
        raise NotImplementedError()

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError()

    def set_seed(self, seed=0):
        random.seed(seed)
        self.env.env.seed(seed)

    @property
    def dtype(self):
        return self.model.dtype
