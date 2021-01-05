from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
from torch.nn import MSELoss  # noqa
from torch.nn.functional import smooth_l1_loss  # noqa
from torch.optim import Adam, Adagrad  # noqa


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
        self.model_prototype = model_prototype
        self.model_params = kwargs.get('model')
        self.memory_prototype = memory_prototype
        self.memory_params = kwargs.get('memory')
        # params
        self.model_name = self.model_params.get('model_name', 'no_name')
        self.model_file = self.model_params.get('model_file', None)
        self.visualize = kwargs.get('visualize', False)
        self.save_best = kwargs.get('save_best', True)
        if self.save_best:
            self.best_step = None  # NOTE: achieves best_reward at this step
            self.best_reward = None  # NOTE: only save a new model if achieves higher reward
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
        self.target_model_update = kwargs.get('target_model_update', 1000)  # update every # steps
        self.batch_size = kwargs.get('batch_size', 32)
        self.enable_double_dqn = kwargs.get('enable_double_dqn', False)
        # count step
        self.step = 0

    def _load_model(self, model_file):
        if model_file is not None:
            self.model.load_state_dict(torch.load(model_file))
            self.logger.info('Loaded  Model: ' + self.model_file)
        else:
            self.logger.info('No pretrained Model. Will train from scratch.')

    def _save_model(self, step, curr_reward):
        if self.save_best:
            if self.best_step is None:
                self.best_step = step
                self.best_reward = curr_reward
            if curr_reward >= self.best_reward:
                self.best_step = step
                self.best_reward = curr_reward
                torch.save(self.model.state_dict(), self.model_name)
            self.logger.info('Saved model: %s at best steps: %d and best reward: %d '
                             % (self.model_name, self.best_step, self.best_reward))
        else:
            torch.save(self.model.state_dict(), self.model_name)
            self.logger.info('Saved model: %s after %d steps: ' % (self.model_name, step))

    def _forward(self, observation):
        raise NotImplementedError()

    def _backward(self, reward, terminal):
        raise NotImplementedError()

    def _eval_model(self):  # evaluation during training
        raise NotImplementedError()

    def fit_model(self):    # training
        raise NotImplementedError()

    def test_model(self):   # testing pre-trained models
        raise NotImplementedError()

    @property
    def dtype():
        raise NotImplementedError()
