from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.spaces.box import Box

from src.replay.base import Experience


class Env(object):
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger', logging.getLogger(__name__))
        self.idx = kwargs.get('env_idx', 0)  # NOTE: for creating multiple environment instances
        # general setup
        self.mode = kwargs.get('mode', 0)  # NOTE: save frames when mode=1
        if self.mode == 1:
            try:
                import scipy.misc
                self.imsave = scipy.misc.imsave
            except ImportError:
                self.logger.warning("WARNING: scipy.misc not found, use plt.imsave")
                self.imsave = plt.imsave
            self.img_dir = join(kwargs.get('root_dir', '~'), "imgs")
            self.logger.info("Frames will be saved to: " + self.img_dir)
            self.frame_idx = 0
        self.seed = kwargs.get('seed', 2020) + self.idx  # NOTE: so to give a different seed to each instance
        self.seq_len = kwargs.get('seq_len', 4)
        self.solved_criteria = kwargs.get('solved_criteria', 100)  # score
        # POMDP setup
        self.pomdp = kwargs.get('pomdp', False)
        self.pomdp_type = kwargs.get('pomdp_type', 'flickering')
        self.pomdp_mask = np.array(kwargs.get('pomdp_mask', []))
        self.pomdp_prob = kwargs.get('pomdp_prob', 0.5)

        self._reset_experience()

    def _reset_experience(self):
        self.exp_action = None
        self.exp_reward = None
        self.exp_terminal1 = None
        self.seq_state0 = deque(maxlen=self.seq_len)
        self.seq_state1 = deque(maxlen=self.seq_len)

    def _preprocessStates(self, states):  # NOTE: padding zeros state if size is less than seq_len
        if not states:
            return np.zeros([self.seq_len, *self.state_shape])
        states = np.array(states)
        if states.shape[0] < self.seq_len:
            states = np.append(np.zeros([self.seq_len - states.shape[0], *self.state_shape]), states, axis=0)
        return states

    def _get_experience(self):
        return Experience(s0=self._preprocessStates(self.seq_state0),
                          a=self.exp_action,
                          r=self.exp_reward,
                          s1=self._preprocessStates(self.seq_state1),
                          t1=self.exp_terminal1)

    def render(self):  # render using the original gl window
        raise NotImplementedError()

    def visual(self):  # visualize onto visdom
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()

    @property
    def state_shape(self):
        raise NotImplementedError()

    @property
    def action_dim(self):  # for now assuming discrete control
        if isinstance(self.env.action_space, Box):
            return self.env.action_space.shape[0]
        else:
            return self.env.action_space.n
