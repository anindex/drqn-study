from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
import numpy as np
import gym

from src.envs.base import Env
from src.utils.helpers import preprocessAtari


class AtariEnv(Env):  # low dimensional observations
    def __init__(self, **kwargs):
        super(AtariEnv, self).__init__(**kwargs)
        self.env_type = 'atari'
        self.game = kwargs.get('game', 'Breakout-v0')
        self.env = gym.make(self.game)
        self.env.seed(self.seed)  # NOTE: so each env would be different

        # state & action space setup
        self.actions = kwargs.get('actions', range(self.action_dim))
        self.scale_factor = kwargs.get('scale_factor', 2)
        self.preprocess_mode = kwargs.get('preprocess_mode', 0)
        self.reset()
        self.logger.info("Action Space: %s", self.actions)
        self.logger.info("State Space: %s", self.state_shape)
        self.last_lives = 0

        # atari POMDP
        self.pomdp_mask = np.ones((1, *self.state_shape[1:]))
        hide_pixels = int(self.state_shape[1] * (1 - self.pomdp_prob))
        low, high = int((self.state_shape[1] - hide_pixels) / 2), int((self.state_shape[1] + hide_pixels) / 2)
        self.pomdp_mask[:, low:high, :] = 0

    def render(self):
        if self.mode == 2:
            frame = self.env.render(mode='rgb_array')
            frame_name = join(self.img_dir, "frame_%04d.jpg" % self.frame_idx)
            self.imsave(frame_name, frame)
            self.frame_idx += 1
            return frame
        else:
            return self.env.render()

    def visual(self):
        pass

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self._reset_experience()
        self.seq_state1.append(preprocessAtari(self.env.reset()))
        self.last_lives = 0
        self.episode_ended = False
        return self._get_experience()

    def step(self, action):
        self.exp_action = self.actions[action]
        self.exp_state1, self.exp_reward, terminal, info = self.env.step(self.exp_action)
        # augmenting telling lost life is bad
        self.episode_ended = terminal
        if info['ale.lives'] < self.last_lives:
            self.exp_terminal1 = True
        else:
            self.exp_terminal1 = terminal
        self.last_lives = info['ale.lives']
        self.exp_state1 = preprocessAtari(self.exp_state1)
        if self.pomdp:
            if self.pomdp_type == 'flickering':
                if np.random.rand() > self.pomdp_prob:
                    self.exp_state1 = np.zeros(self.exp_state1.shape)
            elif self.pomdp_type == 'delete_dim':
                self.exp_state1 = np.array(self.exp_state1) * self.pomdp_mask
        self.seq_state0.append(self.seq_state1[-1])
        self.seq_state1.append(self.exp_state1)
        return self._get_experience()

    @property
    def state_shape(self):
        return self.seq_state1[0].shape
