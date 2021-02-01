from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
import numpy as np
import gym

from src.envs.base import Env


class GymEnv(Env):  # low dimensional observations
    def __init__(self, **kwargs):
        super(GymEnv, self).__init__(**kwargs)
        self.env_type = 'gym'
        self.game = kwargs.get('game', 'CartPole-v1')
        self.env = gym.make(self.game)
        self.env.seed(self.seed)  # NOTE: so each env would be different

        # action space setup
        self.actions = kwargs.get('actions', range(self.action_dim))
        self.logger.info("Action Space: %s", self.actions)

        # state space setup
        self.logger.info("State  Space: %s", self.state_shape)

        # continuous space
        self.enable_continuous = kwargs.get('enable_continuous', False)

        self.reset()

    def render(self):
        if self.mode == 1:
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
        self.seq_state1.append(self.env.reset())
        self.episode_ended = False
        return self._get_experience()

    def step(self, action):
        self.exp_action = action if self.enable_continuous else self.actions[action]
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        self.episode_ended = self.exp_terminal1
        if self.pomdp:
            if self.pomdp_type == 'flickering':
                if np.random.rand() > self.pomdp_prob:
                    self.exp_state1 = np.zeros(self.state_shape)
            elif self.pomdp_type == 'delete_dim' and self.pomdp_mask.size != 0:
                self.exp_state1 = self.pomdp_mask * np.array(self.exp_state1)
        self.seq_state0.append(self.seq_state1[0])
        self.seq_state1.append(self.exp_state1)
        return self._get_experience()

    @property
    def state_shape(self):
        return self.env.observation_space.shape
