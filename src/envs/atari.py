from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
import nump as np
import gym

from base import Env
from utils.helpers import preprocessAtari


class AtariEnv(Env):  # low dimensional observations
    def __init__(self, **kwargs):
        super(AtariEnv, self).__init__(**kwargs)

        assert self.env_type == "gym"

        self.env = gym.make(self.game)
        self.env.seed(self.seed)  # NOTE: so each env would be different

        # action space setup
        self.actions = kwargs.get('actions', range(self.action_dim))
        self.logger.info("Action Space: %s", self.actions)

        # state space setup
        self.scale_factor = kwargs.get('scale_factor', 2)
        self.preprocess_mode = kwargs.get('preprocess_mode', 0)
        self.logger.info("State Space: %s", self.state_shape)

        # POMDP setup
        self.pomdp = kwargs.get('pomdp', False)
        self.pomdp_prob = kwargs.get('pomdp_prob', 0.5)

        self.reset()

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

    def step(self, action):
        self.exp_action = self.actions[action]
        self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        self.exp_state1 = preprocessAtari(self.exp_state1)
        if self.pomdp and np.random.rand() > self.pomdp_prob:
            self.exp_state1 = np.zeros(self.exp_state1.shape)
        self.seq_state0.append(self.seq_state1[-1])
        self.seq_state1.append(self.exp_state1)
        return self._get_experience()

    @property
    def state_shape(self):
        return self.seq_state1[0].shape
