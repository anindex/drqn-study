from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import warnings
import random
from collections import namedtuple
# This is to be understood as a transition: Given `s0`, performing `a`
# yields `r` and results in `s1`, which might be `terminal`.
# NOTE: used as the return format for Env(), and as the format to push into replay memory for off-policy methods (DQN)
Experience = namedtuple('Experience', 's0, a, r, s1, t1')


def sample_batch_indexes(low, high, size):
    '''
    Use sample without replacement by random.sample
    '''
    if high - low >= size:
        r = range(low, high)
        batch_idxs = random.sample(r, size)
    else:
        warnings.warn('Not enough entries to sample without replacement. '
                      'Consider increasing your exploration phase to avoid oversampling!')
        batch_idxs = np.random.random_integers(low, high - 1, size=size)
    assert len(batch_idxs) == size
    return batch_idxs


def zeroed_observation(observation):
    if hasattr(observation, 'shape'):
        return np.zeros(observation.shape)
    elif hasattr(observation, '__iter__'):
        out = []
        for x in observation:
            out.append(zeroed_observation(x))
        return out
    else:
        return 0.


class Memory(object):
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 100000)

    def sample(self):
        raise NotImplementedError()

    def sample_batch(self):
        raise NotImplementedError()

    def add(self, experience):
        raise NotImplementedError()

    def get_config(self):
        return {'size': self.size}
