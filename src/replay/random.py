from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
import numpy as np

from src.replay.sum_tree import SumTree
from src.replay.base import Memory


class RandomMemory(Memory):
    def __init__(self, **kwargs):
        super(RandomMemory, self).__init__(**kwargs)
        self.memory = SumTree(self.size)
        self.e = kwargs.get('e', 0.01)
        self.a = kwargs.get('a', 0.6)  # a=0 means uniformly sampling, a=1 means fully prioritized sampling
        self.b = kwargs.get('b', 0.4)  # b=0 means no correction, b=1 means fully corrected importance sampling

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def sample_batch(self, batch_size=1):
        if self.memory.num_entries == 0:
            raise ValueError('Random memory is empty, could not sample!')
        batch = []
        idxs = []
        prios = []
        segment = self.memory.total() / batch_size
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            batch.append(data)
            idxs.append(idx)
            prios.append(p)
        # compute importance sampling weights
        sampling_prob = np.array(prios) / self.memory.total()
        weights = np.power(self.memory.num_entries * sampling_prob, -self.b)
        weights /= weights.max()
        return batch, idxs, weights

    def add(self, experience, error=0.):
        p = self._get_priority(error)
        self.memory.add(p, experience)

    def update(self, idx, error=0.):
        p = self._get_priority(error)
        self.memory.update(idx, p)

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return self.memory.num_entries
