from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from collections import deque

from base import Memory, Experience, sample_batch_indexes


class RandomMemory(Memory):
    def __init__(self, size):
        super(RandomMemory, self).__init__(size)
        self.memory = deque(maxlen=self.size)

    def sample(self, idx=None):
        if idx is not None:
            return self.memory[idx]
        idx = random.randrange(len(self.memory))
        return self.memory[idx]

    def sample_batch(self, batch_size, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, len(self.memory) - 1, batch_size)
        batch = [self.sample(idx=idx) for idx in batch_idxs]
        return batch

    def append(self, s0, a, r, s1, t1):
        self.memory.append(Experience(s0, a, r, s1, t1))

    def reset(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
