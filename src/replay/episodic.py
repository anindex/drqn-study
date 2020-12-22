from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from collections import deque

from base import Memory, Experience, sample_batch_indexes


class EpisodicMemory(Memory):
    def __init__(self, size, max_episode_length):
        super(EpisodicMemory, self).__init__(size)
        # Max number of transitions possible will be the memory capacity, could be much less
        self.max_episode_length = max_episode_length
        self.num_episodes = size // max_episode_length
        self.memory = deque(maxlen=self.num_episodes)
        self.reset()

    def append(self, s0, a, r, s1, t1):
        self.memory[self.idx].append(Experience(s0, a, r, s1, t1))
        # Terminal states are saved with actions as None, so switch to next episode
        if t1:
            self.memory.append([])
            self.idx = min(self.idx + 1, self.num_episodes - 1)

    # Samples random trajectory
    def sample(self, maxlen=0, idx=None):
        if idx is None:
            idx = random.randrange(0, len(self.memory) - 1)  # -1 because newest traj maybe empty
        mem = self.memory[idx]
        T = len(mem)
        # Take a random subset of trajectory if maxlen specified, otherwise return full trajectory
        if maxlen > 0 and T > maxlen + 1:
            t = random.randrange(T - maxlen - 1)  # Include next state after final "maxlen" state
            return mem[t:t + maxlen + 1]
        return mem

    # Samples batch of trajectories, truncating them to the same length
    def sample_batch(self, batch_size, maxlen=0, truncated=True, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, len(self.memory) - 1, batch_size)
        batch = [self.sample(maxlen=maxlen, idx=idx) for idx in batch_idxs]
        if not truncated:
            return batch
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
        return list(map(list, zip(*batch)))  # Transpose so that timesteps are packed together

    def __len__(self):
        return sum(len(episode) for episode in self.memory)

    def reset(self):
        self.memory.clear()
        self.memory.append([])  # List for first episode
        self.idx = 0

    def get_config(self):
        config = super(EpisodicMemory, self).get_config()
        config.update({
            'num_episodes': self.num_episodes,
            'max_episode_length': self.max_episode_length
        })
        return config
