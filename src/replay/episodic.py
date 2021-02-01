from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
from collections import deque

from src.replay.base import Memory, sample_batch_indexes


class EpisodicMemory(Memory):
    def __init__(self, **kwargs):
        super(EpisodicMemory, self).__init__(**kwargs)
        # Max number of transitions possible will be the memory capacity, could be much less
        self.max_episode_length = kwargs.get('max_episode_length', 0)
        self.num_episodes = self.size // self.max_episode_length if self.max_episode_length > 0 else self.size
        self.memory = deque(maxlen=self.num_episodes)
        self.reset()

    def add(self, experience):
        self.memory[self.idx].append(experience)
        # Terminal states are saved with actions as None, so switch to next episode
        if experience[4]:
            self.memory.append([])
            self.idx = min(self.idx + 1, self.num_episodes - 1)

    def add_episode(self, episode):
        if not self.memory[self.idx] or not self.memory[self.idx][-1].t1:
            self.memory.pop()
            self.idx = min(self.idx + 1, self.num_episodes - 1)
        else:
            self.idx = min(self.idx + 2, self.num_episodes - 1)
        self.memory.append(episode)
        self.memory.append([])

    # Samples random trajectory
    def sample(self, idx=None):
        if idx is None:
            idx = random.randrange(0, len(self.memory) - 1)  # -1 because newest traj maybe empty
        mem = self.memory[idx]
        T = len(mem)
        if self.max_episode_length > 0 and T > self.max_episode_length + 1:
            t = random.randrange(T - self.max_episode_length - 1)  # Include next state after final "maxlen" state
            return mem[t:t + self.max_episode_length + 1]
        return mem

    # Samples batch of trajectories, truncating them to the same length
    def sample_batch(self, batch_size, truncated=True, batch_idxs=None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indexes(0, len(self.memory) - 1, batch_size)
        batch = [self.sample(idx=idx) for idx in batch_idxs]
        if not truncated:
            return batch
        minimum_size = min(len(trajectory) for trajectory in batch)
        batch = [trajectory[:minimum_size] for trajectory in batch]  # Truncate trajectories
        return batch

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
