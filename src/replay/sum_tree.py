import numpy


class SumTree:
    def __init__(self, size):
        self.size = size
        self.clear()

    def clear(self):
        self.tree = numpy.zeros(2 * self.size - 1)
        self.data = numpy.zeros(self.size, dtype=object)
        self.write = 0
        self.num_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.size - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.size:
            self.write = 0
        if self.num_entries < self.size:
            self.num_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.size + 1
        return idx, self.tree[idx], self.data[dataIdx]
