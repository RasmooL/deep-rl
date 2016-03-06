import numpy as np


class ReplayMemory(object):
    def __init__(self, config):
        self.width = config['in_width']
        self.height = config['in_height']
        self.max_size = config['hist_size']
        self.states = np.empty([self.max_size, self.width, self.height, 4], dtype=np.uint8)
        self.actions = np.empty([self.max_size], dtype=np.uint8)
        self.rewards = np.empty([self.max_size], dtype=np.uint32)
        self.nstates = np.empty([self.max_size, self.width, self.height, 4], dtype=np.uint8)
        self.terminals = np.empty([self.max_size], dtype=np.bool)

        self.count = 0

    def get_batch(self, size):
        assert self.count > 0, "Replay memory is empty."

        rands = np.random.randint(low=0, high=min(self.count, self.max_size), size=(size))
        s = self.states[rands, :, :, :]
        a = self.actions[rands]
        r = self.rewards[rands]
        ns = self.nstates[rands, :, :, :]
        t = self.terminals[rands]

        return s, a, r, ns, t

    def add(self, s, a, r, ns, t):
        index = self.count % self.max_size
        self.states[index, :, :, :] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.nstates[index, :, :, :] = ns
        self.terminals[index] = t
        self.count += 1

