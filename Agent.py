from ReplayMemory import ReplayMemory
import random
import numpy as np

class Agent(object):
    def __init__(self, emu, net, config):
        self.mem = ReplayMemory(config)
        self.emu = emu
        self.net = net

        self.eps = config['eps']
        self.eps_decay = config['eps_decay']
        self.eps_min = config['eps_min']

        self.steps = 0

    def get_eps(self):
        return max(self.eps_min, self.eps - (self.eps_decay * self.steps))

    def next(self, action):
        reward = self.emu.act(action)
        screen = self.emu.get_screen_gray()
        t = self.emu.terminal()
        self.mem.add(screen, action, reward, t)
        self.steps += 1
        return reward

    def eps_greedy(self):
        if random.random() < self.get_eps():
            a = random.randrange(self.emu.num_actions)
        else:
            state = self.mem.get_current()
            Q = self.net.predict(state)
            assert len(Q) == self.emu.num_actions
            a = np.argmax(Q)





