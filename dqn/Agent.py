"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import random
import numpy as np
from dqn.ReplayMemory import ReplayMemory


class Agent(object):
    def __init__(self, emu, net, config):
        self.mem = ReplayMemory(config)
        self.emu = emu
        self.net = net

        self.rom_name = config['rom_name']

        self.eps = config['eps']
        self.eps_decay = config['eps_decay']
        self.eps_min = config['eps_min']

        self.train_start = config['train_start']
        self.train_frames = config['train_frames']
        self.update_freq = config['update_freq']
        self.test_freq = config['test_freq']
        self.test_frames = config['test_frames']

        self.target_sync = config['target_sync']

        self.max_reward = -np.inf
        self.steps = 0

    def get_eps(self):
        return max(self.eps_min, self.eps - (self.eps_decay * self.steps))

    def next(self, action):
        reward = self.emu.act(self.emu.actions[action])
        # clip reward
        clipped_reward = reward
        if reward > 1.0:
            clipped_reward = 1.0
        elif reward < -1.0:
            clipped_reward = -1.0
        screen = self.emu.get_screen_gray()
        t = self.emu.terminal()
        self.mem.add(screen, action, clipped_reward, t)
        if t:
            self.emu.new_random_game()
        return reward, t

    def eps_greedy(self):
        if random.random() < self.get_eps():
            a = random.randrange(self.emu.num_actions)
        else:
            state = self.mem.get_current()
            a = self.net.predict(state)

        reward, t = self.next(a)
        return reward, t

    def greedy(self):
        if random.random() < 0.05:  # 5% random
            a = random.randrange(self.emu.num_actions)
        else:
            state = self.mem.get_current()
            a = self.net.predict(state)

        reward, t = self.next(a)
        return reward, t

    def train(self):
        for i in xrange(int(self.train_start)):  # wait for replay memory to fill
            self.next(random.randrange(self.emu.num_actions))
        while self.steps < self.train_frames:
            if self.steps % self.target_sync == 0:
                self.net.sync_target()
            if self.steps % self.test_freq == 0:
                self.test()

            self.eps_greedy()

            if self.steps % self.update_freq == 0:
                s, a, r, ns, t = self.mem.get_minibatch()
                a = self.emu.onehot_actions(a)  # necessary due to tensorflow not having proper indexing
                cost = self.net.train(s, a, r, ns, t)

            self.steps += 1

    def test(self):
        test_steps = 0
        self.emu.new_random_game()
        total_eps = 1
        total_r = 0
        while test_steps < self.test_frames:
            r, t = self.greedy()
            total_r += r
            if t:
                total_eps += 1
            test_steps += 1

        avg_reward = float(total_r) / total_eps
        print "{0} {1}".format(self.steps, avg_reward)

        if avg_reward > self.max_reward:
            self.net.save(self.rom_name)
            self.max_reward = avg_reward


    def test_noprint(self):
        test_steps = 0
        self.emu.new_random_game()
        total_eps = 1
        total_r = 0
        while test_steps < self.test_frames:
            r, t = self.greedy()
            total_r += r
            if t:
                total_eps += 1
            test_steps += 1

        avg_reward = float(total_r) / total_eps

        if avg_reward > self.max_reward:
            self.max_reward = avg_reward

        return avg_reward


