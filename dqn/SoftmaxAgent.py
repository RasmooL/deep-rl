"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import random
import numpy as np
from scipy import stats
from dqn.ReplayMemory import ReplayMemory


class Agent(object):
    def __init__(self, emu, net, config):
        self.mem = ReplayMemory(config)
        self.emu = emu
        self.net = net

        self.rom_name = config['rom_name']

        self.train_start = config['train_start']
        self.train_frames = config['train_frames']
        self.update_freq = config['update_freq']
        self.test_freq = config['test_freq']
        self.test_frames = config['test_frames']

        self.target_sync = config['target_sync']

        self.max_reward = -np.inf
        self.steps = 0

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

    def act_softmax(self):
        # get softmax Q values
        state = self.mem.get_current()
        softmax_Q = self.net.predict(state)


        # sample according to softmax distribution
        xk = np.arange(self.emu.num_actions)
        dist = stats.rv_discrete(values=(xk, softmax_Q))
        a = dist.rvs(size=1)

        reward, t = self.next(a)
        return reward, t

    def act_greedy(self):
        # get softmax Q values
        state = self.mem.get_current()
        softmax_Q = self.net.predict(state)

        # act according to argmax Q (with 5% randomness to enable comparison with DQN)
        if random.random() < 0.05:
            a = random.randrange(self.emu.num_actions)
        else:
            a = np.argmax(softmax_Q)

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

            self.act_softmax()

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
            r, t = self.act_greedy()
            total_r += r
            if t:
                total_eps += 1
            test_steps += 1

        avg_reward = float(total_r) / total_eps
        print "{0} {1}".format(self.steps, avg_reward)

        if avg_reward > self.max_reward:
            self.net.save(self.rom_name)
            self.max_reward = avg_reward




