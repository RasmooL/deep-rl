"""
Asynchronous advantage actor-critic as in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import numpy as np
from ActorCritic import ActorCritic
from core.ALEEmulator import ALEEmulator
import threading


class Agent(object):
    def __init__(self, config):
        self.net = ActorCritic(config)

        self.config = config

        self.max_reward = -np.inf
        self.shared_counter = 0

    def train(self):
        for i in range(self.config['num_threads']):
            A3cThread(i, self.net, self.shared_counter, self.config).start()
        main_thread = threading.currentThread()
        for t in threading.enumerate():
            if t is main_thread:
                continue
            t.join()


class A3cThread(threading.Thread):
    def __init__(self, id, net, shared_counter, config,
                 group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
        super(A3cThread, self).__init__(group, target, name, args, kwargs, verbose)
        self.shared_counter = shared_counter  # shared counter across threads
        self.max_updates = config['max_updates']
        self.n_step = config['n_step']
        self.discount = config['discount']
        self.emu = ALEEmulator(config)  # unique emulator per thread
        self.shared_net = net  # the net we apply gradients to
        self.net = net.get_copy("thread" + str(id))  # network copy does not share weights
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def run(self):
        t = 0
        while self.shared_counter < self.max_updates:
            # TODO: sync weights and use thread-copy network

            # act until terminal or we did 'n_step' steps
            t_start = t
            # TODO: state is config['state_frames'] long, not 1 frame
            self.states.append(self.emu.get_screen_gray())
            while t - t_start < self.n_step:
                self.actions[t], self.values[t] = self.net.predict(self.states[t])
                self.rewards[t], self.states[t+1] = self.next(self.actions[t])
                terminal = self.emu.terminal()
                t += 1
                self.shared_counter += 1
                if terminal:
                    break

            # Construct R, sum of discounted rewards for each step
            steps_done = t - t_start
            R = np.empty(steps_done)
            # bootstrap if last state not terminal
            R[-1] = 0 if terminal else self.rewards[t-1] + self.discount*self.values[t]
            for i in range(steps_done-1):  # [t_start ... t-2] but shifted to start at 0
                R[-i-2] = self.rewards[t-i-2] + self.discount * R[-i-1]

            # apply gradients (tensorflow handles the 'batch', no need to accumulate and then update, I think)
            cost_V, cost_P = self.net.train(self.states[t_start:t],
                                            self.actions[t_start:t],
                                            R)

    def next(self, action):
        reward = self.emu.act(action)
        screen = self.emu.get_screen_gray()

        return reward, screen







