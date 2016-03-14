"""
Asynchronous advantage actor-critic as in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al.


Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. Se the LICENSE.txt file for details.
"""

import numpy as np
import threading


class Agent(object):
    def __init__(self, emu, net, config):
        self.emu = emu
        self.net = net

        self.rom_name = config['rom_name']

        self.eps = config['eps']
        self.eps_decay = config['eps_decay']
        self.eps_min = config['eps_min']

        self.train_start = config['train_start']
        self.train_frames = config['train_frames']
        self.test_freq = config['test_freq']
        self.test_frames = config['test_frames']

        self.target_sync = config['target_sync']

        self.num_threads = config['num_threads']

        self.max_reward = -np.inf

    def train(self):
        for i in range(self.num_threads):
            A3cThread().start()
        main_thread = threading.currentThread()
        for t in threading.enumerate():
            if t is main_thread:
                continue
            t.join()


class A3cThread(threading.Thread):
    def run(self):
        steps = 0






