"""
Experimental agent implementation running separate threads for emulation and GPU training.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. Se the LICENSE.txt file for details.
"""

from Agent import Agent
import random
import threading
import time
import numpy as np


class ParallelAgent(Agent):
    def __init__(self, emu, net, config):
        super(ParallelAgent, self).__init__(emu, net, config)
        self.gpu_lock = threading.Lock()
        self.testing = False

    def train(self):
        cpu = threading.Thread(target=self.ale_worker)
        cpu.setDaemon(True)
        gpu_1 = threading.Thread(target=self.gpu_worker)
        gpu_2 = threading.Thread(target=self.gpu_worker)

        for i in xrange(int(self.train_start)):  # wait for replay memory to fill
            self.next(random.randrange(self.emu.num_actions))
        cpu.start()
        gpu_1.start()
        gpu_2.start()

        gpu_1.join()
        gpu_2.join()
        return

    def test(self):
        self.testing = True
        time.sleep(0.5)  # wait a bit for ALE worker to stop
        super(ParallelAgent, self).test()
        self.testing = False

    def ale_worker(self):
        """
        Performs epsilon greedy action selection, updating the replay memory and emulating with ALE.
        """

        while True:
            if self.testing:
                time.sleep(0.2)
                continue
            self.eps_greedy()

    def gpu_worker(self):
        """
        Gathers a minibatch (on the CPU!) and feeds it to the GPU. Several can run at once, locking the GPU.
        """
        while self.steps < self.train_frames:
            s, a, r, ns, t = self.mem.get_minibatch()  # TODO: ReplayMemory is _not_ thread safe
            a = self.emu.onehot_actions(a)  # necessary due to tensorflow not having proper indexing

            with self.gpu_lock:
                cost = self.net.train(s, a, r, ns, t)

                if self.steps % self.target_sync == 0:
                    self.net.sync_target()
                if self.steps % self.test_freq == 0:
                    self.test()
                if self.steps % self.save_freq == 0:
                    self.net.save(self.steps)

                self.steps += 1
                if self.steps % 100 == 0:  # TODO: remove, just for debugging
                    print 'step ' + str(self.steps)





