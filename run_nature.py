"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import sys
import time
import random
import cv2
from sacred import Experiment
from core.ALEEmulator import ALEEmulator
from dqn.Agent import Agent
from dqn.NatureDQN import NatureDQN

ex = Experiment('nature')

@ex.config
def net_config():
    conv_layers = 3
    conv_units = [32, 64, 64]
    filter_sizes = [8, 4, 3]
    strides = [4, 2, 1]
    state_frames = 4
    fc_layers = 1
    fc_units = [512]
    in_width = 84
    in_height = 84
    discount = 0.99
    device = '/gpu:0'
    lr = 0.00025
    opt_decay = 0.95
    momentum = 0.0
    opt_eps = 0.01
    target_sync = 1e4
    clip_delta = 1.0
    tensorboard = False
    tensorboard_freq = 50
    ckpt = 0


@ex.config
def emu_config():
    rom_path = '../ale-git/roms/'
    rom_name = 'breakout'
    display_screen = False
    frame_skip = 4
    repeat_prob = 0.0
    color_avg = True
    random_seed = 42
    random_start = 30


@ex.config
def agent_config():
    hist_size = 1e6
    eps = 1.0
    eps_decay = 9e-7
    eps_min = 0.1
    batch_size = 32
    train_start = 5e4
    train_frames = 5e7
    test_freq = 2.5e5
    test_frames = 2e4
    update_freq = 4

@ex.command
def covar(_config):
    import tensorflow as tf
    import numpy as np
    import scipy.misc as sp
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = NatureDQN(_config)
    net.load(_config['rom_name'])

    with tf.variable_scope('conv0', reuse=True):
        weight = net.sess.run(tf.get_variable('weight'))
        weight.shape = (8*8*4, 32)
        sp.imsave('covar.png', sp.imresize(np.cov(weight.T), 8.0, 'nearest'))





@ex.command
def visualize(_config):
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = NatureDQN(_config)
    net.load(_config['rom_name'])
    agent = Agent(emu, net, _config)
    agent.next(0)
    cv2.startWindowThread()
    cv2.namedWindow("deconv")

    for n in range(10000):
        agent.greedy()
        recon = net.visualize(agent.mem.get_current())  # (1, W, H, N)
        cv2.imshow("deconv", cv2.resize(recon[0, :, :, 1:4], (84*3, 84*3)))


@ex.command
def drop(_config):
    _config['drop_experiment'] = True
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions

    for layer in range(_config['conv_layers']):
        for map in range(_config['conv_units'][layer]):
            _config['drop_nlayer'] = layer
            _config['drop_nmaps'] = [map]

            net = NatureDQN(_config)
            net.load(_config['rom_name'])
            agent = Agent(emu, net, _config)
            agent.next(0)

            print "Drop {}.{}".format(layer, map)
            agent.test()
            from tensorflow.python.framework import ops
            ops.reset_default_graph()

@ex.command
def test(_config):
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = NatureDQN(_config)
    net.load(_config['rom_name'])
    agent = Agent(emu, net, _config)
    agent.next(0)  # put a frame into the replay memory, TODO: should not be necessary

    agent.test()


@ex.automain
def main(_config, _log):
    sys.stdout = open('log_' + _config['rom_name'] + time.strftime('%H%M%d%m', time.gmtime()), 'w', buffering=True)
    print "#{}".format(_config)
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = NatureDQN(_config)

    agent = Agent(emu, net, _config)

    agent.train()

