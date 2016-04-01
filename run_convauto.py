"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import sys
import time

from sacred import Experiment
from core.ALEEmulator import ALEEmulator
from dqn.Agent import Agent
from continuation.OriginalNet import OriginalNet
from core.ScreenBuffer import ScreenBuffer
import numpy as np
import cv2

ex = Experiment('convauto')


@ex.config
def net_config():
    conv_layers = 3
    conv_units = [32, 64, 64]
    filter_sizes = [8, 4, 2]
    strides = [4, 2, 1]
    hidden_units = 512
    num_heads = 3
    gate_noise = 0.01
    sharpening_slope = 10
    in_width = 84
    in_height = 84
    device = '/gpu:0'
    lr = 0.0001
    opt_decay = 0.95
    momentum = 0.5
    opt_eps = 0.01
    tensorboard = False
    tensorboard_freq = 50


@ex.config
def emu_config():
    rom_path = '../ale-git/roms/'
    rom_name = 'breakout'
    display_screen = True
    frame_skip = 4
    repeat_prob = 0.0
    color_avg = True
    random_seed = 42
    random_start = 30


@ex.config
def agent_config():
    batch_size = 16
    train_start = 5e3
    train_frames = 5e6
    test_freq = 5e4
    test_frames = 5e3
    save_freq = 5e3


@ex.command
def test(_config):
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = DoubleDQN(_config)
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
    net = OriginalNet(_config)

    net.load('cont')

    cv2.startWindowThread()
    cv2.namedWindow("prediction")

    # fill screen history up to batch size
    buf = ScreenBuffer(_config, _config['batch_size'])
    for n in range(_config['batch_size']):
        emu.act(emu.actions[np.random.randint(0, emu.num_actions)])  # act randomly
        buf.insert(emu.get_screen_rgb())
    # train
    step = 100000
    while step < _config['train_frames']:
        cost = net.train(buf.get(), [step])
        print step, cost

        # predict next frame
        hidden = net.encode(buf.get()[np.newaxis, -1])
        pred = net.predict_from_hidden(hidden)

        emu.act(emu.actions[np.random.randint(0, emu.num_actions)])  # act randomly
        buf.insert(emu.get_screen_rgb())

        # display difference between prediction and true frame
        cv2.imshow('prediction', cv2.resize(pred[0], (84 * 4, 84 * 4)))

        if emu.terminal():
            emu.new_game()
        if step % _config['save_freq'] == 0:
            net.save('cont')

        step += 1






