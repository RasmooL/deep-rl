"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import sys
import time

from sacred import Experiment
from core.ALEEmulator import ALEEmulator
from exp.ConvAuto import ConvAuto
from core.ScreenBuffer import ScreenBuffer
import numpy as np
import cv2

ex = Experiment('convauto')


@ex.config
def net_config():
    conv_layers = 1
    conv_units = [16]
    filter_sizes = [10]
    strides = [5]
    hidden_units = 512
    in_width = 210
    in_height = 160
    device = '/gpu:0'
    lr = 0.0002
    opt_decay = 0.99
    momentum = 0.0
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
    update_freq = 2
    train_start = 5e3
    train_frames = 5e6
    test_freq = 5e4
    test_frames = 5e3
    save_freq = 5e3

@ex.command
def test(_config):
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = ConvAuto(_config)
    net.load('autoconv')


@ex.automain
def main(_config, _log):
    #sys.stdout = open('log_' + _config['rom_name'] + time.strftime('%H%M%d%m', time.gmtime()), 'w', buffering=True)
    print "#{}".format(_config)
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = ConvAuto(_config)

    cv2.startWindowThread()
    cv2.namedWindow("prediction")

    # fill screen history up to batch size
    buf = ScreenBuffer(_config, _config['batch_size'])
    for n in range(_config['batch_size']):
        emu.act(emu.actions[np.random.randint(0, emu.num_actions)])  # act randomly
        buf.insert(emu.get_screen_rgb_full())
    # train
    step = 0
    while step < _config['train_frames']:
        if step % _config['update_freq'] == 0:
            cost, recon = net.train(buf.get(), [step])
            print step, cost
            cv2.imshow('prediction', cv2.resize(recon[0], (210, 160)))

        emu.act(emu.actions[np.random.randint(0, emu.num_actions)])  # act randomly
        buf.insert(emu.get_screen_rgb_full())

        if emu.terminal():
            emu.new_game()
        if step % _config['save_freq'] == 0:
            net.save('autoconv')

        step += 1






