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
from continuation.PTNet import PTNet
import numpy as np

ex = Experiment('continuation')


@ex.config
def net_config():
    conv_layers = 3
    conv_units = [32, 64, 64]
    filter_sizes = [8, 4, 2]
    strides = [4, 2, 1]
    hidden_units = 256
    num_heads = 1
    gate_noise = 0.1
    sharpening_slope = 10
    in_width = 84
    in_height = 84
    device = '/gpu:0'
    lr = 0.00025
    opt_decay = 0.95
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
    batch_size = 2
    train_start = 5e3
    train_frames = 5e6
    test_freq = 5e4
    test_frames = 5e3


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
    #sys.stdout = open('log_' + _config['rom_name'] + time.strftime('%H%M%d%m', time.gmtime()), 'w', buffering=True)
    print "#{}".format(_config)
    emu = ALEEmulator(_config)
    _config['num_actions'] = emu.num_actions
    net = PTNet(_config)

    screen = emu.get_screen_rgb()
    emu.act(1)
    screen = screen[np.newaxis, :]
    screen = np.append(screen, emu.get_screen_rgb()[np.newaxis, :], axis=0)
    hidden = net.encode(screen)
    gated, dist = net.gate(screen, [1])
    print gated - hidden[0]
