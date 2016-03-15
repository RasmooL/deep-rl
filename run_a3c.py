"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import sys
import time
from sacred import Experiment
from core.ALEEmulator import ALEEmulator
from async.Agent import Agent
ex = Experiment('a3c')


@ex.config
def net_config():
    conv_layers = 2
    conv_units = [16, 32]
    filter_sizes = [8, 4]
    strides = [4, 2]
    state_frames = 4
    fc_layers = 1
    fc_units = [256]
    in_width = 84
    in_height = 84
    discount = 0.99
    device = '/cpu:0'
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
    random_start = 10


@ex.config
def agent_config():
    eps = 1.0
    eps_decay = 1e-6
    eps_min = 0.1
    max_updates = 5e6
    test_freq = 5e2
    test_frames = 5e4
    num_threads = 4
    n_step = 5


@ex.automain
def main(_config):
    #sys.stdout = open('log_' + _config['rom_name'] + time.strftime('%H%M%d%m', time.gmtime()), 'w', buffering=True)
    tmp_emu = ALEEmulator(_config)
    _config['num_actions'] = tmp_emu.num_actions
    print _config

    agent = Agent(_config)

    agent.train()

