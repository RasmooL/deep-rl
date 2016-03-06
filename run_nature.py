from DQNNature import DQNNature
from Emulator import Emulator
from Agent import Agent
from sacred import Experiment
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
    opt_eps = 0.0

@ex.config
def emu_config():
    rom_path = '../ale-git/roms/breakout.bin'
    display_screen = True
    frame_skip = 4
    repeat_prob = 0.0
    color_avg = True
    random_seed = 42
    pass

@ex.config
def agent_config():
    hist_size = 50000

@ex.automain
def main(_config):
    emu = Emulator(_config)
    _config['num_actions'] = emu.num_actions
    net = DQNNature(_config)
    agent = Agent(emu, net, _config)

    while not emu.terminal():
        reward, scr = agent.next(1)
        t = emu.terminal()



