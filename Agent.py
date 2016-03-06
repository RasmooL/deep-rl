from ReplayMemory import ReplayMemory

class Agent(object):
    def __init__(self, emu, net, config):
        self.memory = ReplayMemory(config)
        self.emu = emu
        self.net = net

    def next(self, action):
        reward = self.emu.act(action)
        next_screen = self.emu.get_screen_gray()
        return reward, next_screen




