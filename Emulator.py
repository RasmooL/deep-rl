from ale_python_interface import ALEInterface


class Emulator(object):
    def __init__(self, config):
        self.ale = ALEInterface()

        self.ale.setInt('frame_skip', config['frame_skip'])
        self.ale.setFloat('repeat_action_probability', config['repeat_prob'])
        self.ale.setBool('color_averaging', config['color_avg'])
        self.ale.setBool('display_screen', config['display_screen'])
        self.ale.setInt('random_seed', config['random_seed'])

        self.ale.loadROM(config['rom_path'])

        self.actions = self.ale.getMinimalActionSet()
        self.num_actions = len(self.actions)

        (self.screen_width, self.screen_height) = self.ale.getScreenDims()

    def reset(self):
        self.ale.reset_game()

    def terminal(self):
        return self.ale.game_over()

    def act(self, action):
        return self.ale.act(action)

    def get_screen_gray(self):
        return self.ale.getScreenGrayscale()


