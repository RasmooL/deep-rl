from ale_python_interface import ALEInterface
import cv2
import random


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
        self.out_width = config['in_width']
        self.out_height = config['in_height']

        self.max_random_steps = config['random_start']

    def new_game(self):
        self.ale.reset_game()

    def new_random_game(self):
        self.new_game()
        for i in range(random.randint(0, self.max_random_steps)):
            self.act(random.randint(0, self.num_actions - 1))
            if self.terminal():
                print "Episode terminated during random start."
                self.new_random_game()
                break


    def terminal(self):
        return self.ale.game_over()

    def act(self, action):
        return self.ale.act(action)

    def get_screen_gray(self):
        screen = self.ale.getScreenGrayscale()
        return cv2.resize(screen, (self.out_width, self.out_height))


