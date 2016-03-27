"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import numpy as np
import cv2


class ScreenBuffer(object):
    def __init__(self, config, max_size):
        self.buffer = np.zeros([max_size,
                                config['in_width'],
                                config['in_height'],
                                3],
                               dtype=np.uint8)

    def insert(self, screen):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = screen

    def get(self):
        return self.buffer

