"""
Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import tensorflow as tf


class BaseDQN(object):
    def __init__(self, config):
        self.saver = tf.train.Saver()

        self.tensorboard = config['tensorboard']
        if self.tensorboard:
            self.merged = tf.merge_all_summaries()
            self.writer = tf.train.SummaryWriter("logs/", self.sess.graph_def)

        self.sess.run(tf.initialize_all_variables())

    def save(self, n):
        self.saver.save(self.sess, "save/model_" + str(n) + ".ckpt")

    def load(self, n):
        self.saver.restore(self.sess, "save/model_" + str(n) + ".ckpt")

    @staticmethod
    def make_weight(shape):
        return tf.get_variable('weight', shape,
                               initializer=tf.uniform_unit_scaling_initializer(factor=1.43))  # 1.43 for relu

    @staticmethod
    def make_bias(shape):
        return tf.get_variable('bias', shape,
                               initializer=tf.constant_initializer(0.001))

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")