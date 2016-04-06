"""
Network for Continuation Learning (Whitney et al., 2016) in the ALE.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import tensorflow as tf
import prettytensor as pt
import numpy as np
import core.deconv
from core.BaseNet import BaseNet


class ConvAuto(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            with pt.defaults_scope(summary_collections=None):
                # placeholders
                self.states = tf.placeholder(tf.float32,
                                            [None, config['in_width'], config['in_height'], 3],
                                            name='states')

                self.hidden = self.__encoder__(config, self.states)
                self.recon = self.__decoder__(config, self.hidden)
                self.cost = self.__cost__(self.states, self.recon)

            self.optimizer = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                    config['momentum'], config['opt_eps'])
            grads_and_vars = self.optimizer.compute_gradients(self.cost)
            capped_grads_and_vars = [(tf.clip_by_value(g, -100.0, 100.0), v) for (g,v) in grads_and_vars]
            self.optimize_op = self.optimizer.apply_gradients(capped_grads_and_vars)

            super(ConvAuto, self).__init__(config)

    def __encoder__(self, config, input):
        with tf.variable_scope('encoder') as scope:
            hidden = pt.wrap(input).sequential()

            # convolutional layers
            for n in range(config['conv_layers']):
                hidden.conv2d(config['filter_sizes'][n],
                              config['conv_units'][n],
                              stride=config['strides'][n],
                              activation_fn=tf.nn.relu,
                              edges='VALID')
                if n == 0:  # just first layer
                    W = tf.all_variables()[0]  # very hacky
                    for (i, img) in enumerate(tf.split(3, config['conv_units'][n], W)):
                        img = tf.transpose(img, [2, 0, 1, 3])
                        tf.image_summary(scope.name + "/W" + str(i), img, max_images=4)

            # get info from last layer
            tmp = hidden.shape
            self.conv_shape = [-1, tmp[1], tmp[2], tmp[3]]
            self.conv_neurons = np.prod(tmp[1:])

            # fully connected to hidden layer
            return (hidden.flatten()
                          .fully_connected(config['hidden_units'], activation_fn=None)).tensor

    def __decoder__(self, config, input):
        with tf.variable_scope('decoder'):
            recon = pt.wrap(input).sequential()

            # linear layer and reshape
            recon.fully_connected(self.conv_neurons, activation_fn=tf.nn.relu)
            recon.reshape([config['batch_size'],
                           self.conv_shape[1],
                           self.conv_shape[2],
                           self.conv_shape[3]])


            # (de)convolutional layers
            for n in range(config['conv_layers'])[::-1]:  # reverse order
                recon.deconv2d(config['filter_sizes'][n],
                               3 if n == 0 else config['conv_units'][n-1],
                               stride=config['strides'][n],
                               activation_fn=(tf.nn.sigmoid if n == 0 else tf.nn.relu),
                               edges='VALID')

            return recon.tensor

    def __cost__(self, original, recon):
        return tf.nn.l2_loss(original - recon)

    def encode(self, s):
            feed_dict = {self.states: s/255.0}

            return self.sess.run(self.hidden, feed_dict)

    def train(self, states, step):
        feed_dict = {self.states: states/255.0}

        cost, recon, _ = self.sess.run([self.cost, self.recon, self.optimize_op], feed_dict)

        return cost, recon
