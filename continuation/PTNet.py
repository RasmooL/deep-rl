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


class PTNet(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.states = tf.placeholder(tf.float32,
                                        [None, config['in_width'], config['in_height'], 3],
                                        name='prev_states')
            self.step = tf.placeholder(tf.float32, [1], name='step')

            self.hidden = self.__encoder__(config, self.states)
            self.gated = self.__gating__(config, self.hidden)
            self.decoded = self.__decoder__(config, self.gated)

            self.cost = self.__cost__(self.states, self.decoded)

            self.optimize_op = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                         config['momentum'], config['opt_eps']).minimize(self.cost)

            super(PTNet, self).__init__(config)

    def __encoder__(self, config, input):
        with tf.variable_scope('encoder'):
            hidden = pt.wrap(input).sequential()

            # convolutional layers
            for n in range(config['conv_layers']):
                hidden.conv2d(config['filter_sizes'][n],
                              config['conv_units'][n],
                              stride=config['strides'][n],
                              activation_fn=tf.nn.relu,
                              edges='VALID')
                print hidden.shape

            # get info from last layer
            tmp = hidden.get_shape()
            self.conv_shape = [config['batch_size'], int(tmp[1]), int(tmp[2]), int(tmp[3])]
            self.conv_neurons = np.prod(self.conv_shape[1:])

            # fully connected to hidden layer
            return (hidden.flatten()
                        .fully_connected(config['hidden_units'], activation_fn=None)).tensor

    def __gating__(self, config, input):
        # creating the gating head distributions, could likely be much prettier
        prev_mask = np.ones(config['batch_size'], dtype=np.bool)
        prev_mask[-1] = 0
        self.cur_mask = np.ones(config['batch_size'], dtype=np.bool)  # useful in cost function
        self.cur_mask[0] = 0
        prev_hidden = tf.boolean_mask(input, prev_mask)
        cur_hidden = tf.boolean_mask(input, self.cur_mask)
        combined_states = tf.concat(1, [prev_hidden, cur_hidden])
        heads = []
        for n in range(config['num_heads']):
            with tf.variable_scope('head' + str(n)):
                sharpened = (pt.wrap(combined_states)
                               .fully_connected(config['hidden_units'])).tensor
                sharpened = sharpened + tf.random_normal([config['hidden_units']],
                                                         stddev=config['gate_noise'])

                # now clip (pow undefined for negative base) and sharpen
                sharpened = tf.clip_by_value(sharpened, 0.0, 1e10)
                sharpened = tf.pow(sharpened, tf.minimum(61.0,  # NOTE: max for single precision
                                                         (1.0 + (self.step / 10000.0)
                                                          * config['sharpening_slope'])))
                s_norm = tf.maximum(tf.reduce_sum(tf.abs(sharpened), 1), 1e-30)
                sharpened = tf.div(sharpened, s_norm)
                heads.append(sharpened)
        heads_tensor = tf.reduce_sum(tf.concat(0, heads), reduction_indices=0)

        # merge head distributions
        self.gate_distribution = tf.clip_by_value(heads_tensor, 0.0, 1.0, name='dist')

        # gate encodings
        return tf.mul(self.gate_distribution, cur_hidden) \
             + tf.mul(tf.sub(1.0, self.gate_distribution), prev_hidden)

    def __decoder__(self, config, input):
        with tf.variable_scope('decoder'):
            decoded = pt.wrap(input).sequential()

            # linear layer and reshape
            decoded.fully_connected(self.conv_neurons, activation_fn=tf.nn.relu)
            decoded.reshape(self.conv_shape)

            # (de)convolutional layers
            for n in range(config['conv_layers'])[::-1]:  # reverse order
                decoded.deconv2d(config['filter_sizes'][n],
                                 3 if n == 0 else config['conv_units'][n-1],
                                 stride=config['strides'][n],
                                 activation_fn=tf.nn.relu,
                                 edges='VALID')
                print decoded.shape

            return decoded.tensor

    def __cost__(self, original, decoded):
        cur_states = tf.boolean_mask(original, self.cur_mask)

        return tf.nn.l2_loss(cur_states - decoded)

    def encode(self, s):
            feed_dict = {self.states: s/255.0}

            return self.sess.run(self.hidden, feed_dict)

    def gate(self, states, step):
        feed_dict = {self.states: states/255.0, self.step: step}

        return self.sess.run([self.gated, self.gate_distribution], feed_dict)

