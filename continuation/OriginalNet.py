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


class OriginalNet(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.states = tf.placeholder(tf.float32,
                                        [None, config['in_width'], config['in_height'], 3],
                                        name='states')
            self.hidden_input = tf.placeholder(tf.float32,
                                               [1, config['hidden_units']],
                                               name='hidden_input')
            self.step = tf.placeholder(tf.float32, [1], name='step')

            self.hidden = self.__encoder__(config, self.states)
            self.gated = self.__gating__(config, self.hidden)
            self.train_decoder, self.test_decoder = self.__decoder__(config, self.gated)

            self.cost = self.__cost__(self.states, self.train_decoder)

            self.optimizer = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                    config['momentum'], config['opt_eps'])
            grads_and_vars = self.optimizer.compute_gradients(self.cost)
            capped_grads_and_vars = [(tf.clip_by_value(g, -100.0, 100.0), v) for (g,v) in grads_and_vars]
            self.optimize_op = self.optimizer.apply_gradients(capped_grads_and_vars)

            super(OriginalNet, self).__init__(config)

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

            # get info from last layer
            tmp = hidden.shape
            self.conv_shape = [-1, tmp[1], tmp[2], tmp[3]]
            self.conv_neurons = np.prod(tmp[1:])

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
                sharpened = tf.clip_by_value(sharpened, 0.0, 1e10, name='clip_zero')
                sharpened = tf.pow(sharpened, tf.minimum(61.0,  # NOTE: max for single precision
                                                         (1.0 + (self.step / 10000.0)
                                                          * config['sharpening_slope'])))
                s_norm = tf.maximum(tf.reduce_sum(sharpened, 1, keep_dims=True), 1e-30, 'calc_norm')
                sharpened = tf.div(sharpened, s_norm, name='norm_div')
                heads.append(sharpened)
        heads_tensor = tf.reduce_sum(tf.concat(0, heads), reduction_indices=0, name='reduce')

        # merge head distributions
        self.gate_distribution = tf.clip_by_value(heads_tensor, 0.0, 1.0, name='clip_dist')

        # gate encodings
        return tf.mul(self.gate_distribution, cur_hidden) \
             + tf.mul(tf.sub(1.0, self.gate_distribution), prev_hidden)

    def __decoder__(self, config, input):
        with tf.variable_scope('decoder'):
            template = pt.template('input')

            # linear layer and reshape
            template = template.fully_connected(self.conv_neurons, activation_fn=tf.nn.relu)
            template = template.reshape([pt.UnboundVariable('batch_size'),
                              self.conv_shape[1],
                              self.conv_shape[2],
                              self.conv_shape[3]])

            # (de)convolutional layers
            for n in range(config['conv_layers'])[::-1]:  # reverse order
                template = template.deconv2d(config['filter_sizes'][n],
                                 3 if n == 0 else config['conv_units'][n-1],
                                 stride=config['strides'][n],
                                 activation_fn=tf.nn.relu,
                                 edges='VALID')

            train_decoder = template.construct(input=input, batch_size=config['batch_size']-1)
            test_decoder = template.construct(input=self.hidden_input, batch_size=1)

            return train_decoder.tensor, test_decoder.tensor

    def __cost__(self, original, decoded):
        cur_states = tf.boolean_mask(original, self.cur_mask)

        return tf.nn.l2_loss(cur_states - decoded)

    def encode(self, s):
            feed_dict = {self.states: s/255.0}

            return self.sess.run(self.hidden, feed_dict)

    def gate(self, states, step):
        feed_dict = {self.states: states/255.0, self.step: step}

        return self.sess.run([self.gated, self.gate_distribution], feed_dict)

    def predict_from_hidden(self, hidden):
        feed_dict = {self.hidden_input: hidden}

        return self.sess.run(self.test_decoder, feed_dict)

    def train(self, states, step):
        feed_dict = {self.states: states/255.0, self.step: step}

        cost, _ = self.sess.run([self.cost, self.optimize_op], feed_dict)

        return cost
