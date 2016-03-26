"""
Network for Continuation Learning (Whitney et al., 2016) in the ALE.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import tensorflow as tf
import numpy as np
from core.BaseNet import BaseNet


class AtariNet(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.states = tf.placeholder("float",
                                        [None, config['in_width'], config['in_height'], 3],
                                        name='prev_states')
            self.step = tf.placeholder("float", [1], name='step')


            outputs = [self.states]
            conv_shapes = []

            # convolutional layers
            for n in range(config['conv_layers']):
                with tf.variable_scope('conv' + str(n)) as scope:
                    shape = [config['filter_sizes'][n],
                             config['filter_sizes'][n],
                             3 if n == 0 else config['conv_units'][n-1],
                             config['conv_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['conv_units'][n])
                    conv = self.conv2d(outputs[-1], W, config['strides'][n])
                    conv = tf.nn.bias_add(conv, b)
                    conv = tf.nn.relu(conv, name=scope.name)
                    outputs.append(conv)
                    conv_shapes.append(conv.get_shape())

            # hidden layer
            conv_neurons = 1
            for d in outputs[-1].get_shape()[1:].as_list():
                conv_neurons *= d
            self.reshape = tf.reshape(outputs[-1], [-1, conv_neurons], name='reshape')
            with tf.variable_scope('flat') as scope:
                shape = [conv_neurons,
                         config['hidden_units']]
                W = self.make_weight(shape)
                b = self.make_bias(config['hidden_units'])
                self.hidden = tf.nn.bias_add(tf.matmul(self.reshape, W), b, name=scope.name + '_out')

            # gating heads
            prev_mask = np.ones(config['batch_size'], dtype=np.bool)
            prev_mask[-1] = 0
            cur_mask = np.ones(config['batch_size'], dtype=np.bool)
            cur_mask[0] = 0
            prev_hidden = tf.boolean_mask(self.hidden, prev_mask)
            cur_hidden = tf.boolean_mask(self.hidden, cur_mask)
            combined_states = tf.concat(1, [prev_hidden, cur_hidden])
            heads = []
            for n in range(config['num_heads']):
                with tf.variable_scope('head' + str(n)) as scope:
                    shape = [2 * config['hidden_units'],
                             config['hidden_units']]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['hidden_units'])
                    fc = tf.nn.relu_layer(combined_states, W, b, name=scope.name) \
                         + tf.random_normal(config['hidden_units'], stddev=config['gate_noise'])
                    sharpened = tf.pow(fc, tf.minimum(100.0, 1 + (self.step / 10000.0)
                                                     * config['sharpening_slope'])) + 1e-20
                    heads.append(tf.nn.l2_normalize(sharpened, 0, epsilon=1e-100))
            heads_tensor = tf.concat(0, heads)

            # merge head distributions
            gate_distribution = tf.clip_by_value(tf.reduce_sum(heads_tensor, 0), 0.0, 1.0, name='dist')

            # gate encodings
            self.output = tf.mul(gate_distribution, cur_hidden) \
                          + tf.mul(tf.sub(1.0, gate_distribution), prev_hidden)

            # decoder
            with tf.variable_scope('decoder') as scope:
                shape = [config['hidden_units'],
                         conv_neurons]
                W = self.make_weight(shape)
                b = self.make_bias(conv_neurons)
                linear = tf.nn.relu_layer(self.output, W, b, name=scope.name + '_linear')
                reshaped = tf.reshape(linear, conv_shapes[-1])

                # (de)convolutional layers
                outputs.append(reshaped)
                for n in range(config['conv_layers'])[::-1]:  # reverse order
                    with tf.variable_scope(scope.name + '_conv' + str(n)) as cscope:
                        shape = [config['filter_sizes'][n],
                                 config['filter_sizes'][n],
                                 3 if n == 0 else config['conv_units'][n - 1],
                                 config['conv_units'][n]]
                        W = self.make_weight(shape)
                        b = self.make_bias(config['conv_units'][n])
                        conv = self.conv2d_transpose(outputs[-1], W,
                                                     conv_shapes[n], config['strides'][n])
                        conv = tf.nn.bias_add(conv, b)
                        conv = tf.nn.relu(conv, name=cscope.name)
                        outputs.append(conv)


            # cost
            # TODO

        super(AtariNet, self).__init__(config)

