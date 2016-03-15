"""
DQN with Double Q-learning as in "Deep Reinforcement Learning with Double Q-learning" by van Hasselt et al.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import tensorflow as tf
from core.BaseNet import BaseNet


class DoubleDQN(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.state = tf.placeholder("float",
                                        [None, config['in_width'], config['in_height'], config['state_frames']],
                                        name='state')
            self.nstate = tf.placeholder("float",
                                         [None, config['in_width'], config['in_height'], config['state_frames']],
                                         name='nstate')
            self.rewards = tf.placeholder("float", [None], name='rewards')
            self.actions = tf.placeholder("float", [None, config['num_actions']], name='actions')  # one-hot
            self.terminals = tf.placeholder("float", [None], name='terminals')

            outputs = [self.state]
            outputs_target = [self.nstate]
            self.assign_ops = []

            # region make conv layers
            for n in range(config['conv_layers']):
                with tf.variable_scope('conv' + str(n)) as scope:
                    shape = [config['filter_sizes'][n],
                             config['filter_sizes'][n],
                             config['state_frames'] if n == 0 else config['conv_units'][n-1],
                             config['conv_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['conv_units'][n])
                    conv = self.conv2d(outputs[-1], W, config['strides'][n])
                    conv = tf.nn.bias_add(conv, b)
                    conv = tf.nn.relu(conv, name=scope.name)
                    outputs.append(conv)

                    # target network and assign ops
                    W_target = tf.Variable(W.initialized_value(), trainable=False)
                    b_target = tf.Variable(b.initialized_value(), trainable=False)
                    conv_target = self.conv2d(outputs_target[-1], W_target, config['strides'][n])
                    conv_target = tf.nn.bias_add(conv_target, b_target)
                    conv_target = tf.nn.relu(conv_target, name=scope.name + '_target')
                    outputs_target.append(conv_target)
                    W_op = W_target.assign(W)
                    b_op = b_target.assign(b)
                    self.assign_ops.append(W_op)
                    self.assign_ops.append(b_op)
            # endregion make conv layers

            # region make fc layers
            conv_neurons = 1
            for d in outputs[-1].get_shape()[1:].as_list():
                conv_neurons *= d
            self.reshape = tf.reshape(outputs[-1], [-1, conv_neurons], name='reshape')
            outputs.append(self.reshape)

            self.reshape_target = tf.reshape(outputs_target[-1], [-1, conv_neurons], name='reshape_target')
            outputs_target.append(self.reshape_target)

            for n in range(config['fc_layers']):
                with tf.variable_scope('fc' + str(n)) as scope:
                    shape = [conv_neurons if n == 0 else config['fc_units'][n-1],
                             config['fc_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['fc_units'][n])
                    fc = tf.nn.relu_layer(outputs[-1], W, b, name=scope.name)
                    outputs.append(fc)

                    # target network and assign ops
                    W_target = tf.Variable(W.initialized_value(), trainable=False)
                    b_target = tf.Variable(b.initialized_value(), trainable=False)
                    fc_target = tf.nn.relu_layer(outputs_target[-1], W_target, b_target, name=scope.name + '_target')
                    outputs_target.append(fc_target)
                    W_op = W_target.assign(W)
                    b_op = b_target.assign(b)
                    self.assign_ops.append(W_op)
                    self.assign_ops.append(b_op)
            # endregion make fc layers

            # region output layer
            with tf.variable_scope('output') as scope:
                shape = [config['fc_units'][-1],
                         config['num_actions']]
                W = self.make_weight(shape)
                b = self.make_bias(config['num_actions'])
                self.Q = tf.nn.bias_add(tf.matmul(outputs[-1], W), b, name=scope.name + '_Q')
                outputs.append(self.Q)
                self.argmax_Q = tf.argmax(self.Q, dimension=1, name=scope.name + '_argmax_Q')
                outputs.append(self.argmax_Q)

                # target network and assign ops
                W_target = tf.Variable(W.initialized_value(), trainable=False)
                b_target = tf.Variable(b.initialized_value(), trainable=False)
                self.Q_target = tf.nn.bias_add(tf.matmul(outputs_target[-1],
                                                         W_target), b_target, name=scope.name + '_Q_target')
                outputs_target.append(self.Q_target)
                W_op = W_target.assign(W)
                b_op = b_target.assign(b)
                self.assign_ops.append(W_op)
                self.assign_ops.append(b_op)
            # endregion output layer

            # region cost
            self.discount = tf.constant(config['discount'])
            # NOTE: one_hot is currently ONLY in git version
            argmax_Q_onehot = tf.one_hot(self.argmax_Q, depth=config['num_actions'], on_value=1.0, off_value=0.0)
            self.Q_next = tf.reduce_sum(tf.mul(self.Q_target, argmax_Q_onehot),
                                        reduction_indices=1)  # main difference to standard DQN is this value
            self.y = tf.add(self.rewards, tf.mul(self.discount, tf.mul(tf.sub(1.0, self.terminals), self.Q_next)))
            self.Q_action = tf.reduce_sum(tf.mul(self.Q, self.actions), reduction_indices=1)  # TODO: see Tensorflow#206

            # td error clipping
            self.clip_delta = tf.constant(config['clip_delta'])
            self.diff = tf.sub(self.y, self.Q_action)
            self.quadratic_part = tf.minimum(tf.abs(self.diff), self.clip_delta)
            self.linear_part = tf.sub(tf.abs(self.diff), self.quadratic_part)
            self.clipped_diff = tf.add(0.5 * tf.square(self.quadratic_part),
                                       tf.mul(self.clip_delta, self.linear_part))
            self.cost = tf.reduce_sum(self.clipped_diff, reduction_indices=0)
            # endregion cost

            self.optimize_op = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                         config['momentum'], config['opt_eps']).minimize(self.cost)

        super(DoubleDQN, self).__init__(config)

    def sync_target(self):
        self.sess.run(self.assign_ops)

    def train(self, s, a, r, ns, t):
        feed_dict = {self.state: s/255.0, self.actions: a, self.rewards: r, self.nstate: ns/255.0, self.terminals: t}

        cost, _ = self.sess.run([self.cost, self.optimize_op], feed_dict)

        return cost

    def predict(self, s):
        feed_dict = {self.state: s/255.0}

        argmax_Q = self.sess.run(self.argmax_Q, feed_dict)[0]

        return argmax_Q


