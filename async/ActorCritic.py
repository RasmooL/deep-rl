"""
Asynchronous advantage actor-critic as in "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al.

Copyright 2016 Rasmus Larsen

This software may be modified and distributed under the terms
of the MIT license. See the LICENSE.txt file for details.
"""

import tensorflow as tf
from core.BaseNet import BaseNet


class ActorCritic(BaseNet):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])

            # placeholders
            self.states = tf.placeholder("float",
                                         [None, config['in_width'], config['in_height'], config['state_frames']],
                                         name='state')
            self.rewards = tf.placeholder("float", [None], name='rewards')
            self.actions = tf.placeholder("int64", [None, config['num_actions']], name='actions')  # one-hot
            self.terminals = tf.placeholder("float", [None], name='terminals')

            outputs = [self.states]

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
            # endregion make conv layers

            # region make fc layers
            conv_neurons = 1
            for d in outputs[-1].get_shape()[1:].as_list():
                conv_neurons *= d
            self.reshape = tf.reshape(outputs[-1], [-1, conv_neurons], name='reshape')
            outputs.append(self.reshape)

            for n in range(config['fc_layers']):
                with tf.variable_scope('fc' + str(n)) as scope:
                    shape = [conv_neurons if n == 0 else config['fc_units'][n-1],
                             config['fc_units'][n]]
                    W = self.make_weight(shape)
                    b = self.make_bias(config['fc_units'][n])
                    fc = tf.nn.relu_layer(outputs[-1], W, b, name=scope.name)
                    outputs.append(fc)
            # endregion make fc layers

            # region output layer
            with tf.variable_scope('output_value') as scope:
                shape = [config['fc_units'][-1],
                         1]
                W = self.make_weight(shape)
                b = self.make_bias([1])
                self.V = tf.nn.bias_add(tf.matmul(outputs[-1], W), b, name=scope.name + '_V')
            with tf.variable_scope('output_policy') as scope:
                shape = [config['fc_units'][-1],
                         config['num_actions']]
                Wp = self.make_weight(shape)
                bp = self.make_bias(config['num_actions'])
                lp = tf.nn.bias_add(tf.matmul(outputs[-1], Wp), bp, name=scope.name + '_lp')
                self.policy = tf.nn.softmax(lp, name='_policy')
                self.log_policy = tf.log(self.policy, name='_log_policy') # no logsoftmax kernel yet...
                self.argmax_policy = tf.argmax(self.policy, dimension=1)
            # endregion output layer

            # region cost
                diff = tf.sub(self.rewards, self.V)
                self.cost_V = tf.square(diff)

                onehot_actions = tf.one_hot(self.actions, config['num_actions'], 1.0, 0.0)
                log_p_ai = tf.reduce_sum(tf.mul(self.log_policy, onehot_actions), reduction_indices=1)
                # self.cost_P = log_p_ai * diff
                # NOTE: should check that this is correct, but I think that the gradient that goes
                # to self.V through diff should be stopped (or we train value weights using policy cost)
                self.cost_P = tf.mul(log_p_ai, tf.stop_gradient(diff))  # pretend diff is a constant
            # endregion cost

            self.optimizer = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                       config['momentum'], config['opt_eps'])
            self.opt_V = self.optimizer.minimize(self.cost_V)
            self.opt_P = self.optimizer.minimize(self.cost_P)  # TODO: wait.. should we maximize?

        super(ActorCritic, self).__init__(config)

    def get_copy(self, namespace):
        new_graph = tf.Graph()
        # TODO: implement NetCopier and get this function to return a full network copy
        return NetCopier()

    def train(self, states, actions, rewards):
        feed_dict = {self.states: states, self.actions: actions, self.rewards: rewards}
        cost_V, cost_P = self.sess.run([self.opt_V, self.opt_P], feed_dict)
        return cost_V, cost_P

    def predict(self, states):
        feed_dict = {self.states: states}
        a, V = self.sess.run([self.argmax_policy, self.V], feed_dict)
        return a, V


class NetCopier(ActorCritic):
    def __init__(self):
        raise NotImplementedError()





