import tensorflow as tf
import numpy as np


class DQNNature(object):
    def __init__(self, config):
        with tf.device(config['device']):
            tf.set_random_seed(config['random_seed'])
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                         log_device_placement=True))

            # placeholders
            self.state = tf.placeholder("float", [None, config['in_width'], config['in_height'], 4], name='state')
            self.nstate = tf.placeholder("float", [None, config['in_width'], config['in_height'], 4], name='nstate')
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
            reshape = tf.reshape(outputs[-1], [-1, conv_neurons], name='reshape')
            outputs.append(reshape)

            reshape_target = tf.reshape(outputs_target[-1], [-1, conv_neurons], name='reshape_target')
            outputs_target.append(reshape_target)

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
                self.max_Q = tf.reduce_max(self.Q, 1, name=scope.name + '_max_Q')
                outputs.append(self.max_Q)

                # target network and assign ops
                W_target = tf.Variable(W.initialized_value(), trainable=False)
                b_target = tf.Variable(b.initialized_value(), trainable=False)
                self.Q_target = tf.nn.bias_add(tf.matmul(outputs_target[-1],
                                                         W_target), b_target, name=scope.name + '_Q_target')
                outputs_target.append(self.Q_target)
                self.max_Q_target = tf.reduce_max(self.Q_target, 1, name=scope.name + '_max_Q_target')
                W_op = W_target.assign(W)
                b_op = b_target.assign(b)
                self.assign_ops.append(W_op)
                self.assign_ops.append(b_op)
            # endregion output layer

            # region cost
            self.discount = tf.constant(config['discount'])
            self.y = tf.add(self.rewards, tf.mul(self.discount, tf.mul(tf.sub(1.0, self.terminals), self.max_Q_target)))
            self.Q_action = tf.reduce_sum(tf.mul(self.Q, self.actions), reduction_indices=0)
            self.cost = tf.reduce_mean(tf.square(self.y - self.Q_action), reduction_indices=0)
            # endregion cost

            self.optimize_op = tf.train.RMSPropOptimizer(config['lr'], config['opt_decay'],
                                                         config['momentum'], config['opt_eps']).minimize(self.cost)

        self.saver = tf.train.Saver()

        self.sess.run(tf.initialize_all_variables())

    @staticmethod
    def make_weight(shape):
        return tf.get_variable('weight', shape,
                               initializer=tf.truncated_normal_initializer(stddev=0.001))

    @staticmethod
    def make_bias(shape):
        return tf.get_variable('bias', shape,
                               initializer=tf.constant_initializer(0.1))

    @staticmethod
    def conv2d(x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def train(self, s, a, r, ns, t):
        feed_dict = {self.state: s, self.actions: a, self.rewards: r, self.nstate: ns, self.terminals: t}

        cost, _ = self.sess.run([self.cost, self.optimize_op], feed_dict)

        return cost

