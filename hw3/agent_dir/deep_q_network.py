import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

def leaky_relu(features, alpha=0.01, name=None):
  """Compute the Leaky ReLU activation function.
  "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
  AL Maas, AY Hannun, AY Ng - Proc. ICML, 2013
  http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
  Args:
    features: A `Tensor` representing preactivation values.
    alpha: Slope of the activation function at x < 0.
    name: A name for the operation (optional).
  Returns:
    The activation value.
  """
  with ops.name_scope(name, "LeakyRelu", [features, alpha]):
    features = ops.convert_to_tensor(features, name="features")
    alpha = ops.convert_to_tensor(alpha, name="alpha")
    return math_ops.maximum(alpha * features, features)


class DeepQNetwork(object):
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            gamma=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 3))

        # consist of [target_net, evaluate_net]
        tf.reset_default_graph()
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=1)

    def _build_net(self):
        # ------------------ all inputs ------------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action
        self.done = tf.placeholder(tf.float32, [None, ], name='done')  # is terminal
        

        # w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        n_neuron = 512

        eval_input = tf.reshape(self.s, [-1, 84, 84, 4])
        target_input = tf.reshape(self.s_, [-1, 84, 84, 4])

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):

            # Convolutional Layer #1
            e_conv1 = tf.layers.conv2d(
                inputs=eval_input,
                filters=32,
                kernel_size=[8, 8],
                strides=4,
                padding="same",
                activation=tf.nn.relu)
            
            # Convolutional Layer #2
            e_conv2 = tf.layers.conv2d(
                inputs=e_conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=2,
                padding="same",
                activation=tf.nn.relu)
            
            # Convolutional Layer #3
            e_conv3 = tf.layers.conv2d(
                inputs=e_conv2,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu)

            e_conv3_flat = tf.contrib.layers.flatten(e_conv3)
            e1 = tf.layers.dense(e_conv3_flat, 
                                n_neuron, 
                                leaky_relu, 
                                # kernel_initializer=w_initializer,
                                # bias_initializer=b_initializer, 
                                name='e1')
            self.q_eval = tf.layers.dense(e1,
                                self.n_actions, 
                                # kernel_initializer=w_initializer,
                                # bias_initializer=b_initializer,
                                name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            # Convolutional Layer #1
            t_conv1 = tf.layers.conv2d(
                inputs=target_input,
                filters=32,
                kernel_size=[8, 8],
                strides=4,
                padding="same",
                activation=tf.nn.relu)
            
            # Convolutional Layer #2
            t_conv2 = tf.layers.conv2d(
                inputs=t_conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=2,
                padding="same",
                activation=tf.nn.relu)
            
            # Convolutional Layer #3
            t_conv3 = tf.layers.conv2d(
                inputs=t_conv2,
                filters=64,
                kernel_size=[3, 3],
                strides=1,
                padding="same",
                activation=tf.nn.relu)

            t_conv3_flat = tf.contrib.layers.flatten(t_conv3)
            t1 = tf.layers.dense(t_conv3_flat,
                                n_neuron, 
                                leaky_relu, 
                                # kernel_initializer=w_initializer,
                                # bias_initializer=b_initializer,
                                name='t1')
            self.q_next = tf.layers.dense(t1,
                                self.n_actions, 
                                # kernel_initializer=w_initializer,
                                # bias_initializer=b_initializer,
                                name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + (1. - self.done) * self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr, decay=0.99).minimize(self.loss)

    def store_transition(self, s, a, r, s_, done):
        done = 1. if done else 0.
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r, done], s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def make_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    def update_epsilon(self):
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

    def update_target_net(self):
        self.sess.run(self.target_replace_op)
        print('\ntarget_params_replaced\n')
        

    def train(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.done: batch_memory[:, self.n_features + 2],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.learn_step_counter += 1

    def save(self, model_path, model_name, episode):
        self.saver.save(self.sess, os.path.join(model_path, model_name), global_step=episode)
    
    def save_interrupt(self, model_path, model_name, episode):
        saver = tf.train.Saver(max_to_keep=1)
        saver.save(self.sess, os.path.join(model_path, model_name), global_step=episode)

    def restore(self, model_path, model_name):
        self.saver.restore(self.sess, os.path.join(model_path, model_name))
