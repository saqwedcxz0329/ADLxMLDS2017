import os

import numpy as np
import tensorflow as tf

class PolicyNetwork(object):
    def __init__(self,
                n_actions,
                n_features,
                learning_rate=0.1,
                reward_decay=0.9):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # to memory this episode states
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(
                tf.float32, [None, self.n_features], name="observations")
            self.tf_acts = tf.placeholder(
                tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(
                tf.float32, [None, ], name="actions_value")

        # input_layer = self.tf_obs
        input_layer = tf.reshape(self.tf_obs, [-1, 80, 80, 1])
        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=16,
            kernel_size=[8, 8],
            strides=4,
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        # pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[4, 4],
            strides=2,
            padding="same",
            activation=tf.nn.relu)
        # pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
        # fc1
        pool2_flat = tf.contrib.layers.flatten(conv2)
        layer = tf.layers.dense(
            inputs=pool2_flat,
            units=128,
            activation=tf.nn.tanh,  # tanh activation
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # use softmax to convert to probability
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)
            # reward guided loss
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(loss)
    
    def make_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={
                                     self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(
            range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    
    def train(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt: discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm
    
    def save(self, model_path, model_name):
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.save(self.sess, os.path.join(model_path, model_name))

    def restore(self, model_path, model_name):
        saver = tf.train.Saver()
        saver.restore(self.sess, os.path.join(model_path, model_name))

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            # reset running_addd when one player gets point
            if self.ep_rs[t] != 0:
               running_add = 0
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
            
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
