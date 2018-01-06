import tensorflow as tf
import tensorflow.contrib as tc
import math

def leaky_relu(x, alpha=0.2):
	return tf.maximum(tf.minimum(0.0, alpha * x), x)

class Generator(object):
	def __init__(self):
		pass
		
	def __call__(self, seq_idx, z, reuse=False, train=True):

		batch_size = tf.shape(seq_idx)[0]

		tags_vectors = seq_idx

		with tf.variable_scope("g_net") as scope:

			if reuse:
				scope.reuse_variables()

			noise_vector = tf.concat([tags_vectors, z], axis=1)

			out = tf.layers.dense(
				noise_vector, 6*6*256,
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=train)
			out = tf.reshape(out, [-1, 6, 6, 256])
			out = tf.nn.relu(out)

			out = tf.layers.conv2d_transpose(
				out, 128, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=train)
			out = tf.nn.relu(out)

			out = tf.layers.conv2d_transpose(
				out, 64, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=train)
			out = tf.nn.relu(out)
			
			out = tf.layers.conv2d_transpose(
				out, 32, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=train)
			out = tf.nn.relu(out)

			out = tf.layers.conv2d_transpose(
				out, 3, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.nn.tanh(out)
			
			return out

	@property
	def vars(self):
		return [var for var in tf.global_variables() if "g_net" in var.name]

class Discriminator(object):
	def __init__(self):
		pass
		
	def __call__(self, seq_idx, img, reuse=True):

		batch_size = tf.shape(seq_idx)[0]

		tags_vectors = seq_idx

		with tf.variable_scope("d_net") as scope:

			if reuse == True:
				scope.reuse_variables()

			out = tf.layers.conv2d(
				img, 32, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=True)
			out = leaky_relu(out)

			out = tf.layers.conv2d(
				out, 64, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=True)
			out = leaky_relu(out)
			
			out = tf.layers.conv2d(
				out, 128, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=True)
			out = leaky_relu(out)

			out = tf.layers.conv2d(
				out, 256, [5, 5], [2, 2],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=True)
			out = leaky_relu(out)

			tags_vectors = tf.expand_dims(tf.expand_dims(tags_vectors, 1), 2)
			tags_vectors = tf.tile(tags_vectors, [1, 6, 6, 1])

			condition_info = tf.concat([out, tags_vectors], axis=-1)

			out = tf.layers.conv2d(
				condition_info, 256, [1, 1], [1, 1],
				padding='same',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.layers.batch_normalization(out, training=True)
			out = leaky_relu(out)

			out = tf.layers.conv2d(
				out, 1, [6, 6], [1, 1],
				padding='valid',
				kernel_initializer=tf.random_normal_initializer(stddev=0.02),
				activation=None
				)
			out = tf.squeeze(out, [1, 2, 3])

			return out
	@property
	def vars(self):
		return [var for var in tf.global_variables() if "d_net" in var.name]


