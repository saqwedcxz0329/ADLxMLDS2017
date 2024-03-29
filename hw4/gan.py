import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
import time
import os
from model import Generator, Discriminator
import utils

class GAN(object):
	def __init__(self, data, FLAGS):
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.per_process_gpu_memory_fraction = 0.5
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)
		self.data = data
		self.FLAGS = FLAGS
		self.img_row = self.data.img_feat.shape[1] if self.data.img_feat is not None else 96
		self.img_col = self.data.img_feat.shape[2] if self.data.img_feat is not None else 96
		self.d_epoch = 1

	def gen_path(self):
		# Output directory for models and summaries
		timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
		self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
		print ("Writing to {}\n".format(self.out_dir))
		# Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)

	def build_model(self):

		self.g_net = Generator()
		self.d_net = Discriminator()

		self.seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_color)+len(self.data.hair_color)], name="seq")
		self.img = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="img")
		self.z = tf.placeholder(tf.float32, [None, self.FLAGS.z_dim])

		self.w_seq = tf.placeholder(tf.float32, [None, len(self.data.eyes_color)+len(self.data.hair_color)], name="w_seq")
		self.w_img = tf.placeholder(tf.float32, [None, self.img_row, self.img_col, 3], name="w_img")

		r_img, r_seq = self.img, self.seq

		self.f_img = self.g_net(r_seq, self.z)
		
		self.sampler = tf.identity(self.g_net(r_seq, self.z, reuse=True, train=False), name='sampler') 

		# TODO 
		"""
			r img, r text -> 1
			f img, r text -> 0
			r img, w text -> 0
			w img, r text -> 0
		"""
		self.d = self.d_net(r_seq, r_img, reuse=False) 	# r img, r text
		self.d_1 = self.d_net(r_seq, self.f_img) 		# f img, r text
		self.d_2 = self.d_net(self.w_seq, r_img)		# r img, w text
		self.d_3 = self.d_net(r_seq, self.w_img)		# w img, r text

		# dcgan
		self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.ones_like(self.d_1))) 

		self.d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d, labels=tf.ones_like(self.d))) \
					+ (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_1, labels=tf.zeros_like(self.d_1))) + \
						tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_2, labels=tf.zeros_like(self.d_2))) +\
						tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_3, labels=tf.zeros_like(self.d_3))) ) / 3 
		

		self.global_step = tf.Variable(0, name='g_global_step', trainable=False)

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.d_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_net.vars)
			self.g_updates = tf.train.AdamOptimizer(self.FLAGS.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_net.vars, global_step=self.global_step)

		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

	def train(self):
		self.gen_path()
		
		print("Start training WGAN...\n")

		for t in range(self.FLAGS.iter):

			d_cost = 0
			g_coat = 0

			for d_ep in range(self.d_epoch):

				img, tags, w_img, w_tags = self.data.next_data_batch(self.FLAGS.batch_size)
				z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)

				feed_dict = {
					self.seq:tags,
					self.img:img,
					self.z:z,
					self.w_seq:w_tags,
					self.w_img:w_img
				}

				_, loss = self.sess.run([self.d_updates, self.d_loss], feed_dict=feed_dict)

				d_cost += loss/self.d_epoch

			z = self.data.next_noise_batch(len(tags), self.FLAGS.z_dim)
			feed_dict = {
				self.img:img,
				self.w_seq:w_tags,
				self.w_img:w_img,
				self.seq:tags,
				self.z:z
			}

			_, loss, step = self.sess.run([self.g_updates, self.g_loss, self.global_step], feed_dict=feed_dict)

			current_step = tf.train.global_step(self.sess, self.global_step)

			g_cost = loss

			if current_step % self.FLAGS.display_every == 0:
				print("Epoch {}, Current_step {}".format(self.data.epoch, current_step))
				print("Discriminator loss :{}".format(d_cost))
				print("Generator loss     :{}".format(g_cost))
				print("---------------------------------")

			if current_step % self.FLAGS.checkpoint_every == 0:
				path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
				print ("\nSaved model checkpoint to {}\n".format(path))

			if current_step % self.FLAGS.dump_every == 0:
				self.eval(current_step)
				print("Dump test image")

	def eval(self, iters):
		
		z = self.data.fixed_z
		feed_dict = {
			self.seq:self.data.test_tag_one_hot,
			self.z:z
		}

		f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)

		utils.dump_img(self.FLAGS.img_dir, f_imgs, iters)

	def restore(self, model_path, model_name):
		self.saver.restore(self.sess, os.path.join(model_path, model_name))

	def gen_test_img(self):
		size = len(self.data.test_tag_one_hot)
		z_dim = self.FLAGS.z_dim
		np.random.seed(2)

		for sample_id in range(5):
			z = np.random.normal(0.0, 1.0, [size, z_dim])

			feed_dict = {
				self.seq:self.data.test_tag_one_hot,
				self.z:z
			}

			f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)

			utils.dump_test_img(self.FLAGS.img_dir, f_imgs, sample_id)


	# def gen_test_img(self, name):

	# 	size = len(self.data.test_tag_one_hot)
	# 	z_dim = self.FLAGS.z_dim
	# 	np.random.seed(2)

	# 	img_feats = []

	# 	for _ in range(5):
	# 		z = np.random.normal(0.0, 1.0, [size, z_dim])

	# 		feed_dict = {
	# 			self.seq:self.data.test_tag_one_hot,
	# 			self.z:z
	# 		}

	# 		f_imgs = self.sess.run(self.sampler, feed_dict=feed_dict)
	# 		img_feats.append(f_imgs)
		
	# 	img_feats = np.array(img_feats)

	# 	utils.dump_test_img(self.FLAGS.img_dir, img_feats, name)
