import sys
import os
import time
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

from loader import Loader




class S2VT(object):
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_video_lstm_step, n_caption_lstm_step):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        # video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden])

        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, caption, caption_mask


data_foleder = sys.argv[1] if len(sys.argv) > 1 else './MLDS_hw2_data'
training_folder =  os.path.join(data_foleder, 'training_data', 'feat/')
testing_folder =  os.path.join(data_foleder, 'testing_data', 'feat/')
training_label = os.path.join(data_foleder, 'training_label.json')

n_epochs = 100
batch_size = 64
dim_hidden = 100
learning_rate = 0.1

n_caption_lstm_step = 35

def build_vocab(x_train_label):
    word_to_idx = {}
    idx_to_word = {}
    
    word_to_idx['<pad>'] = 0
    word_to_idx['<bos>'] = 1
    word_to_idx['<eos>'] = 2

    idx_to_word[0] = '<pad>'
    idx_to_word[1] = '<bos>'
    idx_to_word[2] = '<eos>'
    
    vocab = set()
    for captions in x_train_label:
        for sentence in captions:
            for word in sentence.lower().split(' '):
                vocab.add(word)
    for index, word in enumerate(vocab):
        word_to_idx[word] = index + 3
        idx_to_word[index + 3] = word
    
    return word_to_idx, idx_to_word

def transfer_to_index(x_train_label, word_to_idx, n_caption_lstm_step):
    new_x_train_label = []
    x_train_mask = []
    for captions in x_train_label:
        idx_captions = []
        for sentence in captions:
            sentence = '<bos> ' + sentence
            words = sentence.lower().split(' ')
            if len(words) >= n_caption_lstm_step:
                words = words[:n_caption_lstm_step-1]
            words.append('<eos>')
            idx_sentence = [word_to_idx[word] for word in words]
            idx_captions.append(idx_sentence)
        idx_captions = sequence.pad_sequences(idx_captions, padding='post', maxlen=n_caption_lstm_step)
        idx_captions = np.hstack( [idx_captions, np.zeros( [len(idx_captions), 1] ) ] ).astype(int)
        new_x_train_label.append(idx_captions)

        caption_mask = np.zeros([idx_captions.shape[0], idx_captions.shape[1]])
        nonzeros = np.array( list(map(lambda x: (x != 0).sum() + 1, idx_captions)) )
        for ind, row in enumerate(caption_mask):
            row[:nonzeros[ind]] = 1
        x_train_mask.append(caption_mask)

    return new_x_train_label, x_train_mask


def train():
    loader = Loader()
    id_to_captions = loader.read_captions(training_label)
    x_train, x_train_label = loader.read_data(training_folder, id_to_captions=id_to_captions)
    word_to_idx, idx_to_word = build_vocab(x_train_label)
    x_train_label, x_train_mask = transfer_to_index(x_train_label, word_to_idx, n_caption_lstm_step)

    n_datas = x_train.shape[0]
    n_video_lstm_step = x_train.shape[1]
    dim_image = x_train.shape[2]
    
    model = S2VT(
            dim_image=dim_image,
            n_words=len(word_to_idx),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step)
    
    tf_loss, tf_video, tf_caption, tf_caption_mask = model.build_model()
    sess = tf.Session()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess.run(tf.global_variables_initializer())
    # sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    
    for epoch in range(n_epochs):
        choosed_label = [random.randint(0, len(captions)-1) for captions in x_train_label]

        for start in range(0, n_datas, batch_size):
            start_time = time.time()

            end = start + batch_size if start + batch_size < n_datas else n_datas
            current_features = x_train[start:end]
            current_label = x_train_label[start:end]
            current_mask = x_train_mask[start:end]

            current_choosed = choosed_label[start:end]
            current_captions = np.array([captions[current_choosed[index]] for index, captions in enumerate(current_label)], dtype=np.int32)
            current_caption_masks = np.array([masks[current_choosed[index]] for index, masks in enumerate(current_mask)], dtype=np.float32)
            
            # print('feature: ', current_features.shape)
            # print('captions: ', current_captions.shape)
            # print('mask: ', current_caption_masks.shape)

            _, loss_val = sess.run(
                        [train_op, tf_loss],
                        feed_dict={
                            tf_video: current_features,
                            tf_caption: current_captions,
                            tf_caption_mask: current_caption_masks
                            })
            print('idx: {} Epoch: {} loss: {} Elapsed time: {}'.format(start, epoch, loss_val, time.time() - start_time))
if __name__ == '__main__':
    train()