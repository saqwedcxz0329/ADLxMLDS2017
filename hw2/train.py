import sys
import os
import time
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

from loader import Loader
from s2vt import S2VT

#### Global parameters ####
data_foleder = sys.argv[1] if len(sys.argv) > 1 else './MLDS_hw2_data'

training_folder =  os.path.join(data_foleder, 'training_data', 'feat/')
training_label = os.path.join(data_foleder, 'training_label.json')

model_path = './models'

n_epochs = 200
batch_size = 64
dim_hidden = 256
learning_rate = 0.001

n_caption_lstm_step = 25
#### Global parameters ####

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

    ##### Preprcessing ####
    loader = Loader()
    id_to_captions = loader.read_captions(training_label)
    x_train, x_train_label = loader.read_data(training_folder, id_to_captions=id_to_captions)
    word_to_idx, idx_to_word = build_vocab(x_train_label)
    x_train_label, x_train_mask = transfer_to_index(x_train_label, word_to_idx, n_caption_lstm_step)

    np.save("./wordtoix", word_to_idx)
    np.save('./ixtoword', idx_to_word)

    n_datas = x_train.shape[0]
    n_video_lstm_step = x_train.shape[1]
    dim_image = x_train.shape[2]
    
    model = S2VT(
            dim_image=dim_image,
            n_words=len(word_to_idx),
            dim_hidden=dim_hidden,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step)
    
    tf_loss, tf_video, tf_caption, tf_caption_mask, tf_acc = model.build_model(is_training=True)

    ##### Strat ####
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.111)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    saver = tf.train.Saver(max_to_keep=10, write_version=tf.train.SaverDef.V2)
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(n_epochs):
        choosed_label = [random.randint(0, len(captions)-1) for captions in x_train_label]
        start_time = time.time()
        for start in range(0, n_datas, batch_size):

            end = start + batch_size if start + batch_size < n_datas else n_datas
            current_features = x_train[start:end]
            current_label = x_train_label[start:end]
            current_mask = x_train_mask[start:end]

            current_choosed = choosed_label[start:end]
            current_captions = np.array([captions[current_choosed[index]] for index, captions in enumerate(current_label)], dtype=np.int32)
            current_caption_masks = np.array([masks[current_choosed[index]] for index, masks in enumerate(current_mask)], dtype=np.float32)
            
            sess.run(
                    [train_op],
                    feed_dict={
                        tf_video: current_features,
                        tf_caption: current_captions,
                        tf_caption_mask: current_caption_masks
                        })

            acc_val, loss_val = sess.run(
                [tf_acc, tf_loss],
                feed_dict={
                        tf_video: current_features,
                        tf_caption: current_captions,
                        tf_caption_mask: current_caption_masks
                        })

        print('size: {} Epoch: {} loss: {:.3f} acc: {:.3f} Elapsed time: {:.3f}'.format(end, epoch, loss_val, acc_val, time.time() - start_time))
        
        if np.mod(epoch, 1) == 0:
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


if __name__ == '__main__':
    train()