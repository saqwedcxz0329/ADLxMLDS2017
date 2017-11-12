import sys
import os
import time
import random

import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence

from loader import Loader

class S2VT(object):
    def __init__(self, dim_image, n_words, dim_hidden, n_video_lstm_step, n_caption_lstm_step):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.n_video_lstm_step=n_video_lstm_step
        self.n_caption_lstm_step=n_caption_lstm_step

        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, state_is_tuple=False)
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

    def build_model(self, is_training=True):
        video = tf.placeholder(tf.float32, [None, self.n_video_lstm_step, self.dim_image])

        if is_training:
            caption = tf.placeholder(tf.int32, [None, self.n_caption_lstm_step+1])
            caption_mask = tf.placeholder(tf.float32, [None, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [-1, self.n_video_lstm_step, self.dim_hidden])
        
        zeros_dims = tf.stack([tf.shape(video)[0], self.lstm1.state_size])
        state1 = tf.fill(zeros_dims, 0.0)

        zeros_dims = tf.stack([tf.shape(video)[0], self.lstm2.state_size])
        state2 = tf.fill(zeros_dims, 0.0)
        
        zeros_dims = tf.stack([tf.shape(video)[0], self.dim_hidden])
        padding = tf.fill(zeros_dims, 0.0)

        loss = 0.0
        acc = 0.0
        generated_words = None

        #### Encoding Stage ####
        for i in range(0, self.n_video_lstm_step):
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)
                tf.get_variable_scope().reuse_variables()
                

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
                tf.get_variable_scope().reuse_variables()
                
        #### Decoding Stage ####
        for i in range(0, self.n_caption_lstm_step):
            if i == 0:
                with tf.device('/cpu:0'):
                    ones_dims = tf.stack([tf.shape(video)[0]])
                    boses = tf.fill(ones_dims, 1)
                    current_embed = tf.nn.embedding_lookup(self.Wemb, boses)

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)
                tf.get_variable_scope().reuse_variables()
            
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            pred_words = tf.argmax(logit_words, -1)
            
            flip = random.random() if is_training else 0.0

            if flip > 0.5:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i+1])
            else:
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, pred_words)

            if is_training:
                labels = caption[:, i+1]
                onehot_labels = tf.one_hot(indices=labels, depth=self.n_words)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * caption_mask[:,i+1]
                current_loss = tf.reduce_mean(cross_entropy)
                loss = loss + current_loss

                # accuracy
                truth_labels = tf.argmax(onehot_labels, -1)
                current_acc = tf.equal(pred_words, truth_labels)
                current_acc = tf.cast(current_acc, tf.float32)

                # mask
                mask = tf.cast(caption_mask[:,i+1], dtype=tf.float32)
                current_acc *= mask
                current_acc = tf.reduce_mean(current_acc)
                acc = acc + current_acc
            else:
                pred_words = tf.reshape(pred_words, [-1, 1])
                if generated_words is None:
                    generated_words = pred_words
                else:
                    generated_words = tf.concat([generated_words, pred_words], axis=1)
            
        acc = acc / self.n_caption_lstm_step
        
        if is_training:
            return loss, video, caption, caption_mask, acc
        else:
            return video, generated_words


#### Global parameters ####
data_foleder = sys.argv[1] if len(sys.argv) > 1 else './MLDS_hw2_data'
output_file_name = sys.argv[2] if len(sys.argv) > 2 else './test_output.txt'

training_folder =  os.path.join(data_foleder, 'training_data', 'feat/')
testing_folder =  os.path.join(data_foleder, 'testing_data', 'feat/')
training_label = os.path.join(data_foleder, 'training_label.json')
testing_id = os.path.join(data_foleder, 'testing_id.txt')

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

def test(file_name, model_name):
    ##### Preprcessing ####
    loader = Loader()
    x_test, id_list = loader.read_test_data(testing_id, testing_folder)

    idx_to_word = np.load('./ixtoword.npy').tolist()

    n_datas = x_test.shape[0]
    n_video_lstm_step = x_test.shape[1]
    dim_image = x_test.shape[2]

    model = S2VT(
            dim_image=dim_image,
            n_words=len(idx_to_word),
            dim_hidden=dim_hidden,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step)

    tf_video, tf_generated_words = model.build_model(is_training=False)

    ##### Strat ####
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(model_path, model_name))

    generated_words_index = sess.run(
                tf_generated_words,
                feed_dict={
                        tf_video: x_test
                        })

    output_file = open(file_name, 'w')
    for caption_index, viedo_id in zip(generated_words_index, id_list):
        caption_words = []
        for idx in caption_index:
            word = idx_to_word[idx]
            if word != '<pad>' and word != '<bos>' and word != '<eos>':
                caption_words.append(word)
        sentence = ' '.join(caption_words)
        output_file.write(viedo_id + ',' + sentence + '\n')
    output_file.close()

def download_model():
    file_name = 'model_special.zip'
    if not os.path.isfile(file_name):
        os.system('wget \'https://www.dropbox.com/s/mig2vjo3m19x41s/{}?dl=1\''.format(file_name))
        os.system('mv {}?dl=1 {}'.format(file_name, file_name))
        os.system('unzip {} -d {}'.format(file_name, model_path))

if __name__ == '__main__':
    train()
    # download_model()
    # test(output_file_name, 'model-0')
