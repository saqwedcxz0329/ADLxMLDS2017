import random

import tensorflow as tf

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