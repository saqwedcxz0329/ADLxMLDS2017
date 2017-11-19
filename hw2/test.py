import sys
import os

import numpy as np
import tensorflow as tf

from loader import Loader
from s2vt import S2VT


#### Global parameters ####
data_foleder = sys.argv[1] if len(sys.argv) > 1 else './MLDS_hw2_data'
test_output_file_name = sys.argv[2] if len(sys.argv) > 2 else './test_output.txt'
peer_output_file_name = sys.argv[3] if len(sys.argv) > 3 else './peer_output.txt'

peer_review_folder =  os.path.join(data_foleder, 'peer_review', 'feat/')
peer_review_id = os.path.join(data_foleder, 'peer_review_id.txt')

testing_folder =  os.path.join(data_foleder, 'testing_data', 'feat/')
testing_id = os.path.join(data_foleder, 'testing_id.txt')

model_path = './models'

n_epochs = 200
batch_size = 64
dim_hidden = 256
learning_rate = 0.001

n_caption_lstm_step = 25
#### Global parameters ####

def download_model():
    file_name = 'model_special.zip'
    if not os.path.isfile(file_name):
        os.system('wget \'https://www.dropbox.com/s/mig2vjo3m19x41s/{}?dl=1\''.format(file_name))
        os.system('mv {}?dl=1 {}'.format(file_name, file_name))
        os.system('unzip {} -d {}'.format(file_name, model_path))

def test(out_put_file_name, id_file_name, feature_folder, model_name):
    ##### Preprcessing ####
    loader = Loader()
    x_test, id_list = loader.read_test_data(id_file_name, feature_folder)

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

    output_file = open(out_put_file_name, 'w')
    for caption_index, viedo_id in zip(generated_words_index, id_list):
        caption_words = []
        for idx in caption_index:
            word = idx_to_word[idx]
            if word != '<pad>' and word != '<bos>' and word != '<eos>':
                caption_words.append(word)
        sentence = ' '.join(caption_words)
        output_file.write(viedo_id + ',' + sentence + '\n')
    output_file.close()

if __name__ == '__main__':
    #download_model()
    test(test_output_file_name, testing_id, testing_folder, 'model-199')
    # test(peer_output_file_name, peer_review_id, peer_review_folder, 'model-special')
