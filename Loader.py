from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Conv2D, MaxPooling1D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.models import load_model
import keras.preprocessing.sequence
import numpy as np

class Loader():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.cate_idx = {'aa': 1,'ae': 2,'ah': 3,'ao': 4,'aw': 5,'ax': 6,'ay': 7,'b': 8,'ch': 9,'cl': 10,'d': 11,'dh': 12,'dx': 13,'eh': 14,'el': 15,'en': 16,'epi': 17,'er': 18,'ey': 19,'f': 20,'g': 21,'hh': 22,'ih': 23,'ix': 24,'iy': 25,'jh': 26,'k': 27,'l': 28,'m': 29,'n': 30,'ng': 31,'ow': 32,'oy': 33,'p': 34,'r': 35,'s': 36,'sh': 37,'sil': 38,'t': 39,'th': 40,'uh': 41,'uw': 42,'v': 43,'vcl': 44,'w': 45,'y': 46,'z': 47,'zh': 48}
        self.class_num = len(self.cate_idx) + 1

        self.parse_phone_39(self.data_folder + '/48_39.map')
        self.parse_phone_char(self.data_folder + '/48phone_char.map')

    def load_training_data(self, feature_file_path, train_file_path):
        instance_label = self.parse_train_label(train_file_path)
        prev_sentence_id = None
        max_length = 0
        sentence, label, X, Y = [], [], [], []
        with open(feature_file_path) as file:
            for line in file:
                line = line.strip()
                tmp = line.split(' ')
                instance_id = tmp[0]
                feature = tmp[1:]
                tmp = instance_id.split('_')
                sentence_id = tmp[0] + '_' + tmp[1]
                # print('prev: {}\ncur: {}'.format(prev_sentence_id, sentence_id))
                # print('=========')
                if prev_sentence_id is None:
                    prev_sentence_id = sentence_id
                if prev_sentence_id == sentence_id:
                    feature = np.array(feature, dtype='float32')
                    feature = (feature - np.mean(feature)) / np.std(feature)
                    sentence.append(feature)
                    label.append(self.cate_idx[instance_label[instance_id]])
                else:
                    X.append(sentence)
                    Y.append(np_utils.to_categorical(label, self.class_num))
                    # print('sentence length: {}'.format(len(sentence)))
                    max_length = max(len(sentence), max_length)
                    sentence = []
                    label = []
                prev_sentence_id = sentence_id

            # append last sentence
            X.append(sentence)
            Y.append(np_utils.to_categorical(label, self.class_num))
            max_length = max(len(sentence), max_length)
        return X, Y, max_length

    def load_testing_data(self, feature_file_path, test_file_path):
        prev_sentence_id = None
        sentence = []
        sentence_feature = {}
        max_length = 0
        with open(feature_file_path) as file:
            for line in file:
                line = line.strip()
                tmp = line.split(' ')
                instance_id = tmp[0]
                feature = tmp[1:]
                tmp = instance_id.split('_')
                sentence_id = tmp[0] + '_' + tmp[1]
                if prev_sentence_id is None:
                    prev_sentence_id = sentence_id
                if prev_sentence_id == sentence_id:
                    feature = np.array(feature, dtype='float32')
                    feature = (feature - np.mean(feature)) / np.std(feature)
                    sentence.append(feature)
                else:
                    sentence_feature[prev_sentence_id] = sentence
                    max_length = max(len(sentence), max_length)
                    sentence = []
                prev_sentence_id = sentence_id

            # append last sentence
            sentence_feature[prev_sentence_id] = sentence

        X = []
        X_length = []
        X_id = []
        with open(test_file_path) as file:
            file.readline()
            for line in file:
                line = line.strip()
                sentence_id = line[:-1]
                X.append(sentence_feature[sentence_id])
                X_length.append(len(sentence_feature[sentence_id]))
                X_id.append(sentence_id)
        return X, X_length, X_id, max_length

    def transfer_csv(self, Y, X_length, X_id):
        Y = self._transfer_to_phone(Y)
        Y = self._trim_padding(Y, X_length)
        Y = self._trim_wrong_predict(Y)
        Y = self._trim_duplicate(Y)
        Y = self._trim_sil(Y)

        file = open('submit.csv', 'w')
        file.write('id,phone_sequence\n')
        for index, sentence in enumerate(Y):
            sentence_id = X_id[index]
            phone_sequence = ''
            for phone in sentence:
                phone_sequence += self.phone_char[phone]
            file.write(sentence_id + ',' + phone_sequence + '\n')
        file.close()

    def _transfer_to_phone(self, Y):
        Y_phone = []
        idx_cate = self._get_indix_to_cate()
        for sentence in Y:
            phone_sentence = []
            for frame in sentence:
                max_idx = np.argmax(frame)
                try:
                    phone = idx_cate[max_idx]
                except KeyError:
                    phone = 'pad'
                phone_sentence.append(phone)
            Y_phone.append(phone_sentence)
        return Y_phone

    def _trim_padding(self, Y, X_length):
        Y_trimmed = []
        for index, sentence in enumerate(Y):
            Y_trimmed.append(sentence[:X_length[index]])
        return Y_trimmed

    def _trim_duplicate(self, Y):
        Y_trimmed = []
        for sentence in Y:
            prev_phone = None
            trimed_sentence = []
            for phone in sentence:
                if phone != 'pad':
                    cur_phone = self.phone_39[phone]
                    if prev_phone is None:
                        prev_phone = cur_phone
                    if prev_phone != cur_phone:
                        trimed_sentence.append(prev_phone)
                else:
                    cur_phone = 'pad'
                prev_phone = cur_phone
            # append the last phone
            if prev_phone != 'pad':
                trimed_sentence.append(prev_phone)
            Y_trimmed.append(trimed_sentence)
        return Y_trimmed

    def _trim_sil(self, Y):
        Y_trimmed = []
        for sentence in Y:
            trimmed_sentence = sentence
            leadding = self.phone_39[sentence[0]]
            tailing = self.phone_39[sentence[-1]]
            if leadding == 'sil':
                trimmed_sentence = trimmed_sentence[1:]            
            if tailing == 'sil':
                trimmed_sentence = trimmed_sentence[:-1]
            Y_trimmed.append(trimmed_sentence)
        return Y_trimmed

    def _trim_wrong_predict(self, Y):
        Y_trimmed = []
        for sentence in Y:
            for i in range(1, len(sentence)-1):
                if sentence[i-1] == sentence[i+1]:
                    sentence[i] = sentence[i-1]
            Y_trimmed.append(sentence)
        return Y_trimmed


    def _get_indix_to_cate(self):
        idx_cate = {}
        for category in self.cate_idx:
            idx_cate[self.cate_idx[category]] = category 
        return idx_cate


    def parse_train_label(self, path):
        instance_label = {}
        with open(path) as file:
            for line in file.readlines():
                line = line.strip()
                tmp = line.split(',')
                instance_label[tmp[0]] = tmp[1]
        return instance_label
            
    def parse_phone_char(self, path):
        self.phone_char = {}
        with open(path) as file:
            index = 1
            for line in file.readlines():
                line = line.strip()
                tmp = line.split('\t')
                phone = tmp[0]
                char = tmp[2]
                self.phone_char[phone] = char
                # print('\'{}\': {},'.format(phone, index), end = '')
                index += 1
        return self.phone_char
    
    def parse_phone_39(self, path):
        self.phone_39 = {}
        with open(path) as file:
            for line in file.readlines():
                line = line.strip()
                tmp = line.split('\t')
                phone = tmp[0]
                char = tmp[1]
                self.phone_39[phone] = char

data_folder = './data'
batch_size = 128
class_num = 48 + 1 # 1~48 + padding 0

def build_model(timesteps, vector_size):
    print('Build model...')
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=[5, 5], padding='same', activation='relu', input_shape=(timesteps, vector_size, 1)))
    model.add(Reshape((timesteps, -1)))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True), input_shape=(timesteps, vector_size)))
    model.add(Bidirectional(LSTM(256, activation='tanh', return_sequences=True), input_shape=(timesteps, vector_size)))
    model.add(TimeDistributed(Dense(class_num, activation='softmax'), input_shape=(timesteps, vector_size)))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(64, return_sequences=False))
    # model.add(Dense(class_num, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

def train():
    loader = Loader(data_folder)
    X, Y, max_length = loader.load_training_data(data_folder + '/mfcc/train.ark', data_folder + '/train.lab')
    print('max length: {}'.format(max_length))

    X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='float32', maxlen=max_length, padding='post')
    Y_padded = keras.preprocessing.sequence.pad_sequences(Y, dtype='float32', maxlen=max_length, padding='post')
    print('X shape: {}'.format(X_padded.shape))
    print('Y shape: {}'.format(Y_padded.shape))

    lstm_model = build_model(X_padded.shape[1], X_padded.shape[2])
    print('Train...')

    X_padded = np.expand_dims(X_padded, axis=3)
    print('X expand shape: {}'.format(X_padded.shape))

    lstm_model.fit(X_padded, Y_padded,
              batch_size=batch_size,
              epochs=100)
    score, acc = lstm_model.evaluate(X_padded, Y_padded, batch_size=batch_size)
    print ("score: %d" %score)
    print ("acc: %d" %acc)

    lstm_model.save('rnn_model.h5')
    # loader.parse_feature('./data/mfcc/train.ark')
    # instance_label = loader.parse_train_label('./data/train.lab')
    # loader.parse_phone_char('./data/48phone_char.map')
    # loader.parse_phone_39('./data/48_39.map')

def test():
    loader = Loader(data_folder)
    Test_X, Test_X_length, Test_X_id, max_length = loader.load_testing_data(data_folder + '/mfcc/test.ark', data_folder + '/sample.csv')
    Test_X_padded = keras.preprocessing.sequence.pad_sequences(Test_X, dtype='float32', maxlen=776, padding='post')
    print('max length: {}'.format(max_length))
    print('Test X shape: {}'.format(Test_X_padded.shape))

    lstm_model = load_model('rnn_model.h5')
    Predict_Y = lstm_model.predict(Test_X_padded)

    loader.transfer_csv(Predict_Y, Test_X_length, Test_X_id)

if __name__ == '__main__':
    # loader = Loader()
    # loader.parse_phone_char('./data/48phone_char.map')
    train()
    test()
        