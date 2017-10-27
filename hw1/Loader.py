from keras.utils import np_utils
import numpy as np

class Loader():
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.cate_idx = {'aa': 1,'ae': 2,'ah': 3,'ao': 4,'aw': 5,'ax': 6,'ay': 7,'b': 8,'ch': 9,'cl': 10,'d': 11,'dh': 12,'dx': 13,'eh': 14,'el': 15,'en': 16,'epi': 17,'er': 18,'ey': 19,'f': 20,'g': 21,'hh': 22,'ih': 23,'ix': 24,'iy': 25,'jh': 26,'k': 27,'l': 28,'m': 29,'n': 30,'ng': 31,'ow': 32,'oy': 33,'p': 34,'r': 35,'s': 36,'sh': 37,'sil': 38,'t': 39,'th': 40,'uh': 41,'uw': 42,'v': 43,'vcl': 44,'w': 45,'y': 46,'z': 47,'zh': 48}
        self.class_num = len(self.cate_idx) + 1

        self.parse_phone_39(self.data_folder + '/phones/48_39.map')
        self.parse_phone_char(self.data_folder + '/48phone_char.map')

    def load_training_data(self, feature_file_path, train_file_path):
        instance_label = self.parse_train_label(train_file_path)
        prev_sentence_id = None
        max_length = 0
        sentence, label, X, Y, X_length = [], [], [], [], []
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
                if prev_sentence_id is not None and prev_sentence_id != sentence_id:
                    X.append(sentence)
                    X_length.append(len(sentence))
                    Y.append(np_utils.to_categorical(label, self.class_num))
                    # print('sentence length: {}'.format(len(sentence)))
                    max_length = max(len(sentence), max_length)
                    sentence = []
                    label = []
                feature = np.array(feature, dtype='float32')
                feature = (feature - np.mean(feature)) / np.std(feature)
                sentence.append(feature)
                label.append(self.cate_idx[instance_label[instance_id]])
                prev_sentence_id = sentence_id

            # append last sentence
            X.append(sentence)
            X_length.append(len(sentence))
            Y.append(np_utils.to_categorical(label, self.class_num))
            max_length = max(len(sentence), max_length)
        return X, X_length, Y, max_length

    def load_testing_data(self, feature_file_path):
        prev_sentence_id = None
        X, X_length, X_id, sentence = [], [], [], []
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
                if prev_sentence_id is not None and prev_sentence_id != sentence_id:
                    sentence_feature[prev_sentence_id] = sentence
                    X.append(sentence)
                    X_length.append(len(sentence))
                    X_id.append(prev_sentence_id)
                    max_length = max(len(sentence), max_length)
                    sentence = []
                feature = np.array(feature, dtype='float32')
                feature = (feature - np.mean(feature)) / np.std(feature)
                sentence.append(feature)
                prev_sentence_id = sentence_id

            # append last sentence
            sentence_feature[prev_sentence_id] = sentence
            X.append(sentence)
            X_length.append(len(sentence))
            X_id.append(prev_sentence_id)

        return X, X_length, X_id, max_length

    def transfer_csv(self, Y, X_length, X_id, file_name):
        Y = self._transfer_to_phone(Y)
        Y = self._trim_padding(Y, X_length)
        Y = self._trim_wrong_predict(Y)
        Y = self._trim_duplicate(Y)
        Y = self._trim_sil(Y)

        file = open(file_name, 'w')
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