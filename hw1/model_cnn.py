from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional, Conv2D, MaxPooling2D
from keras.layers import Reshape, Masking, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import keras.preprocessing.sequence
import numpy as np
import tensorflow as tf
import sys
from Loader import Loader

data_folder = sys.argv[1] # change to sys.argv
output_filename = sys.argv[2]
model_name = './cnn_model.h5'
batch_size = 64
epochs = 15
class_num = 48 + 1 # 1~48 + padding 0
validation_size = 100

def build_model(timesteps, vector_size):
    print('Build model...')
    model = Sequential()

    model.add(Conv2D(filters=10, kernel_size=[5, 5], padding='same', input_shape=(timesteps, vector_size, 1)))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Conv2D(filters=15, kernel_size=[5, 5], padding='same'))
    # model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation("tanh"))
    model.add(Reshape((timesteps, -1)))
    model.add(Bidirectional(LSTM(256, activation='tanh', dropout=0.5, return_sequences=True)))
    model.add(Bidirectional(LSTM(256, activation='tanh', dropout=0.5, return_sequences=True)))
    model.add(TimeDistributed(Dense(class_num, activation='softmax')))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  sample_weight_mode='temporal'
                  )
    return model

def train():
    loader = Loader(data_folder)
    X, X_length, Y, max_length = loader.load_training_data(data_folder + '/fbank/train.ark', data_folder + '/label/train.lab')
    print('max length: {}'.format(max_length))

    X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='float32', maxlen=max_length, padding='post')
    Y_padded = keras.preprocessing.sequence.pad_sequences(Y, dtype='float32', maxlen=max_length, padding='post')
    X_padded = np.expand_dims(X_padded, axis=3)

    print('X length: {}'.format(len(X_length)))
    print('Y length: {}'.format(len(Y_padded)))
    sample_weightes = []
    for length, sentence in zip(X_length, Y_padded):
        sentence[length:, 0] = 1.
        weights = [0.] * len(sentence)
        weights[:length] = [1. for _ in range(length)]
        sample_weightes.append(weights)
    sample_weightes = np.array(sample_weightes, dtype='float32')
    sample_weightes = sample_weightes[:X_padded.shape[0]-validation_size]

    x_val = X_padded[X_padded.shape[0]-validation_size:]
    y_val = Y_padded[X_padded.shape[0]-validation_size:]
    x_train = X_padded[:X_padded.shape[0]-validation_size]
    y_train = Y_padded[:X_padded.shape[0]-validation_size]
    print('X shape: {}'.format(X_padded.shape))
    print('Y shape: {}'.format(Y_padded.shape))

    lstm_model = build_model(X_padded.shape[1], X_padded.shape[2])
    print(lstm_model.summary())

    callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=2, verbose=0, mode='auto'),
                keras.callbacks.ModelCheckpoint(model_name, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                ]
    try:            
        lstm_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs, 
              validation_data=(x_val, y_val),
              #callbacks=callbacks,
              sample_weight=sample_weightes
              )
    except KeyboardInterrupt:
        lstm_model.save(model_name)
    
    lstm_model.save(model_name)


def test():
    loader = Loader(data_folder)
    Test_X, Test_X_length, Test_X_id, max_length = loader.load_testing_data(data_folder + '/fbank/test.ark')
    Test_X_padded = keras.preprocessing.sequence.pad_sequences(Test_X, dtype='float32', maxlen=777, padding='post')
    Test_X_padded = np.expand_dims(Test_X_padded, axis=3)
    print('max length: {}'.format(max_length))
    print('Test X shape: {}'.format(Test_X_padded.shape))

    lstm_model = load_model(model_name)
    Predict_Y = lstm_model.predict(Test_X_padded)

    loader.transfer_csv(Predict_Y, Test_X_length, Test_X_id, output_filename)
    print('Finished')

if __name__ == '__main__':
    # train()
    test()
        