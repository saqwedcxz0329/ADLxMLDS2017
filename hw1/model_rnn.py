from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Bidirectional
from keras.models import load_model
import keras.preprocessing.sequence
import numpy as np
import tensorflow as tf
import sys
from Loader import Loader


data_folder = sys.argv[1]
output_filename = sys.argv[2]
model_name = './rnn_model.h5'
batch_size = 128
epochs = 50
class_num = 48 + 1 # 1~48 + padding 0
validation_size = 100

def build_model(timesteps, vector_size):
    print('Build model...')
    model = Sequential()

    model.add(Bidirectional(LSTM(256, activation='tanh', dropout=0.5, return_sequences=True), input_shape=(timesteps, vector_size)))
    model.add(Bidirectional(LSTM(256, activation='tanh', dropout=0.5, return_sequences=True)))
    model.add(TimeDistributed(Dense(class_num, activation='softmax')))

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  )
    return model

def train():
    loader = Loader(data_folder)
    X, X_length, Y, max_length = loader.load_training_data(data_folder + '/fbank/train.ark', data_folder + '/label/train.lab')
    print('max length: {}'.format(max_length))

    X_padded = keras.preprocessing.sequence.pad_sequences(X, dtype='float32', maxlen=max_length, padding='post')
    Y_padded = keras.preprocessing.sequence.pad_sequences(Y, dtype='float32', maxlen=max_length, padding='post')

    print('X length: {}'.format(len(X_length)))
    print('Y length: {}'.format(len(Y_padded)))

    for length, sentence in zip(X_length, Y_padded):
        sentence[length:, 0] = 1.

    x_val = X_padded[X_padded.shape[0]-validation_size:]
    y_val = Y_padded[X_padded.shape[0]-validation_size:]
    x_train = X_padded[:X_padded.shape[0]-validation_size]
    y_train = Y_padded[:X_padded.shape[0]-validation_size]
    print('X shape: {}'.format(X_padded.shape))
    print('Y shape: {}'.format(Y_padded.shape))

    lstm_model = build_model(X_padded.shape[1], X_padded.shape[2])
    print(lstm_model.summary())

    callbacks = [
                keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'),
                keras.callbacks.ModelCheckpoint(model_name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
                ]
    lstm_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs, 
          validation_data=(x_val, y_val),
          callbacks=callbacks
          )

    lstm_model.save(model_name)

def test():
    loader = Loader(data_folder)
    Test_X, Test_X_length, Test_X_id, max_length = loader.load_testing_data(data_folder + '/fbank/test.ark')
    Test_X_padded = keras.preprocessing.sequence.pad_sequences(Test_X, dtype='float32', maxlen=777, padding='post')
    print('max length: {}'.format(max_length))
    print('Test X shape: {}'.format(Test_X_padded.shape))

    lstm_model = load_model(model_name)
    Predict_Y = lstm_model.predict(Test_X_padded)

    loader.transfer_csv(Predict_Y, Test_X_length, Test_X_id, output_filename)
    print('Finished')

if __name__ == '__main__':
    train()
    test()
        