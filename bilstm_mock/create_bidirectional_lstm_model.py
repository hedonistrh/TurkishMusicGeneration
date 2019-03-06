from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, GRU
from keras.layers import LSTM, Input, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.metrics import categorical_accuracy
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import TimeDistributed
from keras.models import load_model
import numpy as np

def create_bidirectional_lstm_model(seq_length, vocab_size, rnn_size=512,
                                    learning_rate=0.001):
    """It will create our bidirectional lstm model.
    
    Arguments:
        seq_length {int} -- To guess next sample, how many samples will be 
            used as input.
        vocab_size {int} -- How many distinct notes we have. :)
    
    Keyword Arguments:
        rnn_size {int} -- [description] (default: {512})
        learning_rate {float} -- [description] (default: {0.001})
    """
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu", return_sequences=True, 
                                 kernel_initializer='random_normal',
                                    bias_initializer='random_normal'),
                            input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.7))
    model.add(Bidirectional(LSTM(rnn_size, activation="relu", return_sequences=True, 
                                kernel_initializer='random_normal',
                                    bias_initializer='random_normal')))
    model.add(Dropout(0.7))
    model.add(Bidirectional(LSTM(rnn_size, activation="relu",
                                kernel_initializer='random_normal',
                                    bias_initializer='random_normal')))
    model.add(Dropout(0.3))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    print("model built!")
    return model

