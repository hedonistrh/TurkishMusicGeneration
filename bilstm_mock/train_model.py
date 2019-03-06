from create_bidirectional_lstm_model import create_bidirectional_lstm_model as create_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import sys
import glob
import os
import numpy as np

def train_model(rnn_size, learning_rate=0.001, batch_size=32, num_epochs=40):
    """[We will train our model which is with given arguments. We will use
    input and output sequences which are created by preprocess.py and they
    are stored in ./feature_storage/
    
    Arguments:
        rnn_size {[int]} -- Size of each LSTM layer
    
    Keyword Arguments:
        learning_rate {float} -- Learning rate of our model (default: {0.01})
        batch_size {int} -- Batch size  (default: {32})
        num_epochs {int} -- Number of epochs (default: {40})
    """

    input_sequences = np.load("./feature_storage/input.npy")
    output_sequences = np.load("./feature_storage/output.npy")
    seq_length, vocab_size = input_sequences.shape[1], input_sequences.shape[2]
    print ("Sequence length: {} and Vocab size: {}".format(seq_length, vocab_size))

    model = create_model(int(seq_length), int(vocab_size),
            int(rnn_size), float(learning_rate))
    print (model.summary())

    save_dir = "./weights/"
    if ((os.path.isdir(save_dir))==False):
        os.mkdir(save_dir)
    else:
        pass

    callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
            ModelCheckpoint(filepath=save_dir + "/" + 'my_model_gen_koma53.{epoch:02d}-{val_loss:.2f}.hdf5',\
                            monitor='val_loss', verbose=0, mode='auto', period=2)]

    history = model.fit(input_sequences, output_sequences,
                    batch_size=int(batch_size),
                    shuffle=True,
                    epochs=int(num_epochs),
                    callbacks=callbacks,
                    validation_split=0.1)

    model.save(save_dir + "/" + 'my_model_generate_koma53.h5')

if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2],
                sys.argv[3], sys.argv[4])