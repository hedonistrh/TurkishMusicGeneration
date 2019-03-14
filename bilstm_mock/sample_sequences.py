import sys
import numpy as np
import os
import pandas as pd
from six.moves import cPickle
from keras.models import load_model
from create_bidirectional_lstm_model import create_bidirectional_lstm_model as create_model

def sample(preds, temperature=1.0):
    """Helper function to sample an index from a probability array
    
    Arguments:
        preds {numpy array} -- Array's values represent
            probability of each note
    
    Keyword Arguments:
        temperature {float} -- used to control the randomness 
            of predictions by scaling the logits before applying softmax 
            (default: {1.0})
    
    Returns:
        {int} -- index of selected sample
    """

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def sample_sequences(seq_length, rnn_size):
    """We will generate our samples. :)
    
    Arguments:
        seq_length {int} -- To guess next sample, how many samples will be 
            used as input. Have to be same with create_model's parameter.
        rnn_size {int} -- Size of each LSTM layer. 
            Have to be same with create_model's parameter.
    """


    seq_length = int(seq_length)
    rnn_size = int(rnn_size)
    save_dir = "./save/"
    # Load vocabulary
    print("loading vocabulary...")
    vocab_file = os.path.join(save_dir, "words_vocab.pkl")

    with open(os.path.join(save_dir, 'words_vocab.pkl'), 'rb') as f:
            words, vocab, vocabulary_inv = cPickle.load(f)

    vocab_size = len(words)

    model = create_model(seq_length, vocab_size, rnn_size)

    print("loading model...")
    model = load_model("./weights/my_model_generate_koma53.h5")

    # Iniatate sentence
    seed_sentences = "327 336 305 310 358 310 344"
    generated = ''
    sentence = []
    for i in range (seq_length):
        sentence.append("322")

    seed = seed_sentences.split()
    for i in range(len(seed)):
        sentence[seq_length-i-1]=seed[len(seed)-i-1]

    generated += ' '.join(sentence)
    print('Generating text with this seed:"' + ' '.join(sentence) + '"')

    words_number = 150

    print ("Our Vocabulary:", vocab)
    # Generate the sequences
    for i in range(words_number):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))

        for t, word in enumerate(sentence):
            x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.2)
        next_word = vocabulary_inv[next_index]

        #add the next word to the text
        generated += " " + next_word

        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]
    
    generated_notes_list = []
    for single_generated in generated.split(" "):
        generated_notes_list.append(int(single_generated))

    np.asarray(generated_notes_list)
    arr = np.asarray(generated_notes_list)
    generated_data_frame = pd.DataFrame({'Koma53':arr})
    print (pd.value_counts(generated_data_frame['Koma53']))

if __name__ == "__main__":
    sample_sequences(sys.argv[1], sys.argv[2])