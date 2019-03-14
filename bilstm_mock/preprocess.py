import glob, os, sys, wget, collections
from six.moves import cPickle
import pandas as pd
from collections import Counter
import numpy as np

def find_nearest(array, note):
    """Find the nearest common note for the uncommon note.
    
    Arguments:
        array {numpy array} -- Store the common notes.
        note {koma53 or komaAE value} -- Uncommon note value
    
    Returns:
        {koma53 or komaAE value} -- Common note value
    """

    array = np.asarray(array)
    idx = (np.abs(array - note)).argmin()
    return (array[idx])

def map_notes(value_counts_of_note, threshold):
    """This function takes value counts for the Koma53-KomaAe 
    and gives the dict to represent which note should be 
    transformed into which note to get rid of uncommon
    note problem.
    
    Arguments:
    valueCountsOfNote: Value counts of Koma53 or KomaAe 
    threshold: What is the minimum number 
        to kept this note as the common note.
        
    Returns:
    map_dict: Dictionary whose key is uncommon notes and v
        value is nearest common note."""
    
    up_thres_key_list = [] # To store, which key has more value than threshold
    map_dict = {} # This dict's keys store the uncommon notes and value store
                 # common notes.
        
    for key, value in value_counts_of_note.items():
        if (value > threshold):
            up_thres_key_list.append(key)
            
    up_thres_key_array = np.asarray(up_thres_key_list)
    
    for key, value in value_counts_of_note.items():
        if key not in up_thres_key_list:
            near_note = find_nearest(up_thres_key_array, key)
            map_dict[key] = near_note
    return map_dict


def preprocess(map_function, sequence_length):
    """This function will convert to our .txt inputs into
    appropriate numpy array for our input and output. 
    
    Arguments:
        map_function {Bool} -- If it is True, we will map uncommon notes into
            nearest note.
        sequence_length {int} -- To guess next sample, how many samples will be 
            used as input.
    """

    root_dir = "./txt/"
    if ((os.path.isdir(root_dir))==False):
        print ("""Please Download Txt File via this
        link https://minhaskamal.github.io/DownGit/#/home?url=https://github.com/MTG/SymbTr/tree/master/txt""")
        raise Exception
    else:
        pass

    root_dir = glob.glob(os.path.join(root_dir, "*txt"))
    df_hicaz_sarki = pd.concat((pd.read_csv(f, 
                sep="\t") for f in root_dir if "hicaz--sarki" in f))

    mapped_koma53 = map_function

    if (mapped_koma53 == "True"):
        mapped_koma53_dict = (map_notes(pd.value_counts(df_hicaz_sarki['Koma53']), 250))
        df_hicaz_sarki["Koma53"] = df_hicaz_sarki.Koma53.replace(to_replace=mapped_koma53_dict)

    print (pd.value_counts(df_hicaz_sarki['Koma53']))

    total_value = 0
    for key, value in pd.value_counts(df_hicaz_sarki['Koma53']).items():
        total_value += value;

    print ("Total Koma53 value: ", total_value)
    note_list = df_hicaz_sarki['Koma53'].tolist()
    note_list = list(map(str, note_list))

    save_dir = 'save' # To store trained DL model
    if ((os.path.isdir(save_dir))==False):
        os.mkdir(save_dir)
    else:
        pass

    vocab_file = os.path.join(save_dir, "words_vocab.pkl")

    # Count the number of words
    word_counts = collections.Counter(note_list)

    # Mapping from index to word : that's the vocabulary
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))

    # Mapping from word to index
    vocab = {x: i for i, x in enumerate(vocabulary_inv)}
    words = [x[0] for x in word_counts.most_common()]

    # Size of vocabulary
    vocab_size = len(words)
    print("vocab size: ", vocab_size)

    # Save the words and vocabulary as a pickle file
    with open(os.path.join(vocab_file), 'wb') as f:
        cPickle.dump((words, vocab, vocabulary_inv), f)

    """We need to create two different list. One list include the previous words, 
    another list inclued the next word."""

    seq_length = int(sequence_length)
    sequences_step = 1

    print ("Seed Length is {}".format(seq_length))

    sequences = []
    next_words = []
    for i in range(0, len(note_list) - seq_length, sequences_step):
        sequences.append(note_list[i: i + seq_length])
        next_words.append(note_list[i + seq_length])

    print('nb sequences:', len(sequences))

    # We can not use this type of array directly. So that, we have to modify
    # this data to use with LSTM. We need 
    # to convert into one-hot vector type array. 

    # List which includes previous words should have a dimension as number 
    # of sequences,  number of words in sequences, number of words in vocabulary. 
    # The other list should have a dimension as number of sequences, 
    # number of words in vocabulary.

    input_sequence = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
    output_sequence = np.zeros((len(sequences), vocab_size), dtype=np.bool)
    for i, sentence in enumerate(sequences):
        for t, word in enumerate(sentence):
            input_sequence[i, t, vocab[word]] = 1
        output_sequence[i, vocab[next_words[i]]] = 1

    input_sequence = np.asarray(input_sequence)
    output_sequence = np.asarray(output_sequence)

    # Save Numpy arrays
    feature_storage_path = "./feature_storage/"
    if ((os.path.isdir(feature_storage_path))==False):
        os.mkdir(feature_storage_path)
    else:
        pass

    np.save(feature_storage_path + "input", input_sequence)
    np.save(feature_storage_path + "output", output_sequence)

if __name__ == "__main__":
    preprocess(sys.argv[1], sys.argv[2])
