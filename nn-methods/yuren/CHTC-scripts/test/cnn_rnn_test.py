"""
Test the trained model with specified testing and validation indices set
The results will be printed in the output file(s) from CHTC and
contain y_true and y_score to compute precision, recall, etc.
"""

import h5py
import numpy as np
import pandas as pd
import os
from itertools import product

import tensorflow as tf
from tensorflow import keras


# 1 - A, 2 - C, 3 - G, 4 - T, 0 - missing
dic_complement = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}
dic_alphabet_int = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 0}


def seq_str_to_int(x):
    """
    Convert list of sequence string to 2d list of integers.
    @param x: input list of sequences to convert to integers
    @return the converted list of sequences
    """
    new_x = []
    for seq in x:
        cur_seq = [dic_alphabet_int[x] for x in seq]
        new_x.append(cur_seq)
    return new_x


def read_xy_from_fasta(file_path, ids_dic, response_dic, x=[], y=[]):
    """
    Read xy from fasta file.
    @param file_path: the fasta file path to read sequences
    @param ids_dic: the dictionary to match id in input file to true id
    @param response_dic: the dictionary to match true id to response(y)
    @param x: the list of input (x) to append to
    @param y: the list of response (y) to append to
    @return the input (x) and response (y) read from fasta file(s)
    """
    print(file_path)
    content = []
    with open(file_path, "r") as f:
        content = f.read().strip().split("\n")

    for i in range(0, len(content), 2):
        cur_id = content[i][1:]
        if cur_id in ids_dic and ids_dic[cur_id] in response_dic:
            y.append(response_dic[ids_dic[cur_id]])
            x.append(content[i+1])
        else:
            print("dna id not found in response:", cur_id)

    return x, y


def read_xy_from_phy(file_path, ids_dic, response_dic, id_len=16, x=[], y=[]):
    """
    Read x and y from phy file.
    @param file_path: the fasta file path to read sequences
    @param ids_dic: the dictionary to match id in input file to true id
    @param response_dic: the dictionary to match true id to response(y)
    @param id_len: the maximum lenght of original id of sequences
    @param x: the list of input (x) to append to
    @param y: the list of response (y) to append to
    @param y_from_distribution: whether generate y from distribution of 
                                seperated set of responsess
    @param response_seperate_val: the value to seperate two class, used when
                                    generating y from distribution
    @return the input (x) and response (y) read from phy file(s)
    """
    print(file_path, len(y))
    content = []

    with open(file_path, "r") as f:
        content = f.read().strip().split("\n")

    for i in range(len(content)):
        cur_id = content[i][:id_len].strip()
        sequence = content[i][id_len:]

        if cur_id in ids_dic and ids_dic[cur_id] in response_dic:
            y.append(response_dic[ids_dic[cur_id]])
            x.append(sequence)
        else:
            print("dna id not found in response:", cur_id)

    return x, y


def read_xy(fasta_files, phy_files, ids_dic, response_dic):
    """
    Read x and y from phy and fasta file(s).
    @param fasta_files: the fasta file path to read sequences
    @param phy_files: the phy file path to read sequences
    @param ids_dic: the dictionary to match id in input file to true id
    @param response_dic: the dictionary to match true id to response(y)
    @param y_from_distribution: whether generate y from distribution of 
                                seperated set of responsess
    @param response_seperate_val: the value to seperate two class, used when
                                    generating y from distribution
    @return the input (x) and response (y) read from phy and fasta file(s)
    """
    print(fasta_files, phy_files)
    x = []
    y = []
    for fasta_file in fasta_files:
        x, y = read_xy_from_fasta(fasta_file, ids_dic, response_dic, x=x, y=y)

    for phy_file in phy_files:
        print(phy_file)
        x, y = read_xy_from_phy(
            phy_file[0], ids_dic, response_dic, id_len=phy_file[1], x=x, y=y)

    x = seq_str_to_int(x)

    return x, y


# Generater for encoding dna words.
def gen_from_arrays(features, labels, word_size=3, region_size=0, onehot=True, expand=True, alphabet='01234'):
    '''
    Create generator from arrays.
    Modified from https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/dna_nn/load.py
    '''
    words = [''.join(p) for p in product(alphabet, repeat=word_size)]
    word_to_idx = {word: i for i, word in enumerate(words)}
    word_to_idx_func = np.vectorize(
        lambda word: word_to_idx[word], otypes=[np.int8])

    def gen():
        for x, y in zip(features, labels):
            #  one hot encoding to size 10
            x = ["".join(map(str, x[i:i+word_size]))
                 for i in range(len(x) - word_size + 1)]
            idx = word_to_idx_func(list(x))
            processed_x = np.zeros((len(idx), len(word_to_idx)))
            processed_x[range(len(idx)), idx] = 1
            processed_x = np.expand_dims(processed_x, axis=-1)
            yield processed_x, y
    return gen


if __name__ == "__main__":
    # Some meta data
    print("numpy version: ", np.__version__)
    print("tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print("physical devices: ", tf.config.list_physical_devices())

    # Manual settings
    # File names
    response_file = "new_responses_original_bin.csv"
    id_file = "matchingIDs.csv"
    fasta_files = ["concatenated.fasta"]
    # each element is phy file name with max lengh of dna id
    phy_files = [["alignment_mimic.phy", 16]]
    model_path = 'test-model-h5.h5'  # model to test

    # mark_str will be print in the result csv for
    mark_str = "cnn, encoding dna, one-hot, rc, not remove same columns"
    # Mesures: carb/toby.max.rate/max.od/lag/auc
    res_choice = "delta.toby.auc"
    res_or_class = "regress"  # regress or classification
    mark_str += ", " + res_choice + ", " + res_or_class
    remove_same = False
    typeUsed = ["false", "true"]

    # delta.carb.max.od1
    res_choice = "delta.toby.max.rate"
    test_index = [185, 56, 163, 30, 79, 102, 9, 63, 36, 45, 173, 129, 6, 121,
                  175, 61, 191, 92, 84, 32, 93, 18, 113, 193, 85, 154, 60, 8, 200, 23]
    val_index = [115, 87, 118, 46, 27, 71, 53, 100, 130, 24, 148, 55, 141, 126,
                 110, 111, 132, 122, 153, 159, 166, 59, 1, 12, 72, 143, 37, 147, 106, 179, 97]

    print(mark_str)

    # Config tf for InternalError,
    # Failed to call ThenRnnForward with model config:InternalError
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    if tf.test.is_built_with_cuda():
        print("The installed TensorFlow includes GPU support.")
    else:
        print("The installed TensorFlow does not include GPU support.")

    # Retrieve raw input and response
    df = pd.read_csv(response_file)
    response_dic = dict(zip(df["lab.id"], df[res_choice]))
    ids_df = pd.read_csv(id_file)
    ids_dic = dict(zip(ids_df["OriginalID"], ids_df["LabID"]))

    # Process x and y
    x, y = read_xy(fasta_files, phy_files, ids_dic, response_dic)

    # Get test and validation set
    x = np.array(x)
    y = np.array(y)

    x_test, y_test = x[test_index], y[test_index]
    x_val, y_val = x[val_index], y[val_index]

    val_gen = gen_from_arrays(x_val, y_val)
    test_gen = gen_from_arrays(x_test, y_test)

    x_val_encode, y_val_encode = [], []
    for x, y in val_gen():
        x_val_encode.append(x)
        y_val_encode.append(y)
    x_val_encode = np.array(x_val_encode)
    y_val_encode = np.array(y_val_encode)

    x_test_encode, y_test_encode = [], []
    for x, y in test_gen():
        x_test_encode.append(x)
        y_test_encode.append(y)
    x_test_encode = np.array(x_test_encode)
    y_test_encode = np.array(y_test_encode)

    model = keras.models.load_model(model_path)

    # testing
    results = model.evaluate(x_test_encode, y_test_encode, verbose=3)
    y_score = model.predict(x_test_encode)
    y_predit = [typeUsed[i]
                for i in np.argmax(y_score, axis=1)]  # predicted class
    y_true = [typeUsed[i] for i in y_test]  # true class

    print("test: %s, %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                          ", ".join(map(str, results)),
                                          ";".join(" ".join(str(num) for num in sub)
                                                   for sub in y_score),
                                          ";".join([str(x)
                                                    for x in y_predit]),
                                          ";".join([str(x) for x in y_true])
                                          ))

    # validation
    results = model.evaluate(x_val_encode, y_val_encode, verbose=3)
    y_score = model.predict(x_val_encode)
    y_true = [typeUsed[i] for i in y_val]  # true class

    print("validation: %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                            ", ".join(map(str, results)),
                                            ";".join(" ".join(str(num) for num in sub)
                                                     for sub in y_score),
                                            ";".join([str(x) for x in y_true])
                                            ))
