import h5py
import numpy as np
import pandas as pd
import gc
import os
import uuid
import random
from itertools import product

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, Conv2D, Embedding, Dense, LSTM
from tensorflow.keras.layers import GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Input, Reshape


# 1 - A, 2 - C, 3 - G, 4 - T, 0 - missing
dic_alphabet_int = {"A": 1, "C": 2, "G": 3, "T": 4, "-": 0}
# Used for data augmentation with reverse complement
dic_complement = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}


def seq_str_to_int(x):
    """
    Convert list of sequence string to 2d list of integers
    @param x The input list of sequence strings
    """
    new_x = []
    for seq in x:
        cur_seq = [dic_alphabet_int[x] for x in seq]
        new_x.append(cur_seq)
    
    return new_x


def read_xy_from_fasta(file_path, ids_dic, response_dic, x=[], y=[]):
    """
    Read xy from fasta file and append to x and y list if dna and response matches
    @param file_path the path to fasta file
    @param ids_dic dictionary from original id to lab id
    @param response_dic dictionary from lab id to reponse (0 or 1)
    @param x the list of existing x to append to, 1d list of sequence strings
    @param y the list of existing y to append to, 1d list of 0s and 1s
    """
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
    Read xy from phy file
    @param file_path the path to phy file
    @param ids_dic dictionary from original id to lab id
    @param response_dic dictionary from lab id to reponse (0 or 1)
    @param id_len the maximum lenght of original id of sequences
    @param x the list of existing x to append to, 1d list of sequence strings
    @param y the list of existing y to append to, 1d list of 0s and 1s
    """
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
    Read fasta and phy files, format the x and y list
    @param fasta_files list of path to fasta files
    @param phy_files list of path to fasta files and maximum length of original ids
    @param ids_dic dictionary from original id to lab id
    @param response_dic dictionary from lab id to reponse (0 or 1)
    """
    x = []
    y = []
    for fasta_file in fasta_files:
        x, y = read_xy_from_fasta(fasta_file, ids_dic, response_dic, x=x, y=y)

    for phy_file in phy_files:
        x, y = read_xy_from_phy(
            phy_file[0], ids_dic, response_dic, id_len=phy_file[1], x=x, y=y)

    # Convert sequence strings to lists of integers
    x = seq_str_to_int(x)

    return x, y


def encoded_shape(x_len, word_size=3, region_size=0, onehot=True, expand=True, alphabet='01234'):
    '''
    Calculate the shape of encoding base on the sequence length
    '''
    dim_1 = x_len - word_size + 1
    dim_2 = ((len(alphabet) ** word_size) if onehot else 1) * (region_size + 1)
    if not region_size and not onehot:
        return (dim_1, 1) if expand else (dim_1,)
    
    return (dim_1, dim_2, 1) if expand else (dim_1, dim_2)


def gen_from_arrays(features, labels, word_size=3, region_size=0, onehot=True, expand=True, alphabet='01234'):
    """
    Create generater functions for training data
    """
    
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


def aug_bootstrap(x_train, y_train):
    """
    Augmentation with bootstrap. Randomly shuffle column for multiple
    times (100 for now) and append to the original matrix. Only
    augment data where y is 1
    @param x_train: x for training data
    @param y_train: y for training data
    """
    new_x_train = np.transpose(np.copy(np.array(x_train)))
    for i in range(100):
        np.random.shuffle(new_x_train)

    x_train = np.append(x_train, np.transpose(np.array(new_x_train)), axis=0)
    y_train = np.append(y_train, y_train, axis=0)

    return x_train, y_train


# Augment the train index for true class (class 1)
def aug_reverse_complement(x_train, y_train, index, features, y,  one_hot=True):
    """
    Augmentation with reverse complement
    @param x_train: x for training data
    @param y_train: y for training data
    @param index_1: index for true in original y
    @param train_ratio: ratio for training data
    @param features: original x before one-hot encoding
    """
    features_train = features[index]
    y_index = y[index]

    print("in augmentation:", y_index)

    for i in range(len(features_train)):
        feature_new = [dic_complement[x] for x in np.flip(features_train[i])]
        # One hot encoding
        if one_hot:
            feature_encode = np.zeros((len(feature_new), 5))
            feature_encode[range(len(feature_new)), feature_new] = 1
        else:
            feature_encode = np.copy(feature_new)
        # Append to trainig set
        y_train = np.append(y_train, [y_index[i]], axis=0)  # class for true
        x_train = np.append(x_train, [feature_encode], axis=0)

    return x_train, y_train


def process_xy(features, response_raw, one_hot=True, remove_same=False):
    """
    Process the input (x)  and label (y), parse x and y,
    split into training, testing, and validation datsets,
    augment the training dataset for true class with reverse complement,
    down-dample the training dataset for false class to have a balanced
    dataset, return the splitted data with their indices.
    @param feature_path: the file containing all the inputs (x)
    @param response_path: the file containing all the labels (y)
    """
    # Features, x
    features = np.array(features)
    if remove_same:
        # Remove the cols that all samples have the same site.
        features = features[:, ~np.all(features[1:] == features[:-1], axis=0)]

    # Labels, y, 0 for false, 1 for True
    y = np.array(response_raw)

    # One hot encoding
    if one_hot:
        x = np.zeros((len(features), len(features[0]), 5))
        for i in range(len(features)):
            x[i, range(len(features[i])), features[i]] = 1
    else:
        x = np.copy(features)

    # Split the indices seperately for 0/1 to ensure balance
    index_1 = np.argwhere(y == 1).flatten().tolist()
    index_0 = np.argwhere(y == 0).flatten().tolist()

    # Split x and y
    random.shuffle(index_1)
    random.shuffle(index_0)
    train_ratio = 0.7
    test_val_ratio = 0.15

    # Ratio for true:false class after augmentation
    # Currently augment each sample to two with reverse complement
    # Used only in training for a balanced dataset
    # Use the whole dataset for val and test
    # Not used for now as we have balanced data
    # ratio_1_0 = len(index_1) / len(index_0)
    # index_0_shorten = index_0[:int(len(index_0)*ratio_1_0)]
    # train_index = index_1[:int(train_ratio * len(index_1))] + \
    #     index_0_shorten[:int(train_ratio * len(index_0_shorten))]

    train_index = index_1[:int(train_ratio * len(index_1))] + \
        index_0[:int(train_ratio * len(index_0))]
    test_index = index_1[int(train_ratio * len(index_1)):int((train_ratio+test_val_ratio) * len(index_1))] + \
        index_0[int(train_ratio * len(index_0)):int((train_ratio+test_val_ratio) * len(index_0))]
    val_index = index_1[int((train_ratio+test_val_ratio) * len(index_1)):] + \
        index_0[int((train_ratio+test_val_ratio) * len(index_0)):]

    print("train:", len(train_index), train_index)
    print("val:", len(val_index), val_index)
    print("test:", len(test_index), test_index)

    x_train, y_train = x[train_index].tolist(), y[train_index].tolist()
    x_test, y_test = x[test_index], y[test_index]
    x_val, y_val = x[val_index], y[val_index]

    # Augment the train index for true class (class 1)
    x_train, y_train = aug_reverse_complement(
        x_train, y_train, train_index, features, y, one_hot=one_hot)
    # x_train, y_train = aug_bootstrap(x_train, y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test, test_index, val_index, features


def cnn_nguyen_conv1d_2_conv2d(x_shape, classes=2):
    """
    CNN model
    Modified from https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn
    """
    strides = (x_shape[0] - x_shape[1] + 1,
               1) if x_shape[0] > x_shape[1] else (1, x_shape[1] - x_shape[0] + 1)
    model = keras.Sequential([
        Conv2D(16, strides, activation='relu', input_shape=x_shape),
        MaxPooling2D(),
        Conv2D(16, 3, activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(
            classes, activation='softmax')
    ])
    return model


def deepram_recurrent_onehot(x_shape, classes=2):
    """
    Deepram recurrent model
    Modified from https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn
    """
    model = keras.Sequential([
        Input(shape=x_shape),
        Dropout(0.5),
        LSTM(16, return_sequences=True),
        LSTM(32),
        Dense(32),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(
            classes, activation='softmax')
    ])
    
    # Will cause OOM
    # model = keras.Sequential([
    #     Input(shape=x_shape),
    #     Dropout(0.5),
    #     LSTM(64, return_sequences=True),
    #     LSTM(128),
    #     Dense(128),
    #     Dropout(0.5),
    #     Dense(1, activation='sigmoid') if classes < 3 else Dense(classes, activation='softmax')
    # ])
    return model


def deepram_conv1d_recurrent_onehot(x_shape, classes=2):
    """
    Deepram convolutional 1d and recurrent model
    Modified from https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn
    """
    model = keras.Sequential([
        Input(shape=x_shape),
        Dropout(0.5),
        Conv1D(64, 3, activation='relu'),
        MaxPooling1D(),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(),
        LSTM(64, return_sequences=True),
        LSTM(128),
        Dense(128),
        Dropout(0.5),
        Dense(1, activation='sigmoid') if classes < 3 else Dense(
            classes, activation='softmax')
    ])
    return model


def tf_setup():
    """
    Set up tf for InternalError - 
    Failed to call ThenRnnForward with model config:InternalError
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    if tf.test.is_built_with_cuda():
        print("The installed version of TensorFlow includes GPU support.")
    else:
        print("The installed version of TensorFlow does not include GPU support.")


if __name__ == "__main__":
    # Some meta data
    print("numpy version: ", np.__version__)
    print("tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print("physical devices: ", tf.config.list_physical_devices())

    response_file = "new_responses_original.csv"
    id_file = "matchingIDs.csv"
    fasta_files = ["concatenated.fasta"]
    # each element is phy file name with max lengh of dna id
    phy_files = [["alignment_mimic.phy", 16]]

    mark_str = "Using deep ram, no encoding dna, use one-hot, rc"
    res_choice = "delta.toby.max.rate"
    mark_str += ", " + res_choice

    print(mark_str)

    # Random str to store the output in case of conflict
    random_str = str(uuid.uuid4())
    LOG_DIR = "./"
    csv_path = LOG_DIR + 'DeepRam-' + random_str + '-csv.csv'
    model_path = LOG_DIR + 'DeepRam-' + random_str + '-h5.h5'
    typeUsed = ["false", "true"]

    tf_setup()

    # Retrieve response
    df = pd.read_csv(response_file)
    response_dic = dict(zip(df["lab.id"], df[res_choice]))
    ids_df = pd.read_csv(id_file)
    ids_dic = dict(zip(ids_df["OriginalID"], ids_df["LabID"]))

    # Retrieve and process x and y
    x, y = read_xy(fasta_files, phy_files, ids_dic, response_dic)

    x_train, y_train, x_val, y_val, x_test, y_test, index_test, index_val, features = process_xy(
        x, y, one_hot=True, remove_same=False)

    x_shape = (len(x_train[0]), len(x_train[0][0]))

    # # For encoding dns words to avoid OOM
    # train_gen = gen_from_arrays(x_train, y_train)
    # val_gen = gen_from_arrays(x_val, y_val)
    # test_gen = gen_from_arrays(x_test, y_test)
    # prefetch = tf.data.experimental.AUTOTUNE

    # batch_size = 4
    # x_shape = encoded_shape(len(features[0]))
    # output_shapes = (x_shape, ())
    # output_types = (tf.float32, tf.float32)

    # train_ds = Dataset.from_generator(train_gen, output_types, output_shapes)
    # train_ds = train_ds.shuffle(500).batch(batch_size).prefetch(prefetch)

    # test_ds = Dataset.from_generator(test_gen, output_types, output_shapes)
    # test_ds = test_ds.batch(batch_size).prefetch(prefetch)

    # val_ds = Dataset.from_generator(val_gen, output_types, output_shapes)
    # val_ds = train_ds.shuffle(500).batch(batch_size).prefetch(prefetch)

    # x_val_encode, y_val_encode = [], []
    # for x, y in val_gen():
    #     x_val_encode.append(x)
    #     y_val_encode.append(y)
    # x_val_encode = np.array(x_val_encode)
    # y_val_encode = np.array(y_val_encode)

    # Compile model
    model = None
    keras.backend.clear_session()
    model = deepram_recurrent_onehot(x_shape)
    # model = cnn_nguyen_conv1d_2_conv2d(x_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics='accuracy')
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        keras.callbacks.CSVLogger(csv_path),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect(),
            # on_train_end=lambda logs: model.save(model_path)
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', verbose=1, patience=80)
    ]

    # # Fit generators.
    # history = model.fit(train_ds, epochs=500, validation_data=(x_val_encode, y_val_encode), callbacks=callbacks, verbose=1, batch_size=4)

    # Fit datasets.
    history = model.fit(np.array(x_train), np.array(y_train), epochs=300, validation_data=(
        np.array(x_val), np.array(y_val)), callbacks=callbacks, verbose=2, batch_size=4)

    # print index of test and val for later reproduce on performance stat
    print("Index test: ", index_test)
    print("Index val: ", index_val)
    print("y_test", y_test)
    print("y_val", y_val)
    print("Path:", csv_path, model_path)
    # Write some meta data
    with open(csv_path, "a") as f:
        f.write(",".join(map(str, index_test)))
        f.write("\n")
        f.write(",".join(map(str, y_test)))
        f.write("\n")
        f.write(",".join(map(str, index_val)))
        f.write("\n")
        f.write(",".join(map(str, y_val)))
        # Marks in csv
        f.write("\n" + mark_str + "\n")

    # Evaluate
    model = keras.models.load_model(model_path)
    results = model.evaluate(x=x_test, y=y_test)
    y_score = model.predict(x_test)
    y_predit = [typeUsed[i]
                for i in np.argmax(y_score, axis=1)]  # predicted class
    y_true = [typeUsed[i] for i in y_test]  # true class
    print("test loss, test acc:", results)

    print("%s, %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                    ", ".join(map(str, results)),
                                    ";".join(" ".join(str(num) for num in sub)
                                             for sub in y_score),
                                    ";".join([str(x)
                                              for x in y_predit]),
                                    ";".join([str(x) for x in y_true])
                                    ))
