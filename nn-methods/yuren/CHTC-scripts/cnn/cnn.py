"""
The file trains the cnn model. It reads data from fasta and phy file(s), 
then processed, augments, and seperates (training/validation/testing sets) the
data, and finally initializes, trains, and tests the model. The training steps
might fail on some servers due to OOM, but work on some other servers. This file
will generate the output (trained) cnn model and csv log file besides default
CHTC output and error file(s).
"""

import h5py
import numpy as np
import pandas as pd
import gc
import os
import uuid
import random
from itertools import product

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Conv2D, Dense, LSTM
from tensorflow.keras.layers import MaxPooling1D, MaxPooling2D
from tensorflow.keras.layers import Dropout, Flatten, Input


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
    print("In read_xy_from_fasta, reading faste file: ", file_path)
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


def read_xy_from_phy(file_path, ids_dic, response_dic, id_len=16, x=[], y=[],
                     y_from_distribution=False, response_seperate_val=0.5):
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
    print("In read_xy_from_phy, file path, length:", file_path, len(y))
    content = []

    with open(file_path, "r") as f:
        content = f.read().strip().split("\n")

    # find the std for two seperated group
    if y_from_distribution:
        responses = np.array(list(response_dic.values()))
        response_larger = responses[np.where(
            responses >= response_seperate_val)[0]]
        response_smaller = responses[np.where(
            responses < response_seperate_val)[0]]
        response_larger_std = np.std(response_larger)
        response_larger_mean = np.mean(response_larger)
        response_smaller_std = np.std(response_smaller)
        response_smaller_mean = np.mean(response_smaller)

    for i in range(len(content)):
        cur_id = content[i][:id_len].strip()
        sequence = content[i][id_len:]

        if cur_id in ids_dic and ids_dic[cur_id] in response_dic:
            x.append(sequence)

            cur_y = response_dic[ids_dic[cur_id]]
            if not y_from_distribution:
                y.append(cur_y)
            else:
                # generate y from normal distribution
                if cur_y < response_seperate_val:
                    y.append(np.random.normal(
                        response_smaller_mean, response_smaller_std))
                else:
                    y.append(np.random.normal(
                        response_larger_mean, response_larger_std))
        else:
            print("dna id not found in response:", cur_id)

    return x, y


def read_xy(fasta_files, phy_files, ids_dic, response_dic,
            y_from_distribution=False, response_seperate_val=0.5):
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
        x, y = read_xy_from_phy(phy_file[0], ids_dic, response_dic,
                                id_len=phy_file[1], x=x, y=y,
                                y_from_distribution=y_from_distribution,
                                response_seperate_val=response_seperate_val)

    x = seq_str_to_int(x)

    return x, y


def encoded_shape(x_len, word_size=3, region_size=0, onehot=True, expand=True,
                  alphabet='01234'):
    """
    Calculate the shape of encoding base on the sequence length.
    Copied from https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/dna_nn/load.py
    """
    dim_1 = x_len - word_size + 1
    dim_2 = ((len(alphabet) ** word_size) if onehot else 1) * (region_size + 1)
    if not region_size and not onehot:
        return (dim_1, 1) if expand else (dim_1,)
    return (dim_1, dim_2, 1) if expand else (dim_1, dim_2)


def gen_from_arrays(features, labels, word_size=3, region_size=0, onehot=True,
                    expand=True, alphabet='01234'):
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


def aug_bootstrap(x, y, shuffle_time=100):
    """
    Augmentation with bootstrap. Randomly shuffle column for multiple
    times and append to the original matrix. Not working so is not used now.
    @param x: input (x) to augment
    @param y: response (y) to augment
    @param shuffle time: number of shuffling data
    @return the augmented data
    """
    # Randomly shuffle column for multiple times (100 for now)
    # and append to the original matrix
    new_x = np.transpose(np.copy(np.array(x)))
    for i in range(shuffle_time):
        np.random.shuffle(new_x)

    x = np.append(x, np.transpose(np.array(new_x)), axis=0)
    y = np.append(y, y, axis=0)

    return x, y


def aug_reverse_complement(x_train, y_train, index, features, y,  one_hot=True):
    """
    Augmentation with reverse complement
    @param x_train: x for training data to augment
    @param y_train: y for training data to augment
    @param index: indices to augment
    @param features: original x before processing
    @param y: original y before processing
    @param one_hot: whether to do one-hot encoding on original data
    @return the augmented data
    """
    features_train = features[index]
    y_index = y[index]

    print("in augmentation:", y_index)

    for i in range(len(features_train)):
        # Reverse complement
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


def process_xy(features, response_raw, one_hot=True, remove_same=False,
               split_one_zero=False, train_ratio=0.7, test_val_ratio=0.15):
    """
    Process the input (x)  and label (y), parse x and y,
    split into training, testing, and validation datsets,
    augment the training dataset for true class with reverse complement,
    down-sample the training dataset for false class to have a balanced
    dataset (if needed)
    @param feature_path: the file containing all the inputs (x)
    @param response_path: the file containing all the labels (y)
    @param split_one_zero: whether split x and y for y is one or zero separately
    @param train_ratio: the ratio for training set
    @param test_val_ratio: the ratio for testing and validation set
    @return the processed and splitted data with their indices.
    """
    # Features, x
    features = np.array(features)

    # Remove the cols that all samples have the same site.
    if remove_same:
        features = features[:, ~np.all(features[1:] == features[:-1], axis=0)]

    y = np.array(response_raw)

    # One hot encoding
    if one_hot:
        x = np.zeros((len(features), len(features[0]), 5))
        for i in range(len(features)):
            x[i, range(len(features[i])), features[i]] = 1
    else:
        x = np.copy(features)

    # split the index between 1 and 0 before randomly splitting to ensure that
    # the data in two classes are balanced
    if split_one_zero:
        # Split the indices seperately for 0/1 to ensure balance
        index_1 = np.argwhere(y == 1).flatten().tolist()
        index_0 = np.argwhere(y == 0).flatten().tolist()

        # Split x and y
        random.shuffle(index_1)
        random.shuffle(index_0)

        train_index = index_1[:int(train_ratio * len(index_1))] + \
            index_0[:int(train_ratio * len(index_0))]
        test_index = index_1[int(train_ratio * len(index_1)):
                             int((train_ratio+test_val_ratio) * len(index_1))] + \
            index_0[int(train_ratio * len(index_0)):
                    int((train_ratio+test_val_ratio) * len(index_0))]

        val_index = index_1[int((train_ratio+test_val_ratio) * len(index_1)):] + \
            index_0[int((train_ratio+test_val_ratio) * len(index_0)):]
    else:
        indices = list(range(len(features)))
        random.shuffle(indices)

        train_index = indices[:int(train_ratio * len(indices))]
        test_index = indices[int(train_ratio * len(indices)):
                             int((train_ratio+test_val_ratio) * len(indices))]
        val_index = indices[int((train_ratio+test_val_ratio) * len(indices)):]

    print("train len, train index:", len(train_index), train_index)
    print("val len, val index:", len(val_index), val_index)
    print("test len, test index:", len(test_index), test_index)

    x_train, y_train = x[train_index].tolist(), y[train_index].tolist()
    x_test, y_test = x[test_index], y[test_index]
    x_val, y_val = x[val_index], y[val_index]

    # Augment the train index
    x_train, y_train = aug_reverse_complement(
        x_train, y_train, train_index, features, y, one_hot=one_hot)
    # x_train, y_train = aug_bootstrap(x_train, y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test, test_index, \
        val_index, features


def cnn_nguyen_conv1d_2_conv2d(x_shape, res_or_class="regress", classes=2):
    """
    CNN model, change final layter to linear dense layer if do regression.
    Modified from https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn
    """
    strides = (x_shape[0] - x_shape[1] + 1,
               1) if x_shape[0] > x_shape[1] else (1, x_shape[1] - x_shape[0] + 1)
    if res_or_class == "regress":
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
            Dense(1, activation="linear"),
        ])
    else:
        # binary classification
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
            Dense(1, activation='sigmoid') if classes < 3 else
            Dense(classes, activation='softmax')
        ])

    return model


if __name__ == "__main__":
    # Manual settings
    # mark_str will be print in the result csv for
    mark_str = "cnn, encoding dna, use one-hot, rc, not remove same columns"
    # Mesures: carb/toby.max.rate/max.od/lag/auc
    res_choice = "delta.toby.max.rate"
    res_or_class = "regress"  # regress or classify
    mark_str += ", " + res_choice + ", " + res_or_class
    print(mark_str)

    remove_same = False
    y_from_distribution = True
    # the value to seperate more or less susceptible
    response_seperate_val = 0.091
    batch_size = 32

    # File names
    # new_responses_original.csv for original float results and
    # new_responses_original_bin.csv for processed binary results.
    response_file = "new_responses_original.csv" \
                    if res_or_class == "regress" \
                    else "new_responses_original_bin.csv"
    id_file = "matchingIDs.csv"
    fasta_files = ["concatenated.fasta"]
    # each element is phy file name with max lengh of dna id
    phy_files = [["alignment_mimic.phy", 16]]

    # Random str to store the output in case of conflict
    random_str = str(uuid.uuid4())
    LOG_DIR = "./"
    csv_path = LOG_DIR + res_or_class + '-' + random_str + '-csv.csv'
    model_path = LOG_DIR + res_or_class + '-' + random_str + '-h5.h5'
    typeUsed = ["false", "true"]

    # Some meta data about env
    print("numpy version: ", np.__version__)
    print("tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print("physical devices: ", tf.config.list_physical_devices())

    # Config tf for InternalError -  Failed to call ThenRnnForward with model
    # config:InternalError
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    if tf.test.is_built_with_cuda():
        print("The installed TensorFlow includes GPU support.")
    else:
        print("The installed TensorFlow does not include GPU support.")

    # Retrieve raw data
    df = pd.read_csv(response_file)
    response_dic = dict(zip(df["lab.id"], df[res_choice]))
    ids_df = pd.read_csv(id_file)
    ids_dic = dict(zip(ids_df["OriginalID"], ids_df["LabID"]))

    # Retrieve and process x and y
    x, y = read_xy(fasta_files, phy_files, ids_dic, response_dic,
                   y_from_distribution=y_from_distribution,
                   response_seperate_val=response_seperate_val)

    split_one_zero = res_or_class == "classification"
    x_train, y_train, x_val, y_val, x_test, y_test, index_test, index_val, \
        features = process_xy(x, y, one_hot = False, remove_same = remove_same,
                              split_one_zero = split_one_zero)

    # For encoding dns words, CNN
    train_gen = gen_from_arrays(x_train, y_train)
    val_gen = gen_from_arrays(x_val, y_val)
    test_gen = gen_from_arrays(x_test, y_test)
    prefetch = tf.data.experimental.AUTOTUNE

    x_shape = encoded_shape(len(features[0]))
    print(x_shape)
    output_shapes = (x_shape, ())
    output_types = (tf.float32, tf.float32)

    train_ds = Dataset.from_generator(train_gen, output_types, output_shapes)
    # Takes long time for shuffling.
    train_ds = train_ds.shuffle(50).batch(batch_size).prefetch(prefetch)

    x_val_encode, y_val_encode = [], []
    for x, y in val_gen():
        x_val_encode.append(x)
        y_val_encode.append(y)
    x_val_encode = np.array(x_val_encode)
    y_val_encode = np.array(y_val_encode)

    # Compile model
    model = None
    keras.backend.clear_session()
    # model = deepram_recurrent_onehot(x_shape)
    model = cnn_nguyen_conv1d_2_conv2d(x_shape, res_or_class=res_or_class)
    if res_or_class == "regress":
        model.compile(optimizer='adam', loss='mean_absolute_percentage_error')
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics='accuracy')
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        keras.callbacks.CSVLogger(csv_path),
        keras.callbacks.LambdaCallback(
            on_epoch_end= lambda epoch, logs: gc.collect(),
            # on_train_end=lambda logs: model.save(model_path)
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', verbose=2, patience=70)
    ]

    # Fit generators.
    history = model.fit(train_ds, epochs=300, validation_data=(
        x_val_encode, y_val_encode), callbacks=callbacks, verbose=1,
        batch_size = batch_size)

    # # Fit datasets, not used due to OOM
    # history = model.fit(np.array(x_train), np.array(y_train), epochs=500,
    #             validation_data=(np.array(x_val), np.array(y_val)),
    #             callbacks=callbacks, verbose=2, batch_size=4)

    # Print index of test and val for later reproduce on performance stat
    print("Index test: ", index_test)
    print("Index val: ", index_val)
    print("y_test:", y_test)
    print("y_val:", y_val)
    print("Path:", csv_path, model_path)
    # Append those meta data to the csv log file
    with open(csv_path, "a") as f:
        f.write("index_text: ")
        f.write(",".join(map(str, index_test)))
        f.write("\ny_text: ")
        f.write(",".join(map(str, y_test)))
        f.write("\nindex_val: ")
        f.write(",".join(map(str, index_val)))
        f.write("\ny_val: ")
        f.write(",".join(map(str, y_val)))
        f.write("\n" + mark_str)

    # Load and evaluate the best model with testing and validation data.
    # Testing
    x_test_encode, y_test_encode = [], []
    for x, y in test_gen():
        x_test_encode.append(x)
        y_test_encode.append(y)
    x_test_encode = np.array(x_test_encode)
    y_test_encode = np.array(y_test_encode)

    model = keras.models.load_model(model_path)
    results = model.evaluate(x_test_encode, y_test_encode, verbose=3)

    print("test loss, test acc:", results)
    with open(csv_path, "a") as f:
        f.write("test loss, test acc:" +
                str(results[0]) + "," + str(results[1]) + '\n')

    y_score = model.predict(x_test_encode)
    y_true = [typeUsed[i] for i in y_test]  # true class

    print("test: %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                      ", ".join(map(str, results)),
                                      ";".join(" ".join(str(num)
                                                        for num in sub)
                                               for sub in y_score),
                                      ";".join([str(x) for x in y_true])
                                      ))

    with open(csv_path, "a") as f:
        f.write("test: %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                            ", ".join(map(str, results)),
                                            ";".join(" ".join(str(num)
                                                              for num in sub)
                                                     for sub in y_score),
                                            ";".join([str(x)
                                                      for x in y_true])
                                            ))

    # validation
    results = model.evaluate(x_val_encode, y_val_encode, verbose=3)
    y_score = model.predict(x_val_encode)
    y_true = [typeUsed[i] for i in y_val]  # true class

    print("test: %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                      ", ".join(map(str, results)),
                                      ";".join(" ".join(str(num)
                                                        for num in sub)
                                               for sub in y_score),
                                      ";".join([str(x) for x in y_true])
                                      ))

    with open(csv_path, "a") as f:
        f.write("test: %s, %s, %s, %s\n" % ("; ".join(map(str, typeUsed)),
                                            ", ".join(map(str, results)),
                                            ";".join(" ".join(str(num)
                                                              for num in sub)
                                                     for sub in y_score),
                                            ";".join([str(x)
                                                      for x in y_true])
                                            ))
