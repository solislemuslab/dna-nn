import h5py
import numpy as np
import pandas as pd
import gc
import os
import uuid
import random

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
dic_complement = {0: 0, 1: 4, 2: 3, 3: 2, 4: 1}


def read_xy(feature_path, response_path):
    """
    Read the input (x) file and label (y) file, parse x and y,
    split into training, testing, and validation datsets,
    augment the training dataset for true class with reverse complement,
    down-dample the training dataset for false class to have a balanced
    dataset, return the splitted data with their indices.
    @param feature_path: the file containing all the inputs (x)
    @param response_path: the file containing all the labels (y)
    """
    # Features, x
    feature_raw = h5py.File(feature_path, "r")
    features = np.array(feature_raw["features"])
    # Remove the cols that all samples have the same site.
    features = features[:, ~np.all(features[1:] == features[:-1], axis=0)]

    # Labels, y, 0 for false, 1 for True
    response_raw = pd.read_csv(response_path).dropna()
    # Use carb for now
    carb_tf = np.array(response_raw["carb"])
    # toby_tf = np.array(response_raw["toby"])

    # One hot encoding
    x = np.zeros((len(features), len(features[0]), 5))
    for i in range(len(features)):
        x[i, range(len(features[i])), features[i]] = 1

    # Covert true/false in y to 0/1
    y = np.array([1 if x else 0 for x in carb_tf])

    # Split the indices seperately for 0/1
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
    ratio_1_0 = len(index_1) / len(index_0) * 2
    index_0_shorten = index_0[:int(len(index_0)*ratio_1_0)]

    train_index = index_1[:int(train_ratio * len(index_1))] + \
        index_0_shorten[:int(train_ratio * len(index_0_shorten))]
    test_index = index_1[int(train_ratio * len(index_1)):int((train_ratio+test_val_ratio) * len(index_1))] + \
        index_0[int(train_ratio * len(index_0))
                    :int((train_ratio+test_val_ratio) * len(index_0))]
    val_index = index_1[int((train_ratio+test_val_ratio) * len(index_1)):] + \
        index_0[int((train_ratio+test_val_ratio) * len(index_0)):]

    print("train:", len(train_index), train_index)
    print("val:", len(val_index), val_index)
    print("test:", len(test_index), test_index)

    x_train, y_train = x[train_index].tolist(), y[train_index].tolist()
    x_test, y_test = x[test_index], y[test_index]
    x_val, y_val = x[val_index], y[val_index]

    # Augment the train index for true class (class 1) with reverse complement
    # Add augment function if there for more augmentation methods later
    train_index_1 = index_1[:int(train_ratio * len(index_1))]
    features_train_1 = features[train_index_1]

    for feature in features_train_1:
        feature_new = [dic_complement[x] for x in np.flip(feature)]
        # One hot encoding
        feature_encode = np.zeros((len(feature_new), 5))
        feature_encode[range(len(feature_new)), feature_new] = 1
        # Append to trainig set
        y_train += [1]  # class for true
        x_train += [feature_encode]

    return x_train, y_train, x_val, y_val, x_test, y_test, test_index, val_index


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


if __name__ == "__main__":
    # Some meta data
    print("numpy version: ", np.__version__)
    print("tensorflow version: ", tf.__version__)
    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))
    print("physical devices: ", tf.config.list_physical_devices())

    # Random str to store the output in case of conflict
    random_str = str(uuid.uuid4())
    LOG_DIR = "./"
    csv_path = LOG_DIR + 'DeepRam-' + random_str + '-csv.csv'
    model_path = LOG_DIR + 'DeepRam-' + random_str + '-h5.h5'
    typeUsed = ["false", "true"]

    # Config tf for InternalError,
    # Failed to call ThenRnnForward with model config:InternalError
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    if tf.test.is_built_with_cuda():
        print("The installed version of TensorFlow includes GPU support.")
    else:
        print("The installed version of TensorFlow does not include GPU support.")

    # Retrieve x and y
    x_train, y_train, x_val, y_val, x_test, y_test, index_test, index_val = read_xy(
        "features.jld", "responses.csv")

    # Compile model
    model = None
    keras.backend.clear_session()
    x_shape = (len(x_train[0]), len(x_train[0][0]))
    model = deepram_conv1d_recurrent_onehot(x_shape)
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
            monitor='val_loss', mode='min', verbose=1, patience=15)
    ]

    history = model.fit(np.array(x_train), np.array(y_train), epochs=100, validation_data=(
        np.array(x_val), np.array(y_val)), callbacks=callbacks, verbose=2, batch_size=4)

    # print index of test and val for later reproduce on performance stat
    print("Index test: ", index_test)
    print("Index val: ", index_val)
    print("y_test", y_test)
    print("y_val", y_val)
    print("Path:", csv_path, model_path)
    
    # Also store the indices and y value to csv file
    with open(csv_path, "a") as f:
        f.write(",".join(map(str, index_test)))
        f.write(",".join(map(str, y_test)))
        f.write(",".join(map(str, index_val)))
        f.write(",".join(map(str, y_val)))

    # Evaluate
    model = keras.models.load_model(model_path)
    results = model.evaluate(x=np.array(x_test), y=np.array(y_test))
    y_score = model.predict(np.array(x_test))
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
