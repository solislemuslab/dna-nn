import h5py
import numpy as np
import pandas as pd
import gc
import os

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


def read_xy(feature_path, response_path):
    """
    Read the input (x) file and label (y) file, parse x and y,
    split into training, testing, and validation datsets.
    Return the splitted data with their indices.
    @param feature_path: the file containing all the inputs (x)
    @param response_path: the file containing all the labels (y)
    """
    # Features, x
    feature_raw = h5py.File(feature_path, "r")
    features = np.array(feature_raw["features"])

    # Labels, y, 0 for false, 1 for True
    response_raw = pd.read_csv(response_path).dropna()
    # Use carb for now
    carb_tf = np.array(response_raw["carb"])
    # toby_tf = np.array(response_raw["toby"])

    # One hot encoding
    x = np.zeros((len(features), len(features[0]), 5))
    for i in range(len(features)):
        x[i, range(len(features[i])), features[i]] = 1

    # Covert true/false in y to 0 and 1
    y = [1 if x else 0 for x in carb_tf]

    x_train, x_testVal, y_train, y_testVal, index_train, index_testVal = train_test_split(
        x, y, range(len(x)), test_size=0.3)
    x_test, x_val, y_test, y_val, index_test, index_val = train_test_split(
        x_testVal, y_testVal, index_testVal, test_size=0.5)

    return x_train, y_train, x_val, y_val, x_test, y_test, index_test, index_val


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
    print("tensorflow version: ", tf.__version__)

    LOG_DIR = "./"
    csv_path = LOG_DIR + 'DeepRam-dynamics.csv'
    model_path = LOG_DIR + 'DeepRam.h5'
    # Labels, y, 0 for false, 1 for True
    typeUsed = ["false", "true"]

    # Config tf to avoid InternalError
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # Retrieve x and y
    x_train, y_train, x_val, y_val, x_test, y_test, index_test, index_val = read_xy(
        "features.jld", "responses.csv")

    # Compile model
    # Modified from https://github.com/solislemuslab/dna-nn-theory/tree/master/cnn
    model = None
    keras.backend.clear_session()
    x_shape = (len(x_train[0]), len(x_train[0][0]))
    model = deepram_recurrent_onehot(x_shape)
    # model = deepram_conv1d_recurrent_onehot(x_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics='accuracy')
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
        keras.callbacks.CSVLogger(csv_path),
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect(),
            # on_train_end=lambda logs: model.save(model_path)
        )
    ]

    history = model.fit(np.array(x_train), np.array(y_train), epochs=50, validation_data=(
        np.array(x_val), np.array(y_val)), callbacks=callbacks, verbose=1, batch_size=4)

    # print index of test and val for later reproduce on performance stat
    print("Index test: ", index_test)
    print("Index val: ", index_val)

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
