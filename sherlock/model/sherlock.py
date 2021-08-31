import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    concatenate,
)
import tensorflow as tf


def _build_char_submodel(char_shape, lamb, do):
    n_weights = 300

    char_model_input = Input(shape=(char_shape,))
    char_model1 = BatchNormalization(axis=1)(char_model_input)
    char_model2 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(char_model1)
    char_model3 = Dropout(do)(char_model2)
    char_model4 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(char_model3)

    return char_model_input, char_model4


def _build_word_submodel(word_shape, lamb, do):
    n_weights = 200

    word_model_input = Input(shape=(word_shape,))
    word_model1 = BatchNormalization(axis=1)(word_model_input)
    word_model2 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(word_model1)
    word_model3 = Dropout(do)(word_model2)
    word_model4 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(word_model3)

    return word_model_input, word_model4


def _build_rest_submodel(rest_shape):

    # Build submodel for remaining features
    rest_model_input = Input(shape=(rest_shape,))
    rest_model1 = BatchNormalization(axis=1)(rest_model_input)

    return rest_model_input, rest_model1


def _add_main_layers(merged_model1, do, lamb, num_classes):
    n_weights = 500

    merged_model2 = BatchNormalization(axis=1)(merged_model1)
    merged_model3 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(merged_model2)
    merged_model4 = Dropout(do)(merged_model3)
    merged_model5 = Dense(
        n_weights,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(merged_model4)
    merged_model_output = Dense(
        num_classes,
        activation=tf.nn.softmax,
        kernel_regularizer=tf.keras.regularizers.l2(lamb),
    )(merged_model5)

    return merged_model_output


def build_sherlock(
    X_train_char,
    X_train_word,
    X_train_rest,
    X_val_char,
    X_val_word,
    X_val_rest,
    y_train,
    y_val,
    nn_id,
):

    num_classes = len(set(y_train))
    lamb = 0.0001
    lr = 0.0001
    do = 0.35

    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save("../sherlock/deploy/classes_{}.npy".format(nn_id), encoder.classes_)

    y_train_int = encoder.transform(y_train)
    y_val_int = encoder.transform(y_val)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    callbacks = [EarlyStopping(monitor="categorical_accuracy", patience=5)]

    char_model_input, char_model = _build_char_submodel(X_train_char.shape[1], lamb, do)
    word_model_input, word_model = _build_word_submodel(X_train_word.shape[1], lamb, do)
    #     par_model_input, par_model = _build_par_submodel(X_train_par.shape[1], lamb, do)
    rest_model_input, rest_model = _build_rest_submodel(X_train_rest.shape[1])

    # Merge submodels and build main network
    merged_model1 = concatenate(
        [
            char_model,
            word_model,
            #                                  par_model,
            rest_model,
        ]
    )

    merged_model_output = _add_main_layers(merged_model1, lamb, do, num_classes)

    multi_input_model = Model(
        [
            char_model_input,
            word_model_input,
            #                                par_model_input,
            rest_model_input,
        ],
        merged_model_output,
    )

    multi_input_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    multi_input_model.fit(
        [
            X_train_char.values,
            X_train_word.values,
            #                                                X_train_par.values,
            X_train_rest.values,
        ],
        y_train_cat,
        validation_data=(
            [
                X_val_char.values,
                X_val_word.values,
                #                                                                 X_val_par.values,
                X_val_rest.values,
            ],
            y_val_cat,
        ),
        callbacks=callbacks,
        epochs=100,
        batch_size=256,
    )

    multi_input_model.save("../models/sherlock_model_{}.json".format(nn_id))
