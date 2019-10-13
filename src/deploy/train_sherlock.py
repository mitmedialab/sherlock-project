import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping

SEED = 13


def _prepare_feature_cols():
    feature_cols_dict = {}
    for feature_set in ['char', 'word', 'par', 'rest']:
        feature_cols_dict[feature_set] = pd.read_csv('../src/features/feature_column_identifiers/{}_col.tsv'.format(feature_set),
                                                     sep='\t', index_col=0, header=None).values.flatten()
    return feature_cols_dict


def _prepare_label_vectors(y_train, y_val, nn_id):

    # Prepare categorical label encoder
    encoder = LabelEncoder()
    encoder.fit(y_train)
    np.save('../src/deploy/classes_{}.npy'.format(nn_id), encoder.classes_)

    # Convert train labels
    y_train_int = encoder.transform(y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    # Convert val labels
    y_val_int = encoder.transform(y_val)
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def _prepare_sherlock_model(nn_id):

    lr = 0.0001
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]

    file = open('../src/models/{}_model.json'.format(nn_id), 'r')
    sherlock = model_from_json(file.read())
    file.close()

    sherlock.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

    return sherlock, callbacks


def _save_retrained_sherlock_model(sherlock, nn_id):

    model_json = sherlock.to_json()
    with open('../src/models/{}_model.json'.format(nn_id), 'w') as json:
        json.write(model_json)

    sherlock.save_weights('../src/models/{}_weights.h5'.format(nn_id))


# Input: X_train and X_val dataframes as returned by the build_features method,
#        y_train and y_val arrays of string labels,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: Stored retrained model
def train_sherlock(X_train, y_train, X_val, y_val, nn_id):

    y_train_cat, y_val_cat = _prepare_label_vectors(y_train, y_val, nn_id)
    feature_cols = _prepare_feature_cols()
    sherlock, callbacks = _prepare_sherlock_model(nn_id)

    print('Successfully loaded and compiled model, now fitting model on data.')

    sherlock.fit([X_train[feature_cols['char']].values,
                  X_train[feature_cols['word']].values,
                  X_train[feature_cols['par']].values,
                  X_train[feature_cols['rest']].values],
                 y_train_cat,
                 validation_data=([X_val[feature_cols['char']].values,
                                   X_val[feature_cols['word']].values,
                                   X_val[feature_cols['par']].values,
                                   X_val[feature_cols['rest']].values],
                                  y_val_cat),
                 callbacks=callbacks, epochs=100, batch_size=256)

    _save_retrained_sherlock_model(sherlock, nn_id)

    print('Retrained Sherlock.')
