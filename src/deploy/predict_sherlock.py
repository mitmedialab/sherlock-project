import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


def _prepare_feature_cols():
    feature_cols_dict = {}
    for feature_set in ['char', 'word', 'par', 'rest']:
        feature_cols_dict[feature_set] = pd.read_csv('../src/features/feature_column_identifiers/{}_col.tsv'.format(feature_set),
                                                     sep='\t', index_col=0, header=None).values.flatten()
    return feature_cols_dict


def _prepare_sherlock_model(nn_id):

    # Load Sherlock architecture and weights from files
    file = open('../src/models/{}_model.json'.format(nn_id), 'r')
    sherlock_file = file.read()
    sherlock = tf.keras.models.model_from_json(sherlock_file)
    file.close()

    sherlock.load_weights('../src/models/{}_weights.h5'.format(nn_id))
    sherlock.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])
    return sherlock


def _transform_predictions_to_classes(y_pred, nn_id):

    y_pred_int = np.argmax(y_pred, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load('../src/deploy/classes_{}.npy'.format(nn_id), allow_pickle=True)
    y_pred = encoder.inverse_transform(y_pred_int)

    return y_pred


# Input: dataframe holding feature vectors,
#        nn_id indicating whether to take a retrained model or sherlock.
# Output: predicted labels.
def predict_sherlock(X, nn_id):

    sherlock = _prepare_sherlock_model(nn_id)
    feature_cols_dict = _prepare_feature_cols()
    y_pred = sherlock.predict([X[feature_cols_dict['char']].values,
                               X[feature_cols_dict['word']].values,
                               X[feature_cols_dict['par']].values,
                               X[feature_cols_dict['rest']].values])

    return _transform_predictions_to_classes(y_pred, nn_id)
