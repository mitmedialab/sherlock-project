import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelEncoder


# Input: feature_vectors numpy ndarray as returned by build_features,
#        nn_id indicating whether to take a retrained model or sherlock
# Output: predicted targets
def predict_sherlock(feature_vectors, nn_id):

    # Load Sherlock model
    # Load json and create model
    file = open('../src/models/{}_model.json'.format(nn_id), 'r')
    sherlock_file = file.read()
    sherlock = tf.keras.models.model_from_json(sherlock_file)
    file.close()

    # Load weights into new model
    sherlock.load_weights('../src/models/{}_weights.h5'.format(nn_id))

    # Compile model
    sherlock.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['categorical_accuracy'])

    y_pred = sherlock.predict(feature_vectors)
    y_pred_int = np.argmax(y_pred, axis=1)

    encoder = LabelEncoder()
    encoder.classes_ = np.load('../src/deploy/classes_{}.npy'.format(nn_id), allow_pickle=True)
    y_pred = encoder.inverse_transform(y_pred_int)

    return y_pred
