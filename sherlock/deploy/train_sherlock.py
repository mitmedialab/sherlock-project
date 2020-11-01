import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sherlock.deploy import model_helpers

SEED = 13


def _get_categorical_label_encodings(y_train, y_val, nn_id) -> (list, list):
    """Encode semantic type string labels as categoricals.
    
    Parameters
    ----------
    y_train
        Train labels.
    y_val
        Validation labels.
    nn_id
        Identifier of retrained model.
        
    Returns
    -------
    y_train_cat
        Categorical encodings of train labels.  
    y_val_cat
        Categorical encodings of validation labels.
    """

    # Prepare categorical label encoder
    encoder = LabelEncoder()
    encoder.fit(y_train)

    np.save(f"../sherlock/deploy/classes_{nn_id}.npy", encoder.classes_)

    # Convert train labels
    y_train_int = encoder.transform(y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    # Convert val labels
    y_val_int = encoder.transform(y_val)
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def _save_retrained_sherlock_model(sherlock_model, nn_id: str):
    """Save weights of retrained sherlock model.
    
    Parameters
    ----------
    sherlock_model
        Retrained sherlock model.
    nn_id
        Identifier for retrained model.
    """

    model_json = sherlock_model.to_json()
    with open(f"../models/{nn_id}_model.json", "w") as json:
        json.write(model_json)

    sherlock_model.save_weights(f"../models/{nn_id}_weights.h5")


def train_sherlock(
    X_train: pd.DataFrame,
    y_train: list,
    X_val: pd.DataFrame,
    y_val: list,
    nn_id: str,
):
    """Train weights of sherlock model from existing NN architecture.
    
    Parameters
    ----------
    X_train
        Train data to train model on.
    y_train
        Train labels to train model with.
    X_val
        Validation data to steer early stopping.
    y_val
        Validation labels.
    nn_id
        Identifier for retrained model.
    """
    
    if nn_id == "sherlock":
        raise ValueError(
            """nn_id cannot be equal to 'sherlock' 
            to avoid overwriting pretrained model.
            """
        )
    
    feature_cols = model_helpers.categorize_features()
    y_train_cat, y_val_cat = _get_categorical_label_encodings(y_train, y_val, nn_id)
    sherlock_model, callbacks = model_helpers.construct_sherlock_model(nn_id, False)

    print("Successfully loaded and compiled model, now fitting model on data.")

    sherlock_model.fit(
        [
            X_train[feature_cols['char']].values,
            X_train[feature_cols['word']].values,
            X_train[feature_cols['par']].values,
            X_train[feature_cols['rest']].values,
        ],
        y_train_cat,
        validation_data=(
            [
                X_val[feature_cols['char']].values,
                X_val[feature_cols['word']].values,
                X_val[feature_cols['par']].values,
                X_val[feature_cols['rest']].values,
            ],
            y_val_cat
        ),
        callbacks=callbacks, epochs=100, batch_size=256
    )

    _save_retrained_sherlock_model(sherlock_model, nn_id)

    print('Retrained Sherlock.')
