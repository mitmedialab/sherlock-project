import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder


def categorize_features() -> dict:
    """Get feature identifiers per feature set, to map features to feature sets.

    Returns
    -------
    feature_cols_dict
        Dictionary with lists of feature identifiers per feature set.
    """
    feature_cols_dict = {}
    for feature_set in ["char", "word", "par", "rest"]:
        feature_cols_dict[feature_set] = pd.read_csv(
            f"../sherlock/features/feature_column_identifiers/{feature_set}_col.tsv",
            sep="\t",
            index_col=0,
            header=None,
            squeeze=True,
        ).to_list()
    return feature_cols_dict


def _get_categorical_label_encodings(y_train, y_val, model_id: str) -> (list, list):
    """Encode semantic type string labels as categoricals.

    Parameters
    ----------
    y_train
        Train labels.
    y_val
        Validation labels.
    model_id
        Identifier of retrained model weighs.

    Returns
    -------
    y_train_cat
        Categorical encodings of train labels.
    y_val_cat
        Categorical encodings of validation labels.
    """
    if model_id == "sherlock":
        raise ValueError(
            "`model_id` cannot be `sherlock` to avoid overwriting original class encodings."
        )
    # Prepare categorical label encoder
    encoder = LabelEncoder()
    encoder.fit(y_train)

    np.save(f"../model_files/classes_{model_id}.npy", encoder.classes_)

    # Convert train labels
    y_train_int = encoder.transform(y_train)
    y_train_cat = tf.keras.utils.to_categorical(y_train_int)

    # Convert val labels
    y_val_int = encoder.transform(y_val)
    y_val_cat = tf.keras.utils.to_categorical(y_val_int)

    return y_train_cat, y_val_cat


def _proba_to_classes(y_pred, model_id: str = "sherlock") -> np.array:
    """Get predicted semantic types from prediction vectors.

    Parameters
    ----------
    y_pred
        Nested vector with for each sample a vector of likelihoods per semantic type.
    model_id
        Identifier of model to use.

    Returns
    -------
    y_pred
        Predicted semantic labels.
    """
    y_pred_int = np.argmax(y_pred, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        f"../model_files/classes_{model_id}.npy", allow_pickle=True
    )

    y_pred = encoder.inverse_transform(y_pred_int)

    return y_pred
