import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
from tensorflow.keras import models


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


def construct_sherlock_model(nn_id: str, with_weights: bool):
    """Load model architecture and populate with pretrained weights.

    Parameters
    ----------
    nn_id
        Identifier for retrained model.
    with_weights
        Whether to populate the model with trained weights.

    Returns
    -------
    sherlock_model
        Compiled sherlock model.
    callbacks
        Callback configuration for model retraining.
    """

    lr = 0.0001
    callbacks = [EarlyStopping(monitor="categorical_accuracy", patience=5)]

    sherlock_model = models.load_model(f"../models/sherlock_model_{nn_id}.json")

    sherlock_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return sherlock_model, callbacks
