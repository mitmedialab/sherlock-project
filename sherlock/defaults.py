import os
from collections import OrderedDict
from itertools import chain
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json

from sherlock import make_data_path

DEFAULT_FEATURE_ORDER = ["char", "word", "par", "rest"]


def default_features() -> OrderedDict:
    """Get feature identifiers per feature set, to map features to feature sets.

    Returns
    -------
    feature_cols_dict
        Dictionary with lists of feature identifiers per feature set.
    """
    feature_cols_dict = OrderedDict()
    feature_path = make_data_path("feature_column_identifiers")

    for feature_set in DEFAULT_FEATURE_ORDER:
        feature_file = os.path.join(feature_path, f"{feature_set}_col.tsv")
        feature_data = pd.read_csv(
            feature_file, sep="\t", index_col=0, header=None, squeeze=True
        )
        feature_cols_dict[feature_set] = feature_data.to_list()
    return feature_cols_dict


def default_encoder():
    encoder = LabelEncoder()
    class_file_path = make_data_path("classes_sherlock.npy")
    encoder.classes_ = np.load(class_file_path, allow_pickle=True)
    return encoder


def construct_model(
    model_path: Optional[str] = None,
    weight_path: Optional[str] = None,
    with_weights: bool = True,
):
    """Load model architecture and populate with pretrained weights.

    Parameters
    ----------
    model_path
        Location of model file
    weight_path
        Location of weight file
    with_weights
        Whether to populate the model with trained weights.

    Returns
    -------
    model
        Compiled model.
    callbacks
        Callback configuration for model retraining.
    """
    if model_path is None:
        model_path = make_data_path("sherlock_model.json")

    if weight_path is None:
        weight_path = make_data_path("sherlock_weights.h5")

    with open(model_path, "r") as model_file:
        model = model_from_json(model_file.read())

    if with_weights:
        model.load_weights(weight_path)

    learning_rate = 0.0001
    callbacks = [EarlyStopping(monitor="val_loss", patience=5)]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    return model, callbacks


DEFAULT_FEATURES_DICT: OrderedDict = default_features()
DEFAULT_FEATURES = list(chain(*[cols for cols in DEFAULT_FEATURES_DICT.values()]))
DEFAULT_ENCODER = default_encoder()
DEFAULT_MODEL, DEFAULT_CALLBACKS = construct_model()
