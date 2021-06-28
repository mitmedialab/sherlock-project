import os
import random
from ast import literal_eval
from collections import OrderedDict
from functools import cache
from typing import Optional, Union

import numpy as np
import pandas as pd
from google_drive_downloader import GoogleDriveDownloader as gd

from sherlock import make_data_path
from sherlock.features.bag_of_characters import \
    extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.paragraph_vectors import \
    infer_paragraph_embeddings_features
from sherlock.features.word_embeddings import extract_word_embeddings_features


def prepare_feature_extraction():
    """Download embedding files from Google Drive if they do not exist yet."""
    word_embedding_file = make_data_path("glove.6B.50d.txt")
    paragraph_vector_file = make_data_path(
        "par_vec_trained_400.pkl.docvecs.vectors_docs.npy"
    )

    if not os.path.exists(word_embedding_file):
        print("Downloading GloVe word embedding vectors.")
        file_name = word_embedding_file
        gd.download_file_from_google_drive(
            file_id="1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk",
            dest_path=file_name,
            unzip=False,
            showsize=True,
        )

        print("GloVe word embedding vectors were downloaded.")

    if not os.path.exists(paragraph_vector_file):
        print("Downloading pretrained paragraph vectors.")
        file_name = paragraph_vector_file
        gd.download_file_from_google_drive(
            file_id="1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu",
            dest_path=file_name,
            unzip=False,
            showsize=True,
        )

        print("Trained paragraph vector model was downloaded.")


@cache
def prepare_word_embeddings():
    word_vectors_f = open(make_data_path("glove.6B.50d.txt"), encoding="utf-8")
    word_to_embedding = {}

    for w in word_vectors_f:
        term, vector = w.strip().split(" ", 1)
        vector = np.array(vector.split(" "), dtype=float)
        word_to_embedding[term] = vector

    return word_to_embedding


def extract_feature(data: pd.Series, n_samples=None):
    word_to_embedding = prepare_word_embeddings()

    if n_samples:
        random.seed(13)
        n_samples = min(data.shape[0], n_samples)
        data = pd.Series(random.choices(data, k=n_samples))

    not_na_idx = data.notna() if data.hasnans else np.array([True] * data.shape[0])
    data[not_na_idx] = data[not_na_idx].astype(str)

    f = (
        extract_bag_of_characters_features(data[not_na_idx])
        | extract_word_embeddings_features(data[not_na_idx], word_to_embedding)
        | extract_bag_of_words_features(data)
        | infer_paragraph_embeddings_features(data[not_na_idx])
    )

    return f


def extract_features(
    data: Union[pd.DataFrame, pd.Series], n_samples: Optional[int] = None
) -> pd.DataFrame:
    """Extract features from raw data.

    Parameters
    ----------
    data
        A pandas DataFrame or Series with each row a list of string values.
    n_samples
        An optional integer indicating the number of samples to use for feature extraction

    Returns
    -------
    DataFrame with featurized column samples.
    """
    prepare_feature_extraction()

    if isinstance(data, pd.Series):
        return pd.DataFrame([extract_feature(data, n_samples)])

    return pd.DataFrame([extract_feature(data[col], n_samples) for col in data.columns])
