from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats

NUM_EMBEDDINGS = 50


def make_default_response():
    default_response = OrderedDict()
    default_response.setdefault("word_embedding_feature", 0)
    for i in range(NUM_EMBEDDINGS):
        default_response.setdefault("word_embedding_avg_{}".format(i), np.nan)
    for i in range(NUM_EMBEDDINGS):
        default_response.setdefault("word_embedding_std_{}".format(i), np.nan)
    for i in range(NUM_EMBEDDINGS):
        default_response.setdefault("word_embedding_med_{}".format(i), np.nan)
    for i in range(NUM_EMBEDDINGS):
        default_response.setdefault("word_embedding_mode_{}".format(i), np.nan)
    return default_response


def embedding_getter(word_to_embedding):
    def inner(value: str):
        if value in word_to_embedding:
            embedding = word_to_embedding.get(value)
        else:
            embeddings = [
                word_to_embedding.get(v)
                for v in value.split(" ")
                if v in word_to_embedding
            ]
            embedding = np.mean(embeddings, axis=0)
        return embedding

    return inner


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding word embedding features
def extract_word_embeddings_features(
    values: pd.Series, word_to_embedding: dict
) -> OrderedDict:
    f = make_default_response()

    get_embedding = embedding_getter(word_to_embedding)
    embeddings = [get_embedding(val) for val in values.str.lower()]

    if len(embeddings) == 0:
        return f

    f["word_embedding_feature"] = 1
    mean_embeddings = np.nanmean(embeddings, axis=0)
    med_embeddings = np.nanmedian(embeddings, axis=0)
    std_embeddings = np.nanstd(embeddings, axis=0)
    mode_embeddings = stats.mode(embeddings, axis=0, nan_policy="omit")[0].flatten()

    for i in range(NUM_EMBEDDINGS):
        f[f"word_embedding_avg_{i}"] = mean_embeddings[i]
        f[f"word_embedding_std_{i}"] = std_embeddings[i]
        f[f"word_embedding_med_{i}"] = med_embeddings[i]
        f[f"word_embedding_mode_{i}"] = mode_embeddings[i]

    return f
