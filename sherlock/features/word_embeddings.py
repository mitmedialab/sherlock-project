import numpy as np

from scipy import stats
from collections import OrderedDict
from datetime import datetime
import pandas as pd

word_to_embedding = {}


def initialise_word_embeddings():
    start = datetime.now()

    global word_to_embedding

    print("Initialising word embeddings")
    word_vectors_f = open('../sherlock/features/glove.6B.50d.txt', encoding='utf-8')

    for w in word_vectors_f:
        term, vector = w.strip().split(' ', 1)
        vector = np.array(vector.split(' '), dtype=float)
        word_to_embedding[term] = vector

    end = datetime.now()
    x = end - start

    word_vectors_f.close()

    print(f'Initialise Word Embeddings process took {x} seconds.')


def extract_word_embeddings_features(series: pd.Series):
    features = OrderedDict()

    extract_word_embeddings_features(series, features)

    return features


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding word embedding features
def extract_word_embeddings_features(col_values: list, features: OrderedDict):

    num_embeddings = 50

    embeddings = []

    global word_to_embedding

    if not word_to_embedding:
        initialise_word_embeddings()

    for v in col_values:
        v = str(v).lower()

        if v in word_to_embedding:
            embeddings.append(word_to_embedding.get(v))
        else:
            words = v.split(' ')
            embeddings_to_all_words = []

            for w in words:
                if w in word_to_embedding:
                    embeddings_to_all_words.append(word_to_embedding.get(w))

            if embeddings_to_all_words:
                mean_of_word_embeddings = np.nanmean(embeddings_to_all_words, axis=0)
                embeddings.append(mean_of_word_embeddings)

    if len(embeddings) == 0:
        # need to maintain same insertion order as the other case, hence running for loop per feature
        for i in range(num_embeddings): features['word_embedding_avg_' + str(i)] = np.nan
        for i in range(num_embeddings): features['word_embedding_std_' + str(i)] = np.nan
        for i in range(num_embeddings): features['word_embedding_med_' + str(i)] = np.nan
        for i in range(num_embeddings): features['word_embedding_mode_' + str(i)] = np.nan

        features['word_embedding_feature'] = 0

        return features

    else:
        mean_embeddings = np.nanmean(embeddings, axis=0)
        med_embeddings = np.nanmedian(embeddings, axis=0)
        std_embeddings = np.nanstd(embeddings, axis=0)

        # if only one dimension, then mode is equivalent to the embedding data
        if len(embeddings) == 1:
            mode_embeddings = embeddings[0]
        else:
            mode_embeddings = stats.mode(embeddings, axis=0, nan_policy='omit')[0].flatten()

        for i, e in enumerate(mean_embeddings): features['word_embedding_avg_' + str(i)] = e
        for i, e in enumerate(std_embeddings): features['word_embedding_std_' + str(i)] = e
        for i, e in enumerate(med_embeddings): features['word_embedding_med_' + str(i)] = e
        for i, e in enumerate(mode_embeddings): features['word_embedding_mode_' + str(i)] = e

        features['word_embedding_feature'] = 1
