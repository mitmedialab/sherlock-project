import numpy as np

from scipy import stats
from collections import OrderedDict


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding word embedding features
def extract_word_embeddings_features(values):

    num_embeddings = 50
    f = OrderedDict()
    embeddings = []

    word_vectors_f = open('../sherlock/features/glove.6B.50d.txt', encoding='utf-8')
    word_to_embedding = {}

    for w in word_vectors_f:

        term, vector = w.strip().split(' ', 1)
        vector = np.array(vector.split(' '), dtype=float)
        word_to_embedding[term] = vector
    
    values = values.dropna()

    for v in values:

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
        for i in range(num_embeddings): f['word_embedding_avg_{}'.format(i)] = np.nan
        for i in range(num_embeddings): f['word_embedding_std_{}'.format(i)] = np.nan
        for i in range(num_embeddings): f['word_embedding_med_{}'.format(i)] = np.nan
        for i in range(num_embeddings): f['word_embedding_mode_{}'.format(i)] = np.nan

        f['word_embedding_feature'] = 0

        return f

    else:
        mean_embeddings = np.nanmean(embeddings, axis=0)
        med_embeddings = np.nanmedian(embeddings, axis=0)
        std_embeddings = np.nanstd(embeddings, axis=0)
        mode_embeddings = stats.mode(embeddings, axis=0, nan_policy='omit')[0].flatten()

        for i, e in enumerate(mean_embeddings): f['word_embedding_avg_{}'.format(i)] = e
        for i, e in enumerate(std_embeddings): f['word_embedding_std_{}'.format(i)] = e
        for i, e in enumerate(med_embeddings): f['word_embedding_med_{}'.format(i)] = e
        for i, e in enumerate(mode_embeddings): f['word_embedding_mode_{}'.format(i)] = e

        f['word_embedding_feature'] = 1

        return f
