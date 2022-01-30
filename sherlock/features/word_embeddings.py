import numpy as np

from collections import OrderedDict
from datetime import datetime
from sherlock.global_state import is_first
import itertools
import statistics as statistics
import math as math
from sherlock.features.stats_helper import mode

word_to_embedding = {}


def initialise_word_embeddings():
    start = datetime.now()

    global word_to_embedding

    print("Initialising word embeddings")
    word_vectors_f = open('../sherlock/features/glove.6B.50d.txt', encoding='utf-8')

    for w in word_vectors_f:
        term, vector = w.strip().split(' ', 1)
        vector = list(map(float, vector.split(' ')))

        word_to_embedding[term] = vector

    end = datetime.now()
    x = end - start

    word_vectors_f.close()

    print(f'Initialise Word Embeddings process took {x} seconds.')


ZEROS = [0] * 50

# nans for mean, median, stdev and mode for each embedding
NANS = ','.join(map(str, [np.nan] * 50 * 4))


def transpose(a):
    # transpose array:
    #   >>> theArray = [['a','b','c'],['d','e','f'],['g','h','i']]
    #   >>> [*zip(*theArray)]
    #   [('a', 'd', 'g'), ('b', 'e', 'h'), ('c', 'f', 'i')]
    #
    #   https://stackoverflow.com/questions/4937491/matrix-transpose-in-python
    return [*zip(*a)]


# Input: a single column in the form of a Python list
# Output: ordered dictionary holding word embedding features
def extract_word_embeddings_features(col_values: list, features: OrderedDict):
    num_embeddings = 50

    embeddings = []

    global word_to_embedding

    if not word_to_embedding:
        initialise_word_embeddings()

    for col_value in map(str.lower, col_values):
        if col_value in word_to_embedding:
            embeddings.append(word_to_embedding.get(col_value))
        else:
            embeddings_to_all_words = [word_to_embedding.get(w) for w in col_value.split(' ') if w in word_to_embedding]

            n = len(embeddings_to_all_words)

            if n == 1:
                embeddings.append(embeddings_to_all_words[0])
            elif n > 1:
                mean_of_word_embeddings = np.mean(embeddings_to_all_words, dtype=float, axis=0)
                embeddings.append(mean_of_word_embeddings)

    n_rows = len(embeddings)

    if n_rows == 0:
        if is_first():
            # the first output needs fully expanded keys (to drive CSV header)
            # need to maintain same insertion order as the other case, hence running for loop per feature
            for i in range(num_embeddings):
                features['word_embedding_avg_' + str(i)] = np.nan
            for i in range(num_embeddings):
                features['word_embedding_std_' + str(i)] = np.nan
            for i in range(num_embeddings):
                features['word_embedding_med_' + str(i)] = np.nan
            for i in range(num_embeddings):
                features['word_embedding_mode_' + str(i)] = np.nan
        else:
            # subsequent lines only care about values, so we can pre-render a block of CSV. This
            # cuts overhead of storing granular values in the features dictionary
            features['word_embedding-pre-rendered'] = NANS

        features['word_embedding_feature'] = 0

    else:
        if n_rows > 1:
            mean_embeddings = []
            std_embeddings = []
            med_embeddings = []
            mode_embeddings = []

            # transpose array (if using numpy, stats would operate on axis=0):
            for axis0 in transpose(embeddings):
                # mode requires sorted list, and Python's sort is super quick on presorted arrays. The subsequent
                # median calc also calls sorted(), so this helps there, too.
                #
                # Re: sorting in Python
                #   https://stackoverflow.com/questions/1436962/python-sort-method-on-list-vs-builtin-sorted-function
                axis0 = sorted(axis0)

                _mean = sum(axis0) / n_rows

                mean_embeddings.append(_mean)

                _variance = sum((x - _mean) ** 2 for x in axis0) / n_rows
                std_embeddings.append(math.sqrt(_variance))

                med_embeddings.append(statistics.median(axis0))

                mode_embeddings.append(mode(axis0, True))
        # n_rows == 1
        else:
            # if only one dimension, then mean, median and mode are equivalent to the embedding data
            mean_embeddings = med_embeddings = mode_embeddings = embeddings[0]
            std_embeddings = ZEROS

        if is_first():
            # the first output needs fully expanded keys (to drive CSV header)
            for i, e in enumerate(mean_embeddings):
                features['word_embedding_avg_' + str(i)] = e
            for i, e in enumerate(std_embeddings):
                features['word_embedding_std_' + str(i)] = e
            for i, e in enumerate(med_embeddings):
                features['word_embedding_med_' + str(i)] = e
            for i, e in enumerate(mode_embeddings):
                features['word_embedding_mode_' + str(i)] = e
        else:
            # subsequent lines only care about values, so we can pre-render a block of CSV. This
            # cuts overhead of storing granular values in the features dictionary
            features['word_embedding-pre-rendered'] = \
                ','.join(map(lambda x: '%g' % x,
                             itertools.chain(mean_embeddings, std_embeddings, med_embeddings, mode_embeddings)))

        features['word_embedding_feature'] = 1
