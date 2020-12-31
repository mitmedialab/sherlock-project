import math
import nltk
import numpy as np
from sherlock.features.stats_helper import compute_stats
from collections import OrderedDict
import pandas as pd
from sherlock.features.helpers import escape_for_regex
import string
import re
from array import array


def extract_bag_of_words_features(series: pd.Series, n_val):
    features = OrderedDict()

    extract_bag_of_words_features(series, features, n_val)

    return features


def count_pattern_in_cells(series: pd.Series, pat: re.Pattern):
    cell_counts = array('i')
    matching_cell_count = 0
    for s in series:
        char_count = len(re.findall(pat, s))
        if char_count > 0:
            matching_cell_count = matching_cell_count + 1
        cell_counts.append(char_count)

    return matching_cell_count, cell_counts


NUMBER_PATTERN = re.compile(r'[0-9]')
TEXT_PATTERN = re.compile(r'[a-zA-Z]')
WORD_PATTERN = re.compile(r'[\w+]')

SPECIAL_CHARACTERS_REGEX = '[' + ''.join(escape_for_regex(c) for c in string.printable
                                         if c not in ('\n', '\f', '\v', '\r', '\t')
                                         and not re.match(r'[a-zA-Z0-9\s]', c)) + ']'

SPECIAL_CHARACTERS_PATTERN = re.compile(SPECIAL_CHARACTERS_REGEX)


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(series: pd.Series, features: OrderedDict, n_val):
    if not n_val:
        return

    # Entropy of column
    freq_dist = nltk.FreqDist(series)
    probs = [freq_dist.freq(l) for l in freq_dist]
    features['col_entropy'] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = series.nunique()
    features['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    numeric_cell_count, numeric_char_counts = count_pattern_in_cells(series, NUMBER_PATTERN)
    text_cell_count, text_char_counts = count_pattern_in_cells(series, TEXT_PATTERN)

    features['frac_numcells'] = numeric_cell_count / n_val
    features['frac_textcells'] = text_cell_count / n_val

    # Average + std number of numeric tokens in cells
    features['avg_num_cells'] = np.mean(numeric_char_counts)
    features['std_num_cells'] = np.std(numeric_char_counts)

    # Average + std number of textual tokens in cells
    features['avg_text_cells'] = np.mean(text_char_counts)
    features['std_text_cells'] = np.std(text_char_counts)

    # Average + std number of special characters in each cell
    spec_cells, spec_char_counts = count_pattern_in_cells(series, SPECIAL_CHARACTERS_PATTERN)

    features['avg_spec_cells'] = np.mean(spec_char_counts)
    features['std_spec_cells'] = np.std(spec_char_counts)

    # Average number of words in each cell
    word_cells, word_counts = count_pattern_in_cells(series, WORD_PATTERN)

    features['avg_word_cells'] = np.mean(word_counts)
    features['std_word_cells'] = np.std(word_counts)

    features['n_values'] = n_val

    lengths = array('i')

    n_none = 0
    for s in series:
        str_len = len(s)

        if str_len == 0:
            n_none = n_none + 1

        lengths.append(str_len)

    has_any = any(lengths)

    if has_any:
        _mean, _variance, _skew, _kurtosis = compute_stats(lengths)

        features['length-agg-any'] = 1
        features['length-agg-all'] = 1 if all(lengths) else 0
        features['length-agg-mean'] = _mean
        features['length-agg-var'] = _variance
        features['length-agg-min'] = np.min(lengths)
        features['length-agg-max'] = np.max(lengths)
        features['length-agg-median'] = np.median(lengths)
        features['length-agg-sum'] = np.sum(lengths)
        features['length-agg-kurtosis'] = _kurtosis
        features['length-agg-skewness'] = _skew
    else:
        features['length-agg-any'] = 0
        features['length-agg-all'] = 0
        features['length-agg-mean'] = 0
        features['length-agg-var'] = 0
        features['length-agg-min'] = 0
        features['length-agg-max'] = 0
        features['length-agg-median'] = 0
        features['length-agg-sum'] = 0
        features['length-agg-kurtosis'] = -3
        features['length-agg-skewness'] = 0

    features['none-agg-has'] = 1 if n_none > 0 else 0
    features['none-agg-percent'] = n_none / n_val
    features['none-agg-num'] = n_none
    features['none-agg-all'] = 1 if n_none == n_val else 0
