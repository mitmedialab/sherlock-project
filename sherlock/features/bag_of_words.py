import math
import nltk
import numpy as np
from sherlock.features.stats_helper import compute_stats
from collections import OrderedDict
from sherlock.features.helpers import escape_for_regex
import string
import re
import statistics as statistics
from sherlock.global_state import is_first


def count_pattern_in_cells(values: list, pat):
    return [len(re.findall(pat, s)) for s in values]


def count_pattern_in_cells_with_non_zero_count(values: list, pat):
    cell_counts = [len(re.findall(pat, s)) for s in values]

    return sum(1 for c in cell_counts if c > 0), cell_counts


NUMBER_PATTERN = re.compile(r"[0-9]")
TEXT_PATTERN = re.compile(r"[a-zA-Z]")
WORD_PATTERN = re.compile(r"[\w+]")


# SPECIAL_CHARACTERS_REGEX = '[' + ''.join(escape_for_regex(c) for c in string.printable
#                                          if c not in ('\n', '\f', '\r', '\t')
#                                          and not re.match(r'[a-zA-Z0-9\s]', c)) + ']'
#
# SPECIAL_CHARACTERS_PATTERN = re.compile(SPECIAL_CHARACTERS_REGEX)

SPECIAL_CHARACTERS_PATTERN = re.compile(r'[!@#$%^&*(),.?":{}|<>]')


# Input: a single column in the form of a Python list
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(col_values: list, features: OrderedDict, n_val):
    if not n_val:
        return

    # Entropy of column
    freq_dist = nltk.FreqDist(col_values)
    probs = [freq_dist.freq(l) for l in freq_dist]
    features["col_entropy"] = -sum(p * math.log(p, 2) for p in probs)

    # Fraction of cells with unique content
    num_unique = len(set(col_values))
    features["frac_unique"] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    numeric_cell_nz_count, numeric_char_counts = count_pattern_in_cells_with_non_zero_count(
        col_values, NUMBER_PATTERN
    )
    text_cell_nz_count, text_char_counts = count_pattern_in_cells_with_non_zero_count(
        col_values, TEXT_PATTERN
    )

    features["frac_numcells"] = numeric_cell_nz_count / n_val
    features["frac_textcells"] = text_cell_nz_count / n_val

    # Average + std number of numeric tokens in cells
    features["avg_num_cells"] = np.mean(numeric_char_counts)
    features["std_num_cells"] = np.std(numeric_char_counts)

    # Average + std number of textual tokens in cells
    features["avg_text_cells"] = np.mean(text_char_counts)
    features["std_text_cells"] = np.std(text_char_counts)

    # Average + std number of special characters in each cell
    spec_char_counts = count_pattern_in_cells(col_values, SPECIAL_CHARACTERS_PATTERN)

    features["avg_spec_cells"] = np.mean(spec_char_counts)
    features["std_spec_cells"] = np.std(spec_char_counts)

    # Average number of words in each cell
    word_counts = count_pattern_in_cells(col_values, WORD_PATTERN)

    features["avg_word_cells"] = np.mean(word_counts)
    features["std_word_cells"] = np.std(word_counts)

    features["n_values"] = n_val

    lengths = [len(s) for s in col_values]
    n_none = sum(1 for _l in lengths if _l == 0)

    has_any = any(lengths)

    if has_any:
        _any = 1
        _all = 1 if all(lengths) else 0
        _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(lengths)
        _median = statistics.median(lengths)

        if is_first():
            # the first output needs fully expanded keys (to drive CSV header)
            features["length-agg-any"] = _any
            features["length-agg-all"] = _all
            features["length-agg-mean"] = _mean
            features["length-agg-var"] = _variance
            features["length-agg-min"] = _min
            features["length-agg-max"] = _max
            features["length-agg-median"] = _median
            features["length-agg-sum"] = _sum
            features["length-agg-kurtosis"] = _kurtosis
            features["length-agg-skewness"] = _skew
        else:
            # subsequent lines only care about values, so we can pre-render a block of CSV. This
            # cuts overhead of storing granular values in the features dictionary
            features[
                "length-pre-rendered"
            ] = f"{_any},{_all},{_mean},{_variance},{_min},{_max},{_median},{_sum},{_kurtosis},{_skew}"
    else:
        if is_first():
            features["length-agg-any"] = 0
            features["length-agg-all"] = 0
            features["length-agg-mean"] = 0
            features["length-agg-var"] = 0
            features["length-agg-min"] = 0
            features["length-agg-max"] = 0
            features["length-agg-median"] = 0
            features["length-agg-sum"] = 0
            features["length-agg-kurtosis"] = -3
            features["length-agg-skewness"] = 0
        else:
            # assign pre-rendered defaults
            features["length-pre-rendered"] = "0,0,0,0,0,0,0,0,-3,0"

    features["none-agg-has"] = 1 if n_none > 0 else 0
    features["none-agg-percent"] = n_none / n_val
    features["none-agg-num"] = n_none
    features["none-agg-all"] = 1 if n_none == n_val else 0
