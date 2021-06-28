import math
from collections import OrderedDict

import nltk
import numpy as np
from scipy.stats import kurtosis, skew


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(data):

    f = OrderedDict()
    n_val = data.shape[0]

    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = np.array([freq_dist.freq(item) for item in freq_dist])
    f["col_entropy"] = (probs * np.log2(probs)).sum()

    # Fraction of cells with unique content
    num_unique = len(freq_dist)
    f["frac_unique"] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = data.str.count("[0-9]")
    text_cells = data.str.count("[a-z]|[A-Z]")
    f["frac_numcells"] = num_cells[num_cells > 0].shape[0] / n_val
    f["frac_textcells"] = text_cells[text_cells > 0].shape[0] / n_val

    # Average + std number of numeric tokens in cells
    num_agg = num_cells.agg(["mean", "std"]).to_dict()
    f["avg_num_cells"] = num_agg["mean"]
    f["std_num_cells"] = num_agg["std"]

    # Average + std number of textual tokens in cells
    text_agg = text_cells.agg(["mean", "std"]).to_dict()
    f["avg_text_cells"] = text_agg["mean"]
    f["std_text_cells"] = text_agg["std"]

    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    spec_agg = data.str.count(spec_reg).agg(["mean", "std"]).to_dict()
    f["avg_spec_cells"] = spec_agg["mean"]
    f["std_spec_cells"] = spec_agg["std"]

    # Average number of words in each cell
    space_reg = '[" "]'
    word_agg = (data.str.count(space_reg) + 1).agg(["mean", "std"]).to_dict()
    f["avg_word_cells"] = word_agg["mean"]
    f["std_word_cells"] = word_agg["std"]

    all_value_features = OrderedDict()

    data_no_null = data.dropna() if data.hasnans else data

    f["n_values"] = n_val

    all_value_features["length"] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        f["{}-agg-any".format(value_feature_name)] = any(value_features)
        f["{}-agg-all".format(value_feature_name)] = all(value_features)
        f["{}-agg-mean".format(value_feature_name)] = np.mean(value_features)
        f["{}-agg-var".format(value_feature_name)] = np.var(value_features)
        f["{}-agg-min".format(value_feature_name)] = np.min(value_features)
        f["{}-agg-max".format(value_feature_name)] = np.max(value_features)
        f["{}-agg-median".format(value_feature_name)] = np.median(value_features)
        f["{}-agg-sum".format(value_feature_name)] = np.sum(value_features)
        f["{}-agg-kurtosis".format(value_feature_name)] = kurtosis(value_features)
        f["{}-agg-skewness".format(value_feature_name)] = skew(value_features)

    n_none = data.size - data_no_null.size - (data == "").sum()
    f["none-agg-has"] = n_none > 0
    f["none-agg-percent"] = n_none / len(data)
    f["none-agg-num"] = n_none
    f["none-agg-all"] = n_none == len(data)

    return f
