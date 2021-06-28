import string
from collections import OrderedDict

import numpy as np
from scipy.stats import kurtosis, skew

ignore_chars = {"\n", "\\", "\v", "\r", "\t", "^"}
characters_to_check = [f"[{c}]" for c in string.printable if c not in ignore_chars]
characters_to_check.extend(["[\\\\]", "[\^]"])


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(data):
    f = OrderedDict()

    all_value_features = OrderedDict()

    for c in characters_to_check:
        all_value_features[f"n_{c}"] = data.str.count(c)

    for value_feature_name, value_features in all_value_features.items():
        f[f"{value_feature_name}-agg-any"] = any(value_features)
        f[f"{value_feature_name}-agg-all"] = all(value_features)
        f[f"{value_feature_name}-agg-mean"] = np.mean(value_features)
        f[f"{value_feature_name}-agg-var"] = np.var(value_features)
        f[f"{value_feature_name}-agg-min"] = np.min(value_features)
        f[f"{value_feature_name}-agg-max"] = np.max(value_features)
        f[f"{value_feature_name}-agg-median"] = np.median(value_features)
        f[f"{value_feature_name}-agg-sum"] = np.sum(value_features)
        f[f"{value_feature_name}-agg-kurtosis"] = kurtosis(value_features)
        f[f"{value_feature_name}-agg-skewness"] = skew(value_features)

    return f
