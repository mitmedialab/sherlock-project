import string
import numpy  as np
import pandas as pd
from scipy.stats import skew, kurtosis
from collections import OrderedDict
from re import search

ZERO_SERIES = pd.Series([0], name='Empty')

CHARACTERS_TO_CHECK = (
        ['[' + c + ']' for c in string.printable if c not in ('\n', '\\', '\v', '\r', '\t', '^', '[', ']')]
        + ['[\\\\]', '[\\^]', '[\\[]', '[\\]]']
)

ZERO_FLOAT64 = np.float64(0)
DEFAULT_KURTOSIS_FLOAT64 = np.float64(-3.0)


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(data_no_null):
    all_value_features = OrderedDict()

    # Create a set of unique chars from the string vectors to quickly test whether to perform expensive
    # counting for any given char in the subsequent loop
    char_set = ''.join(set(''.join(data_no_null.str.join(''))))

    for c in CHARACTERS_TO_CHECK:
        if search(c, char_set):
            all_value_features['n_' + c] = data_no_null.str.count(c)
        else:
            all_value_features['n_' + c] = ZERO_SERIES

    f = OrderedDict()

    for value_feature_name, value_features in all_value_features.items():
        if value_features.name == 'Empty':
            has_any = False
        else:
            has_any = any(value_features)

        if has_any:
            f[value_feature_name + '-agg-any'] = has_any
            f[value_feature_name + '-agg-all'] = all(value_features)
            f[value_feature_name + '-agg-mean'] = np.mean(value_features)
            f[value_feature_name + '-agg-var'] = np.var(value_features)
            f[value_feature_name + '-agg-min'] = np.min(value_features)
            f[value_feature_name + '-agg-max'] = np.max(value_features)
            f[value_feature_name + '-agg-median'] = np.median(value_features)
            f[value_feature_name + '-agg-sum'] = np.sum(value_features)
            f[value_feature_name + '-agg-kurtosis'] = kurtosis(value_features)
            f[value_feature_name + '-agg-skewness'] = skew(value_features)
        else:
            f[value_feature_name + '-agg-any'] = False
            f[value_feature_name + '-agg-all'] = False
            f[value_feature_name + '-agg-mean'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-var'] = ZERO_FLOAT64
            f[value_feature_name + '-agg-min'] = 0
            f[value_feature_name + '-agg-max'] = 0
            f[value_feature_name + '-agg-median'] = 0
            f[value_feature_name + '-agg-sum'] = 0
            f[value_feature_name + '-agg-kurtosis'] = DEFAULT_KURTOSIS_FLOAT64
            f[value_feature_name + '-agg-skewness'] = ZERO_FLOAT64
