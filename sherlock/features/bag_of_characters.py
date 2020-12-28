import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from collections import OrderedDict
from re import search
from sherlock.features.helpers import CHARACTERS_TO_CHECK


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(series: pd.Series, features: OrderedDict):
    all_value_features = OrderedDict()

    # Create a set of unique chars from the string vectors to quickly test whether to perform expensive
    # counting for any given char in the subsequent loop
    char_set = ''.join(set(''.join(series.str.join(''))))

    for c in CHARACTERS_TO_CHECK:
        if search(c, char_set):
            all_value_features['n_' + c] = np.array(series.str.count(c))
        else:
            all_value_features['n_' + c] = None

    for value_feature_name, value_features in all_value_features.items():
        if value_features is None:
            has_any = False
        else:
            has_any = any(value_features)

        if has_any:
            features[value_feature_name + '-agg-any'] = 1
            features[value_feature_name + '-agg-all'] = 1 if all(value_features) else 0
            features[value_feature_name + '-agg-mean'] = float(np.mean(value_features))
            features[value_feature_name + '-agg-var'] = float(np.var(value_features))
            features[value_feature_name + '-agg-min'] = np.min(value_features)
            features[value_feature_name + '-agg-max'] = np.max(value_features)
            features[value_feature_name + '-agg-median'] = float(np.median(value_features))
            features[value_feature_name + '-agg-sum'] = np.sum(value_features)
            features[value_feature_name + '-agg-kurtosis'] = float(kurtosis(value_features))
            features[value_feature_name + '-agg-skewness'] = float(skew(value_features))
        else:
            features[value_feature_name + '-agg-any'] = 0
            features[value_feature_name + '-agg-all'] = 0
            features[value_feature_name + '-agg-mean'] = 0.0
            features[value_feature_name + '-agg-var'] = 0.0
            features[value_feature_name + '-agg-min'] = 0
            features[value_feature_name + '-agg-max'] = 0
            features[value_feature_name + '-agg-median'] = 0.0
            features[value_feature_name + '-agg-sum'] = 0
            features[value_feature_name + '-agg-kurtosis'] = -3.0
            features[value_feature_name + '-agg-skewness'] = 0.0
