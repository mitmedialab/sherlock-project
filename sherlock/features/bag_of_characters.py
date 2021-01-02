import statistics as statistics
import pandas as pd
from sherlock.features.stats_helper import compute_stats
from collections import OrderedDict
from sherlock.features.helpers import CHARACTERS_TO_CHECK
from array import array


# Input: a single column in the form of pandas series
# Output: ordered dictionary holding bag of character features
def extract_bag_of_characters_features(col_values: list, features: OrderedDict):

    # Create a set of unique chars from the string vectors to quickly test whether to perform expensive
    # processing for any given char
    char_set = set(''.join(col_values))

    for c in CHARACTERS_TO_CHECK:
        value_feature_name = f'n_[{c}]'

        if c in char_set:
            counts = array('i')
            for s in col_values:
                counts.append(s.count(c))

            has_any = any(counts)
        else:
            has_any = False

        if has_any:
            _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(counts)

            features[value_feature_name + '-agg-any'] = 1
            features[value_feature_name + '-agg-all'] = 1 if all(counts) else 0
            features[value_feature_name + '-agg-mean'] = _mean
            features[value_feature_name + '-agg-var'] = _variance
            features[value_feature_name + '-agg-min'] = _min
            features[value_feature_name + '-agg-max'] = _max
            features[value_feature_name + '-agg-median'] = statistics.median(counts)
            features[value_feature_name + '-agg-sum'] = _sum
            features[value_feature_name + '-agg-kurtosis'] = _kurtosis
            features[value_feature_name + '-agg-skewness'] = _skew
        else:
            features[value_feature_name + '-agg-any'] = 0
            features[value_feature_name + '-agg-all'] = 0
            features[value_feature_name + '-agg-mean'] = 0
            features[value_feature_name + '-agg-var'] = 0
            features[value_feature_name + '-agg-min'] = 0
            features[value_feature_name + '-agg-max'] = 0
            features[value_feature_name + '-agg-median'] = 0
            features[value_feature_name + '-agg-sum'] = 0
            features[value_feature_name + '-agg-kurtosis'] = -3
            features[value_feature_name + '-agg-skewness'] = 0
