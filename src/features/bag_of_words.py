import math
import nltk
import numpy as np
from scipy.stats import skew, kurtosis
from collections import OrderedDict


# Input: a single column in the form of a pandas series
# Output: ordered dictionary holding bag of words features
def extract_bag_of_words_features(data, n_val):
    
    f = OrderedDict()
    data = data.dropna()
    
    #n_val = data.size
    
    if not n_val: return
    
    # Entropy of column
    freq_dist = nltk.FreqDist(data)
    probs = [freq_dist.freq(l) for l in freq_dist]
    f['col_entropy'] = -sum(p * math.log(p,2) for p in probs)

    # Fraction of cells with unique content
    num_unique = data.nunique()
    f['frac_unique'] = num_unique / n_val

    # Fraction of cells with numeric content -> frac text cells doesn't add information
    num_cells = np.sum(data.str.contains('[0-9]', regex=True))
    text_cells = np.sum(data.str.contains('[a-z]|[A-Z]', regex=True))
    f['frac_numcells']  = num_cells / n_val
    f['frac_textcells'] = text_cells / n_val
    
    # Average + std number of numeric tokens in cells
    num_reg = '[0-9]'
    f['avg_num_cells'] = np.mean(data.str.count(num_reg))
    f['std_num_cells'] = np.std(data.str.count(num_reg))
    
    # Average + std number of textual tokens in cells
    text_reg = '[a-z]|[A-Z]'
    f['avg_text_cells'] = np.mean(data.str.count(text_reg))
    f['std_text_cells'] = np.std(data.str.count(text_reg))
    
    # Average + std number of special characters in each cell
    spec_reg = '[[!@#$%^&*(),.?":{}|<>]]'
    f['avg_spec_cells'] = np.mean(data.str.count(spec_reg))
    f['std_spec_cells'] = np.std(data.str.count(spec_reg))
    
    # Average number of words in each cell
    space_reg = '[" "]'
    f['avg_word_cells'] = np.mean(data.str.count(space_reg) + 1)
    f['std_word_cells'] = np.std(data.str.count(space_reg) + 1)

    all_value_features = OrderedDict()

    data_no_null = data.dropna()

    f['n_values'] = n_val

    all_value_features['length'] = data_no_null.apply(len)

    for value_feature_name, value_features in all_value_features.items():
        f['{}-agg-any'.format(value_feature_name)] = any(value_features)
        f['{}-agg-all'.format(value_feature_name)] = all(value_features)
        f['{}-agg-mean'.format(value_feature_name)] = np.mean(value_features)
        f['{}-agg-var'.format(value_feature_name)] = np.var(value_features)
        f['{}-agg-min'.format(value_feature_name)] = np.min(value_features)
        f['{}-agg-max'.format(value_feature_name)] = np.max(value_features)
        f['{}-agg-median'.format(value_feature_name)] = np.median(value_features)
        f['{}-agg-sum'.format(value_feature_name)] = np.sum(value_features)
        f['{}-agg-kurtosis'.format(value_feature_name)] = kurtosis(value_features)
        f['{}-agg-skewness'.format(value_feature_name)] = skew(value_features)

    n_none = data.size - data_no_null.size - len([ e for e in data if e == ''])
    f['none-agg-has'] = n_none > 0
    f['none-agg-percent'] = n_none / len(data)
    f['none-agg-num'] = n_none
    f['none-agg-all'] = (n_none == len(data))
    
    return f
