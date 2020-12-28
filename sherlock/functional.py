from ast import literal_eval
import random
import pandas as pd
import pyarrow.lib
import re
from collections import OrderedDict
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features


def as_py_str(x: pyarrow.lib.StringScalar):
    return x.as_py()


def to_literal(x):
    return literal_eval(x)


def randomise_sample(values):
    random.seed(13)
    return pd.Series(random.sample(values, k=min(1000, len(values))))


# Clean whitespace from strings by:
#   * trimming leading and trailing whitespace
#   * normalising all whitespace to spaces
#   * reducing whitespace sequences to a single space
def normalise_whitespace(data):
    if isinstance(data, str):
        return re.sub(r'\s{2,}', ' ', data.strip())
    else:
        return data


def normalise_string_whitespace(series: pd.Series):
    return series.apply(normalise_whitespace)


def as_str_series(series: pd.Series):
    return series.astype(str)


def dropna(series: pd.Series):
    return series.dropna()


def extract_features(series: pd.Series):
    features = OrderedDict()

    extract_bag_of_characters_features(series, features)
    extract_word_embeddings_features(series, features)
    extract_bag_of_words_features(series, features, series.count())
    infer_paragraph_embeddings_features(series, features, dim=400, reuse_model=True)

    return features


is_first = True


# prints floats without using scientific notation
# remove small imprecision (7 zeros before last digit) - e.g. ('%.16f' % 1.35)  -> '1.3500000000000001'
# remove trailing zeros and decimal point
def normalise_float(value):
    # return re.sub(r'0{7,}[0-9]$', '', ('%.16f' % value)).rstrip('0').rstrip('.')
    return '%g' % value


def values_to_str(values):
    return ','.join(map(normalise_float, values))


def keys_on_first(od: OrderedDict):
    global is_first

    if is_first:
        is_first = False
        return list(od.keys()), values_to_str(od.values())
    else:
        return None, values_to_str(od.values())


# Only return OrderedDict.values. Useful in some benchmarking scenarios.
def values_only(od: OrderedDict):
    return list(od.values())


# Eliminate serialisation overhead for return values. Useful in some benchmarking scenarios.
def black_hole(od: OrderedDict):
    return None
