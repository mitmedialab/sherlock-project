import multiprocessing
import os
import random
import re

from collections import OrderedDict
from datetime import datetime
from functools import partial

import pyarrow.lib

from functional import pseq, seq
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features
from sherlock.features.helpers import literal_eval_as_str, keys_to_csv
from sherlock.global_state import is_first, set_first, reset_first


def as_py_str(x):
    return x.as_py() if isinstance(x, pyarrow.lib.StringScalar) else x


def to_string_list(x):
    return literal_eval_as_str(x, none_value="")


def random_sample(values: list):
    random.seed(13)
    return random.sample(values, k=min(1000, len(values)))


# Clean whitespace from strings by:
#   * trimming leading and trailing whitespace
#   * normalising all whitespace to spaces
#   * reducing whitespace sequences to a single space
def normalise_whitespace(data):
    if isinstance(data, str):
        return re.sub(r"\s{2,}", " ", data.strip())
    else:
        return data


def normalise_string_whitespace(col_values: list):
    return list(map(normalise_whitespace, col_values))


def extract_features(col_values: list):
    features = OrderedDict()

    extract_bag_of_characters_features(col_values, features)
    extract_word_embeddings_features(col_values, features)
    extract_bag_of_words_features(col_values, features, len(col_values))
    infer_paragraph_embeddings_features(col_values, features, dim=400, reuse_model=True)

    return features


def normalise_float(value):
    if isinstance(value, str):
        return value

    return "%g" % value


def values_to_str(values):
    return ",".join(map(normalise_float, values)) + "\n"


def numeric_values_to_str(od: OrderedDict):
    return od.keys(), values_to_str(od.values())


def keys_on_first(key_value_tuple, first_keys_only: bool):
    if first_keys_only:
        if is_first():
            set_first()
            return list(key_value_tuple[0]), key_value_tuple[1]
        else:
            return None, key_value_tuple[1]
    else:
        return list(key_value_tuple[0]), key_value_tuple[1]


# Only return OrderedDict.values. Useful in some benchmarking scenarios.
def values_only(od: OrderedDict):
    return list(od.values())


# Eliminate serialisation overhead for return values. Useful in some benchmarking scenarios.
def black_hole(od: OrderedDict):
    return None


def ensure_path_exists(output_path):
    path = os.path.dirname(output_path)

    if not os.path.exists(path):
        os.makedirs(path)


def extract_features_to_csv(output_path, parquet_values):
    verify_keys = False
    first_keys = None
    i = 0
    key_count = 0
    core_count = multiprocessing.cpu_count()

    reset_first()

    # retrieve keys for every row only if verify_keys=True
    drop_keys = partial(keys_on_first, first_keys_only=(not verify_keys))

    start = datetime.now()

    print(
        f"Starting {output_path} at {start}. Rows={len(parquet_values)}, using {core_count} CPU cores"
    )

    ensure_path_exists(output_path)

    with open(output_path, "w") as outfile:
        # Comparable performance with using pool.imap directly, but the code is *much* cleaner
        # for keys, values_str in seq(map(as_py_str, parquet_values)) \
        for keys, values_str in (
            pseq(
                map(as_py_str, parquet_values), processes=core_count, partition_size=100
            )
            .map(to_string_list)
            .map(random_sample)
            .map(normalise_string_whitespace)
            .map(extract_features)
            .map(numeric_values_to_str)
            .map(drop_keys)
        ):
            i = i + 1

            if keys is not None:
                key_count = key_count + 1

            if first_keys is None:
                first_keys = keys
                first_keys_str = keys_to_csv(keys)

                print(f"Exporting {len(first_keys)} column features")

                outfile.write(keys_to_csv(keys))
            elif verify_keys:
                keys_str = ",".join(keys)
                if first_keys_str != keys_str:
                    key_list = list(keys)

                    print(
                        f"keys are NOT equal. k1 len={len(first_keys)}, k2 len={len(keys)}"
                    )

                    for idx, k1 in enumerate(first_keys):
                        k2 = key_list[idx]

                        if k1 != k2:
                            print(f"{k1} != {k2}")

            outfile.write(values_str)

    print(
        f"Finished. Processed {i} rows in {datetime.now() - start}, key_count={key_count}"
    )
