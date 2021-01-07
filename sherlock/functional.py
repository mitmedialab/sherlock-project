import random
import pyarrow.lib
import re
import io
import csv
from collections import OrderedDict
from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features
from sherlock.features.helpers import literal_eval_as_str
from sherlock.global_state import is_first, set_first
from datetime import datetime
from functional import pseq
from functools import partial


def as_py_str(x: pyarrow.lib.StringScalar):
    return x.as_py()


def to_string_list(x):
    return literal_eval_as_str(x, none_value='')


def random_sample(values: list):
    random.seed(13)
    return random.sample(values, k=min(1000, len(values)))


# Clean whitespace from strings by:
#   * trimming leading and trailing whitespace
#   * normalising all whitespace to spaces
#   * reducing whitespace sequences to a single space
def normalise_whitespace(data):
    if isinstance(data, str):
        return re.sub(r'\s{2,}', ' ', data.strip())
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

    return '%g' % value


def values_to_str(values):
    return ','.join(map(normalise_float, values)) + '\n'


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


def keys_to_csv(keys):
    with io.StringIO() as output:
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(keys)

        return output.getvalue()


def extract_features_to_csv(output_path, parquet_values):
    verify_keys = False
    first_keys = None
    i = 0
    key_count = 0

    # retrieve keys for every row only if verify_keys=True
    drop_keys = partial(keys_on_first, first_keys_only=(not verify_keys))

    start = datetime.now()

    print(f'Starting {output_path} at {start}')

    with open(output_path, "w") as outfile:
        # Comparable performance with using pool.imap directly, but the code is *much* cleaner
        for keys, values_str in pseq(map(as_py_str, parquet_values), processes=8, partition_size=100) \
                .map(to_string_list) \
                .map(random_sample) \
                .map(normalise_string_whitespace) \
                .map(extract_features) \
                .map(numeric_values_to_str) \
                .map(drop_keys):
            i = i + 1

            if keys is not None:
                key_count = key_count + 1

            if first_keys is None:
                first_keys = keys
                first_keys_str = keys_to_csv(keys)

                print(f'Exporting {len(first_keys)} column features')

                outfile.write(keys_to_csv(keys))
            elif verify_keys:
                keys_str = ','.join(keys)
                if first_keys_str != keys_str:
                    key_list = list(keys)

                    print(f'keys are NOT equal. k1 len={len(first_keys)}, k2 len={len(keys)}')

                    for idx, k1 in enumerate(first_keys):
                        k2 = key_list[idx]

                        if k1 != k2:
                            print(f'{k1} != {k2}')

            outfile.write(values_str)

    print(f'Finished. Processed {i} rows in {datetime.now() - start}, key_count={key_count}')


def extract_features_to_csv2(output_path, parquet_values, n=None):
    verify_keys = False
    first_keys = None
    i = 0
    key_count = 0

    # retrieve keys for every row only if verify_keys=True
    drop_keys = partial(keys_on_first, first_keys_only=(not verify_keys))

    start = datetime.now()

    print(f'Starting {output_path} at {start}')

    with open(output_path, "w") as outfile:
        for pv in parquet_values:
            v = as_py_str(pv)

            v_str = to_string_list(v)
            rnd_sample = random_sample(v_str)
            normalised = normalise_string_whitespace(rnd_sample)

            features = extract_features(normalised)

            keys, values_str = numeric_values_to_str(features)

            i = i + 1

            if keys is not None:
                key_count = key_count + 1

            if first_keys is None:
                first_keys = keys
                first_keys_str = keys_to_csv(keys)

                print(f'Exporting {len(first_keys)} column features')

                outfile.write(keys_to_csv(keys))
                set_first()
            elif verify_keys:
                keys_str = ','.join(keys)
                if first_keys_str != keys_str:
                    key_list = list(keys)

                    print(f'keys are NOT equal. k1 len={len(first_keys)}, k2 len={len(keys)}')

                    for idx, k1 in enumerate(first_keys):
                        k2 = key_list[idx]

                        if k1 != k2:
                            print(f'{k1} != {k2}')

            outfile.write(values_str)

            if n is not None:
                if i == n:
                    break

    print(f'Finished. Processed {i} rows in {datetime.now() - start}, key_count={key_count}')
