import random
import os

from collections import OrderedDict
from typing import Union, Tuple

import gdown
import pandas as pd

from functools import partial
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features
from sherlock.global_state import set_first, reset_first
from sherlock.features.helpers import literal_eval_as_str, keys_to_csv


def prepare_feature_extraction():
    """Download embedding files from Google Drive if they do not exist yet."""
    word_embedding_file = "../sherlock/features/glove.6B.50d.txt"
    first_paragraph_vector_file = (
        "../sherlock/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy"
    )
    second_paragraph_vector_file = (
        "../sherlock/features/par_vec_trained_400.pkl.trainables.syn1neg.npy"
    )
    third_paragraph_vector_file = (
        "../sherlock/features/par_vec_trained_400.pkl.wv.vectors.npy"
    )

    print(
        f"""Preparing feature extraction by downloading 4 files:
        \n {word_embedding_file}, \n {first_paragraph_vector_file},
        \n {second_paragraph_vector_file}, and \n {third_paragraph_vector_file}.
        """
    )

    if not os.path.exists(word_embedding_file):
        print("Downloading GloVe word embedding vectors.")
        file_name = word_embedding_file
        gdown.download(
            url="https://drive.google.com/uc?id=1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk",
            output=file_name,
        )

        print("GloVe word embedding vectors were downloaded.")

    if not os.path.exists(first_paragraph_vector_file):
        print("Downloading first paragraph vector file.")
        file_name = first_paragraph_vector_file
        gdown.download(
            url="https://drive.google.com/uc?id=1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu",
            output=file_name,
        )

    if not os.path.exists(second_paragraph_vector_file):
        print("Downloading second paragraph vector file.")
        file_name = second_paragraph_vector_file
        gdown.download(
            url="https://drive.google.com/uc?id=1hwE8We05oZLrACRibY8jc81NGughv79q",
            output=file_name,
        )

        print("Downloaded second paragraph vector file.")

    if not os.path.exists(third_paragraph_vector_file):
        print("Downloading third paragraph vector file.")
        file_name = third_paragraph_vector_file
        gdown.download(
            url="https://drive.google.com/uc?id=1StGoalk5SMbWX8Z-5weSbIAtK771UwoC",
            output=file_name,
        )

        print("Downloaded third paragraph vector file.")

    print("All files for extracting word and paragraph embeddings are present.")


def convert_string_lists_to_lists(
    data: Union[pd.DataFrame, pd.Series],
    labels: Union[pd.DataFrame, pd.Series],
    data_column_name: str = None,
    labels_column_name: str = None,
) -> Tuple[list, list]:
    """Convert strings of arrays with values to arrays of strings of values.
    Each row in de dataframe or series corresponds to a column, represented by a string of a list.
    Each string-list will be converted to a list with string values.

    Parameters
    ----------
    data
        Data to convert column from.
    labels
        Labels of each row corresponding to semantic column type.
    data_column_name
        Name of column of the data to convert.
    labels_column_name
        Name of column with the labels to convert.

    Returns
    -------
    converted_data
        Series with all rows a list of string values.
    converted_labels
        List with labels.
    """
    tqdm.pandas()

    literal_eval = partial(literal_eval_as_str, none_value="")

    if isinstance(data, pd.DataFrame):
        if data_column_name is None:
            raise ValueError("Missing column name of data.")
        converted_data = data[data_column_name].progress_apply(literal_eval)
    elif isinstance(data, pd.Series):
        converted_data = data.progress_apply(literal_eval)
    else:
        raise TypeError("Unexpected data type of samples.")

    if isinstance(labels, pd.DataFrame):
        if labels_column_name is None:
            raise ValueError("Missing column name of labels.")
        converted_labels = labels[labels_column_name].to_list()
    elif isinstance(labels, pd.Series):
        converted_labels = labels.to_list()
    else:
        raise TypeError("Unexpected data type of labels.")
    print("types")
    print(type(converted_data))
    print(type(converted_labels))
    return converted_data, converted_labels


def load_parquet_values(path):
    pf = ParquetFile(source=path)
    row_df = pf.read_row_group(0)

    return row_df["values"]


def extract_features(output_filename, data: Union[pd.DataFrame, pd.Series]):
    """Extract features from raw data.

    Parameters
    ----------
    output_filename
        filename to output featurized column samples
    data
        A pandas DataFrame or Series with each row as a list of string values.
    """
    vec_dim = 400
    reuse_model = True
    verify_keys = False

    first_keys = None

    reset_first()

    with open(output_filename, "w") as outfile:
        for raw_sample in tqdm(data, desc="Extracting Features"):
            n_samples = 1000
            n_values = len(raw_sample)

            if n_samples > n_values:
                n_samples = n_values

            random.seed(13)
            raw_sample = random.sample(raw_sample, k=n_samples)

            features = OrderedDict()

            extract_bag_of_characters_features(raw_sample, features)
            extract_word_embeddings_features(raw_sample, features)
            extract_bag_of_words_features(raw_sample, features, n_samples)

            # TODO use data_no_null version?
            infer_paragraph_embeddings_features(
                raw_sample, features, vec_dim, reuse_model
            )

            if first_keys is None:
                first_keys = features.keys()
                print(f"Exporting {len(first_keys)} column features")

                first_keys_str = keys_to_csv(features.keys())

                outfile.write(first_keys_str + "\n")

                set_first()
            elif verify_keys:
                keys = ",".join(features.keys())
                if first_keys_str != keys:
                    key_list = list(features.keys())

                    print(
                        f"keys are NOT equal. k1 len={len(first_keys)}, k2 len={len(keys)}"
                    )

                    for idx, k1 in enumerate(first_keys):
                        k2 = key_list[idx]

                        if k1 != k2:
                            print(f"{k1} != {k2}")

            outfile.write(",".join(map(str, features.values())) + "\n")
