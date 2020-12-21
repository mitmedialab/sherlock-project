from ast import literal_eval
from collections import OrderedDict
import random
import os
from typing import Union
import csv

from google_drive_downloader import GoogleDriveDownloader as gd
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel

from sherlock.features.bag_of_characters import extract_bag_of_characters_features
from sherlock.features.bag_of_words import extract_bag_of_words_features
from sherlock.features.word_embeddings import extract_word_embeddings_features
from sherlock.features.paragraph_vectors import infer_paragraph_embeddings_features


def prepare_feature_extraction():
    """Download embedding files from Google Drive if they do not exist yet."""
    word_embedding_file = '../sherlock/features/glove.6B.50d.txt'
    paragraph_vector_file = '../sherlock/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy'
    
    print(
        f"""Preparing feature extraction by downloading 2 files:
        \n {word_embedding_file} and \n {paragraph_vector_file}.
        """
    )

    if not os.path.exists(word_embedding_file):
        print('Downloading GloVe word embedding vectors.')
        file_name = word_embedding_file
        gd.download_file_from_google_drive(
            file_id='1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk',
            dest_path=file_name,
            unzip=False,
            showsize=True
        )

        print("GloVe word embedding vectors were downloaded.")

    if not os.path.exists(paragraph_vector_file):
        print("Downloading pretrained paragraph vectors.")
        file_name = paragraph_vector_file
        gd.download_file_from_google_drive(
            file_id='1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu',
            dest_path=file_name,
            unzip=False,
            showsize=True
        )
        
        print("Trained paragraph vector model was downloaded.")
        
    print("All files for extracting word and paragraph embeddings are present.")
    
    
def convert_string_lists_to_lists(
    data: Union[pd.DataFrame, pd.Series],
    labels: Union[pd.DataFrame, pd.Series],
    data_column_name: str = None,
    labels_column_name: str = None,
) -> pd.Series:
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

    pandarallel.initialize()
    
    if isinstance(data, pd.DataFrame):
        if data_column_name is None: raise ValueError("Missing column name of data.")
        converted_data = data[data_column_name].parallel_apply(literal_eval)
    elif isinstance(data, pd.Series):
        converted_data = data.progress_apply(literal_eval)
    else:
        raise TypeError("Unexpected data type of samples.")

    if isinstance(labels, pd.DataFrame):
        if labels_column_name is None: raise ValueError("Missing column name of labels.")
        converted_labels = labels[labels_column_name].to_list()
    elif isinstance(labels, pd.Series):
        converted_labels = labels.to_list()
    else:
        raise TypeError("Unexpected data type of labels.")
    
    return converted_data, converted_labels


def extract_features(output_filename, data: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """Extract features from raw data.
    
    Parameters
    ----------
    output_filename
        filename to output featurized column samples
    data
        A pandas DataFrame or Series with each row a list of string values.
    """
    n_samples = 1000
    vec_dim = 400
    reuse_model = True
    verify_keys = False

    first_keys = None

    with open(output_filename, "w") as outfile:
        csvwriter = csv.writer(outfile)

        for raw_sample in tqdm(data, desc='Extracting Features'):
            n_values = len(raw_sample)

            if n_samples > n_values:
                n_samples = n_values

            random.seed(13)
            raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

            # f_source = OrderedDict()
            # f_source['source'] = '\\;'.join(raw_sample.str.join(''))
            #
            # s = f_source['source']
            # print(f'{i} "{s}"')
            #
            # source_features = list(f_source.items())
            #

            data_no_null = raw_sample.dropna()

            characters_features = list(extract_bag_of_characters_features(data_no_null).items())
            embeddings_features = list(extract_word_embeddings_features(data_no_null).items())
            words_features = list(extract_bag_of_words_features(data_no_null, n_values).items())

            # TODO use data_no_null version?
            paragraph_features = list(infer_paragraph_embeddings_features(raw_sample, vec_dim, reuse_model).items())

            # f = OrderedDict(source_features + characters_features + embeddings_features + words_features + paragraph_features)
            f = OrderedDict(words_features + embeddings_features + characters_features + paragraph_features)

            if first_keys is None:
                first_keys = f.keys()
                first_keys_str = ','.join(f.keys())

                print(f'Exporting {len(first_keys)} column features')

                csvwriter.writerow(first_keys)
            elif verify_keys:
                keys = ','.join(f.keys())
                if first_keys_str != keys:
                    key_list = list(f.keys())

                    print(f'keys are NOT equal. k1 len={len(first_keys)}, k2 len={len(keys)}')

                    for idx, k1 in enumerate(first_keys):
                        k2 = key_list[idx]

                        if k1 != k2:
                            print(f'{k1} != {k2}')

            csvwriter.writerow(list(f.values()))
