import random
import os
import pandas as pd

from collections import OrderedDict
from google_drive_downloader import GoogleDriveDownloader as gd
from src.features.bag_of_characters import extract_bag_of_characters_features
from src.features.bag_of_words import extract_bag_of_words_features
from src.features.word_embeddings import extract_word_embeddings_features
from src.features.paragraph_vectors import infer_paragraph_embeddings_features


def _prepare_feature_extraction():
    
    word_embedding_file = '../src/features/glove.6B.50d.txt'
    paragraph_vector_file = '../src/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy'
    
    print('Preparing feature extraction by downloading 3 files: \n {} and \n {}.'.format(word_embedding_file,
                                                                                         paragraph_vector_file))

    if not os.path.exists(word_embedding_file):
        print('Downloading GloVe word embedding vectors.')
        file_name = word_embedding_file
        gd.download_file_from_google_drive(file_id='1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk',
                                           dest_path=file_name,
                                           unzip=False,
                                           showsize=True)

    print('GloVe word embedding vectors were downloaded.')

    if not os.path.exists(paragraph_vector_file):
        print('Downloading pretrained paragraph vectors.')
        file_name = paragraph_vector_file
        gd.download_file_from_google_drive(file_id='1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu',
                                           dest_path=file_name,
                                           unzip=False,
                                           showsize=True)
        
    print('Trained paragraph vector model was downloaded.')


def _get_data():

    data_dir = '../data/data.zip'
    print('Downloading the raw and preprocessed data into {}.'.format(data_dir))

    if not os.path.exists(data_dir):
        print('Downloading data directory.')
        dir_name = data_dir
        gd.download_file_from_google_drive(file_id='1-g0zbKFAXz7zKZc0Dnh74uDBpZCv4YqU',
                                           dest_path=dir_name,
                                           unzip=True,
                                           showsize=True)

    print('Data was downloaded.')


# Input: a pandas DataFrame with each row a numpy array representing a full data column
# Output: feature_vectors stored as numpy ndarray
def build_features(data):

    _get_data()
    _prepare_feature_extraction()

    features_list = []
    df_par = pd.DataFrame()
    n_samples = 1000
    vec_dim = 400
    i = 0
    for raw_sample in data:

        i = i + 1
        if i % 100 == 0:
            print('Extracting features for data column ', i)

        n_values = len(raw_sample)

        if n_samples > n_values:
            n_samples = n_values
        random.seed(13)
        raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

        f = OrderedDict(list(extract_bag_of_characters_features(raw_sample).items()) +
                        list(extract_word_embeddings_features(raw_sample).items()) +
                        list(extract_bag_of_words_features(raw_sample, n_values).items()))
        features_list.append(f)

        df_par = df_par.append(infer_paragraph_embeddings_features(raw_sample, vec_dim))

    return pd.concat([pd.DataFrame(features_list).reset_index(drop=True),
                      df_par.reset_index(drop=True)],
                     axis=1, sort=False)
