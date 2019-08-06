import random
import os
import pandas as pd

from google_drive_downloader import GoogleDriveDownloader as gd
from src.features.bag_of_characters import extract_bag_of_characters_features
from src.features.bag_of_words import extract_bag_of_words_features
from src.features.word_embeddings import extract_word_embeddings_features
from src.features.paragraph_vectors import infer_paragraph_embeddings_features


def prepare_feature_extraction():
    
    word_embedding_file = '../src/features/glove.6B.50d.txt'
    paragraph_vector_file = '../src/features/par_vec_trained_400.pkl.docvecs.vectors_docs.npy'
    
    print('Preparing feature extraction by downloading 2 files: \n {} and \n {}.'.format(word_embedding_file, 
                                                                                         paragraph_vector_file))

    if not os.path.exists(word_embedding_file):
        print('Downloading GloVe word embedding vectors.')
        file_name = word_embedding_file
        gd.download_file_from_google_drive(file_id='1kayd5oNRQm8-NCvA8pIrtezbQ-B1_Vmk',
                                           dest_path=file_name,
                                           unzip=True,
                                           showsize=True)

    print('GloVe word embedding vectors were downloaded.')
    
    if not os.path.exists(paragraph_vector_file):
        print('Downloading pretrained paragraph vectors.')
        file_name = paragraph_vector_file
        gd.download_file_from_google_drive(file_id='1vdyGJ4aB71FCaNqJKYX387eVufcH4SAu',
                                           dest_path=file_name,
                                           unzip=True,
                                           showsize=True)
        
    print('Trained paragraph vector model was downloaded.')


# Input: a pandas DataFrame with each row a numpy array representing a full data column
# Output: feature_vectors stored as numpy ndarray
def build_features(data):

    prepare_feature_extraction()

    df_char = pd.DataFrame()
    df_word = pd.DataFrame()
    df_par = pd.DataFrame()
    df_stat = pd.DataFrame()

    n_samples = 1000
    vec_dim = 400
    i = 0
    for raw_sample in data.iterrows():

        i = i + 1
        if i % 20 == 0:
            print('Extracting features for data column ', i)

        n_values = len(raw_sample)

        if n_samples > n_values:
            n_samples = n_values

        # Sample n_samples from data column, and convert cell values to string values
        raw_sample = pd.Series(random.choices(raw_sample, k=n_samples)).astype(str)

        df_char = df_char.append(extract_bag_of_characters_features(raw_sample, n_values), ignore_index=True)
        df_word = df_word.append(extract_word_embeddings_features(raw_sample), ignore_index=True)
        df_par = df_par.append(infer_paragraph_embeddings_features(raw_sample, vec_dim), ignore_index=True)
        df_stat = df_stat.append(extract_bag_of_words_features(raw_sample), ignore_index=True)

    df_char.fillna(df_char.mean(), inplace=True)
    df_word.fillna(df_word.mean(), inplace=True)
    df_par.fillna(df_par.mean(), inplace=True)
    df_stat.fillna(df_stat.mean(), inplace=True)

    feature_vectors = [df_char.values, df_word.values, df_par.values, df_stat.values]

    return feature_vectors
