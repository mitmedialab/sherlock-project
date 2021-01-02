import random
import multiprocessing
import gensim.models.doc2vec
from collections import OrderedDict
import pandas as pd
import nltk
from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


def tokenise(values):
    tokens = []

    joined = ' '.join(s for s in values if len(s) >= 2).lower()
    filtered = ''.join(e for e in joined if e.isalnum() or e.isspace())

    for word in nltk.word_tokenize(filtered):
        if len(word) < 2 or word in STOPWORDS_ENGLISH:
            continue

        tokens.append(word)

    return tokens


# Input: a collection of columns stored in a dataframe column 'values'
# Output: tagged columns.
# Only needed for training.
def tagcol_paragraph_embeddings_features(train_data: pd.Series, train_labels: list):
    random.seed(13)

    columns = []

    for i, col in enumerate(train_data):
        label = train_labels[i]
        values = random.sample(col, min(1000, len(col)))

        if len(values) > 0:
            values = list(map(lambda s: '' if s is None else str(s), values))

        tokens = tokenise(values)

        columns.append(TaggedDocument(tokens, label))

    return columns


# Input: returned tagged document collection from tagcol_paragraph_embeddings_features
# Output: a stored retrained model
# Only needed for training.
def train_paragraph_embeddings_features(columns, dim):

    # Train Doc2Vec model
    model = Doc2Vec(columns, dm=0, negative=3, workers=multiprocessing.cpu_count(), vector_size=dim, epochs=20, min_count=2, seed=13)

    # Save trained model
    model_file = f'../sherlock/features/par_vec_retrained_{dim}.pkl'
    model.save(model_file)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


DIM = 400
model: Doc2Vec = None


def initialise_pretrained_model(dim):
    start = datetime.now()
    global model

    filename = f'../sherlock/features/par_vec_retrained_{dim}.pkl'

    assert dim == DIM

    model = Doc2Vec.load(filename)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print(f'Initialise Doc2Vec Model, {dim} dim, process took {datetime.now() - start} seconds. (filename = {filename})')


STOPWORDS_ENGLISH = None


def initialise_nltk():
    start = datetime.now()

    nltk.download('punkt')
    nltk.download('stopwords')

    global STOPWORDS_ENGLISH

    STOPWORDS_ENGLISH = stopwords.words('english')

    print(f'Initialised NLTK, process took {datetime.now() - start} seconds.')


KEYS = (
        [f'par_vec_{i}' for i in range(DIM)]
)


# Input: a single column in the form of a pandas Series.
# Output: ordered dictionary holding paragraph vector features
def infer_paragraph_embeddings_features(col_values: list, features: OrderedDict, dim, reuse_model):

    if not reuse_model or model is None:
        # Load pretrained paragraph vector model
        initialise_pretrained_model(dim)

    # Resetting the random seed before inference keeps the inference vectors deterministic. Gensim uses random values
    # in the inference process, so setting the seed just before hand makes the inference repeatable.
    # https://github.com/RaRe-Technologies/gensim/issues/447
    model.random.seed(13)

    tokens = tokenise(col_values)

    # Infer paragraph vector for data sample.
    inferred = model.infer_vector(tokens, steps=20, alpha=0.025)

    for idx, value in enumerate(inferred):
        features[KEYS[idx]] = value
