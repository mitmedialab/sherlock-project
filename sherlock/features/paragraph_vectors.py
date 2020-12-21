import random
import multiprocessing
import gensim.models.doc2vec
from collections import OrderedDict

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from datetime import datetime

assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"


# Input: a collection of columns stored in a dataframe column 'values'
# Output: tagged columns.
# Only needed for training.
def tagcol_paragraph_embeddings_features(train_data):
    # Expects a dataframe with a 'values' column
    train_data_values = train_data['values']

    random.seed(13)
    columns = [TaggedDocument(random.sample(col, min(1000, len(col))), [i]) for i, col in
               enumerate(train_data_values.values)]

    return columns


# Input: returned tagged document collection from tagcol_paragraph_embeddings_features
# Output: a stored retrained model
# Only needed for training.
def train_paragraph_embeddings_features(columns, dim):

    # Train Doc2Vec model
    model = Doc2Vec(columns, dm=0, negative=3, workers=multiprocessing.cpu_count(), vector_size=dim, epochs=20, min_count=2, seed=13)

    # Save trained model
    model_file = '../sherlock/features/par_vec_retrained_{}.pkl'.format(dim)
    model.save(model_file)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


model = None


def initialise_pretrained_model(dim):
    start = datetime.now()
    global model
    model = Doc2Vec.load('../sherlock/features/par_vec_trained_{}.pkl'.format(dim))
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print(f'Initialise Doc2Vec Model, {dim} dim, process took {datetime.now() - start} seconds.')


# Input: a single column in the form of a pandas Series.
# Output: ordered dictionary holding paragraph vector features
def infer_paragraph_embeddings_features(data, dim, reuse_model):
    global model

    if not reuse_model or model is None:
        # Load pretrained paragraph vector model
        initialise_pretrained_model(dim)

    if len(data) > 1000:
        random.seed(13)
        vec = random.sample(data, 1000)
    else:
        vec = data

    # Resetting the random seed before inference keeps the inference vectors deterministic. Gensim uses random values
    # in the inference process, so setting the seed just before hand makes the inference repeatable.
    # https://github.com/RaRe-Technologies/gensim/issues/447
    model.random.seed(13)

    # Infer paragraph vector for data sample
    inferred = model.infer_vector(vec, steps=20, alpha=0.025)

    f = OrderedDict()
    i = 0

    for v in inferred:
        f['par_vec_{}'.format(i)] = v
        i = i + 1

    return f
