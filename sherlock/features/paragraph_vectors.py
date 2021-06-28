import random
from collections import OrderedDict
from functools import cache
from typing import Union

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sherlock import make_data_path


@cache
def get_paragraph_vector_model():
    par_vec_file = make_data_path(f"par_vec_trained_{400}.pkl")
    paragraph_vector_model = Doc2Vec.load(par_vec_file)
    return paragraph_vector_model


# Input: a collection of columns stored in a dataframe column 'values'
# Output: tagged columns.
# Only needed for training.
def tagcol_paragraph_embeddings_features(train_data):
    # Expects a dataframe with a 'values' column
    train_data_values = train_data["values"]
    random.seed(13)
    columns = [
        TaggedDocument(random.sample(col, min(1000, len(col))), [i])
        for i, col in enumerate(train_data_values.values)
    ]

    return columns


# Input: returned tagged document collection from tagcol_paragraph_embeddings_features
# Output: a stored retrained model
# Only needed for training.
def train_paragraph_embeddings_features(columns, dim):
    # Train Doc2Vec model
    model = Doc2Vec(
        columns,
        dm=0,
        negative=3,
        workers=8,
        vector_size=dim,
        epochs=20,
        min_count=2,
        seed=13,
    )

    # Save trained model
    model_file = make_data_path(f"par_vec_retrained_{dim}.pkl")
    model.save(model_file)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)


# Input: a single column in the form of a pandas Series.
# Output: ordered dictionary holding paragraph vector features
def infer_paragraph_embeddings_features(
    data: Union[np.array, pd.Series, list]
) -> OrderedDict:
    # pandas and numpy
    if not isinstance(data, list):
        data = data.tolist()

    model = get_paragraph_vector_model()
    embedding = model.infer_vector(data, steps=20, alpha=0.025)

    res = OrderedDict()
    for i, v in enumerate(embedding):
        res[f"par_vec_{i}"] = v
    return res
