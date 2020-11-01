import pandas as pd
import random

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# Input: a collection of columns stored in a dataframe column 'values'
# Output: tagged columns.
# Only needed for training.
def tagcol_paragraph_embeddings_features(train_data):
    
    # Expects a dataframe with a 'values' column
    train_data_values = train_data['values']
    random.seed(13)
    columns = [TaggedDocument(random.sample(col, min(1000, len(col))), [i]) for i, col in enumerate(train_data_values.values)]
    
    return columns


# Input: returned tagged document collection from tagcol_paragraph_embeddings_features
# Output: a stored retrained model
# Only needed for training.
def train_paragraph_embeddings_features(columns, dim):

    # Train Doc2Vec model
    model = Doc2Vec(columns, dm=0, negative=3, workers=8, vector_size=dim, epochs=20, min_count=2, seed=13)

    # Save trained model
    model_file = '../sherlock/features/par_vec_retrained_{}.pkl'.format(dim)
    model.save(model_file)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    

# Input: a single column in the form of a pandas Series.
# Output: ordered dictionary holding paragraph vector features
def infer_paragraph_embeddings_features(data, dim):

    # Load pretrained paragraph vector model
    model = Doc2Vec.load('../sherlock/features/par_vec_trained_{}.pkl'.format(dim))

    f = pd.DataFrame()

    if len(data) > 1000:
        random.seed(13)
        vec = random.sample(data, 1000)
    else:
        vec = data

    # Infer paragraph vector for data sample
    f = f.append(pd.Series(model.infer_vector(vec, steps=20,
                                              alpha=0.025)), ignore_index=True)

    col_names = []
    for i, col in enumerate(f):
        col_names.append('par_vec_{}'.format(i))
        
    f.columns = col_names

    return f
