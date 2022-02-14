# Sherlock: code, data, and trained model.

Sherlock is a deep-learning approach to semantic data type detection, i.e. labeling tables with column types such as `name`, `address`, etc. This is helpful for, among others, data validation, processing and integration. This repository provides data and code to guide usage of Sherlock, retraining the model, and replication of results. Visit https://sherlock.media.mit.edu for more background on this project.


## Installation of package
You can install Sherlock by cloning this repository, and run `pip install .`.


## Demonstration of usage
The notebooks in `notebooks/` prefixed with `01-data processing.ipynb` and `02-1-train-and-test-sherlock.ipynb` can be used to reproduce the results, and demonstrate the usage of Sherlock (from data preprocessing to model training and evaluation). The `00-WIP-use-sherlock-out-of-the-box.ipynb` notebook demonstrates usage of the readily trained model for a given table (WIP).


## Data
The raw data (corresponding to annotated table columns) can be downloaded using the `download_data()` function in the `helpers` module.
This will download 3.6GB of data into the `data` directory. Use the `01-data-preprocessing.ipynb` notebook to preprocess this data. Each column is then represented by a feature vector of dimensions 1x1588. The extracted features per column are based on "paragraph" embeddings (full column), word embeddings (aggregated from each column cell), character count statistics (e.g. average number of "." in a column's cells) and column-level statistics (e.g. column entropy).


## The Sherlock model
The `SherlockModel` class is specified in the `sherlock.deploy.model` module. This model constitutes a multi-input neural network which specifies a separate network for each feature set (e.g. the word embedding features), concatenates them, and finally adds a few shared layers.


## Making predictions
To use the readily trained Sherlock model for generating predictions for a dataset, features can be extracted using the `features.preprocessing` module. With the resulting feature vectors, Sherlock can be used in a scikit-learn fashion (`fit`, `predict`, `predict_proba`). Training and evaluation is demonstrated in the `02-1-train-and-test-sherlock.ipynb` notebook.


## Retraining Sherlock
The notebook `02-1-train-and-test-sherlock.ipynb` illustrates how Sherlock, as constructed from the `SherlockModel` or loaded from a json file, can be retrained. The model will infer the number of unique classes from the training labels unless you load a model from a json file, the number of classes will be 78 in that case.


## Project structure
    ├── data   <- Placeholder directory to download data into.

    ├── docs   <- Files for https://sherlock.media.mit.edu landing page.

    ├── model_files  <- Files with trained model weights and specification.
        ├── sherlock_model.json
        └── sherlock_weights.h5
        
    ├── notebooks   <- Notebooks demonstrating data preprocessing and train/test of Sherlock.
        └── 00-WIP-use-sherlock-out-of-the-box.ipynb
        └── 01-data-preprocessing.ipynb
        └── 02-1-train-and-test-sherlock.ipynb
        └── 02-2-train-and-test-sherlock-rf-ensemble.ipynb
        └── 03-train-paragraph-vector-features-optional.ipynb
        
    ├── sherlock  <- Package.
        ├── deploy  <- Code for (re)training Sherlock, as well as model specification.
            └── helpers.py
            └── model.py
        ├── features     <- Files to turn raw data, storing raw data columns, into features.
            ├── feature_column_identifiers   <- Directory with feature names categorized by feature set.
            └── bag_of_characters.py
            └── bag_of_words.py
            └── par_vec_trained_400.pkl
            └── paragraph_vectors.py
            └── preprocessing.py
            └── word_embeddings.py
        ├── helpers.py     <- Supportive modules.

------------
