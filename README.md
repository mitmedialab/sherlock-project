# Sherlock: data and deployment scripts.

Sherlock is a deep-learning approach to semantic data type detection which is helpful for, among others, data validation, processing and integration. This repository provides data and code to guide usage of Sherlock and replication of results.


### Installation of package
You can install Sherlock by cloning this repository, and run `pip install .`.


### Demonstration of usage
The notebooks in `notebooks/` prefixed with `01-data processing.ipynb` and `02-1-train-and-test-sherlock.ipynb` can be used to reproduce the results, and demonstrate the usage of Sherlock (from data preprocessing to model training and evaluation). The `00-WIP-use-sherlock-out-of-the-box.ipynb` notebook demonstrates usage of the readily trained model for a given table (WIP).


### Data
The raw data (corresponding to annotated table columns) can be downloaded using the `download_data()` function in the `helpers` module.
This will download 3.6GB of data into the `data` directory. Use the `01-data-preprocessing.ipynb` notebook to preprocess this data.


### Making predictions for new dataset
To use the readily trained model for generating predictions for a new dataset, features can be extracted using the `features.preprocessing` module. With the resulting feature vectors, Sherlock can be used in a scikit-learn fashion (`fit`, `predict`, `predict_proba`). Training and evaluation is demonstrated in the `02-1-train-and-test-sherlock.ipynb` notebook.


### Retraining Sherlock
The notebook `02-1-train-and-test-sherlock.ipynb` illustrates how Sherlock, as constructed from the `SherlockModel` or loaded from a json file, can be retrained. The model will infer the number of unique classes from the training labels unless you load a model from a json file, the number of classes will be 78 in that case.


## Project Organization
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
