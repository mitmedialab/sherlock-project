# Sherlock: data and deployment scripts.

Sherlock is a deep-learning approach to semantic data type detection which is important for, among others, data cleaning and schema matching. This repository provides data and scripts to guide the deployment of Sherlock.

##### Material to be added: code of model and experiments.

## Project Organization
     
    ├── docs   <- Files for https://sherlock.media.mit.edu landing page.
     
    ├── data   <- Placeholder directory to download data into.
     
    ├── notebooks   <- Notebooks demonstrating the deployment of Sherlock using this repository.
            └── retrain_sherlock.ipynb
     
    ├── src                
        ├── deploy  <- Scripts to (re)train models on new data and generate predictions.
            └── classes_sherlock.npy
            └── predict_sherlock.py
            └── train_sherlock.py
        ├── features     <- Scripts to turn raw data, storing raw data columns, into features.
            ├── feature_column_identifiers   <- directory to hold feature names categorized by feature set.
               └── char_col.tsv
               └── par_col.tsv
               └── rest_col.tsv
               └── word_col.tsv
            └── bag_of_characters.py
            └── bag_of_words.py
            └── build_features.py
            └── par_vec_trained_400.pkl
            └── paragraph_vectors.py
            └── word_embeddings.py
        ├── models  <- Trained models.
            ├── sherlock_model.json
            └── sherlock_weights.h5
    
    └── requirements.txt <- Project dependencies.

------------
