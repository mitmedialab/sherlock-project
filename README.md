# Sherlock: data and deployment scripts.

Sherlock is a deep-learning approach to semantic data type detection which is important for, among others, data cleaning and schema matching. This repository provides data and scripts to guide the deployment of Sherlock.

##### More details about this repository follow. 

## Project Organization

    ├── data
        ├── processed      <- Examples of preprocessed data sets (feature vectors and labels).
        └── raw            <- Raw data example corresponding to preprocessed data.
     
    ├── notebooks          <- Notebooks demonstrating the deployment of Sherlock using this repository.
            └── run_sherlock.py
     
    ├── src                <- Source code for working with this project.
        ├── deploy         <- Scripts to (re)train models on new data, and generate predictions.
            └── classes_sherlock.npy
            └── predict_sherlock.py
            └── train_sherlock.py
        ├── features       <- Scripts to turn raw data, storing raw data columns, into features.
            └── bag_of_characters.py
            └── bag_of_words.py
            └── build_features.py
            └── par_vec_trained_400.pkl
            └── paragraph_vectors.py
            └── word_embeddings.py
        ├── models         <- Trained models.
            ├── sherlock_model.json
            └── sherlock_weights.h5
            
    ├── LICENSE
    
    ├── TO COME.txt       <- File describing expected repository contents.
    
    └── requirements.txt  <- Dependencies for reproducing the work, and using the provided scripts.

------------
