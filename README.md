# Sherlock: code, data, and trained model.

Sherlock is a deep-learning approach to semantic data type detection, i.e. labeling tables with column types such as `name`, `address`, etc. This is helpful for, among others, data validation, processing and integration. This repository provides data and code to guide usage of Sherlock, retraining the model, and replication of results. Visit https://sherlock.media.mit.edu for more background on this project.

## Installation of package
1. You can install Sherlock by cloning this repository, and run `pip install .`.
2. Install dependencies using `pip install -r requirements.txt` (or `requirements38.txt` depending on your Python version).

## Demonstration of usage
The `00-use-sherlock-out-of-the-box.ipynb` notebook demonstrates usage of the readily trained model for a given table.

The notebooks in `notebooks/` prefixed with `01-data processing.ipynb` and `02-1-train-and-test-sherlock.ipynb` can be used to reproduce the results, and demonstrate the usage of Sherlock (from data preprocessing to model training and evaluation).

## Data
The raw data (corresponding to annotated table columns) can be downloaded using the `download_data()` function in the `helpers` module.
This will download +/- 500MB of data into the `data` directory. Use the `01-data-preprocessing.ipynb` notebook to preprocess this data. Each column is then represented by a feature vector of dimensions 1x1588. The extracted features per column are based on "paragraph" embeddings (full column), word embeddings (aggregated from each column cell), character count statistics (e.g. average number of "." in a column's cells) and column-level statistics (e.g. column entropy).

## The Sherlock model
The `SherlockModel` class is specified in the `sherlock.deploy.model` module. This model constitutes a multi-input neural network which specifies a separate network for each feature set (e.g. the word embedding features), concatenates them, and finally adds a few shared layers. Interaction with the model follows the scikit-learn interface, with methods `fit`, `predict` and `predict_proba`.

## Making predictions
The originally trained `SherlockModel` can be used for generating predictions for a dataset. First, extract features using the `features.preprocessing` module. The original weights of Sherlock are provided in the repository in the `model_files` directory and can be loaded using the `initialize_model_from_json` method of the model. The procedure for making predictions (on the data) is demonstrated in the `02-1-train-and-test-sherlock.ipynb` notebook.


## Retraining Sherlock
The notebook `02-1-train-and-test-sherlock.ipynb` also illustrates how Sherlock can be retrained. The model will infer the number of unique classes from the training labels unless you load a model from a json file, the number of classes will be 78 in that case.


## Citing this work

To cite this work, please use the below bibtex:

```
@inproceedings{Hulsebos:2019:SDL:3292500.3330993,
 author = {Hulsebos, Madelon and Hu, Kevin and Bakker, Michiel and Zgraggen, Emanuel and Satyanarayan, Arvind and Kraska, Tim and Demiralp, {\c{C}}a{\u{g}}atay and Hidalgo, C{\'e}sar},
 title = {Sherlock: A Deep Learning Approach to Semantic Data Type Detection},
 booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \&\#38; Data Mining},
 year={2019},
 publisher = {ACM},
}
```

## Project structure
    ├── data   <- Placeholder directory to download data into.

    ├── docs   <- Files for https://sherlock.media.mit.edu landing page.

    ├── model_files  <- Files with trained model weights and specification.
        ├── sherlock_model.json
        └── sherlock_weights.h5

    ├── notebooks   <- Notebooks demonstrating data preprocessing and train/test of Sherlock.
        └── 00-use-sherlock-out-of-the-box.ipynb
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

---------
