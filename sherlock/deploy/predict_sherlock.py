import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from sherlock.deploy import model_helpers


def _transform_predictions_to_classes(y_pred, nn_id) -> np.array:
    """Get predicted semantic types from prediction vectors.
    
    Parameters
    ----------
    y_pred
        Nested vector with for each sample a vector of likelihoods per semantic type.
    nn_id
        Identifier of model to use.
        
    Returns
    -------
    y_pred
        Predicted semantic labels.
    """
    y_pred_int = np.argmax(y_pred, axis=1)
    encoder = LabelEncoder()
    encoder.classes_ = np.load(
        f"../sherlock/deploy/classes_{nn_id}.npy",
        allow_pickle=True
    )
    y_pred = encoder.inverse_transform(y_pred_int)

    return y_pred


def predict_sherlock(X: pd.DataFrame, nn_id: str) -> np.array:
    """Use sherlock model to generate predictions for X.
    
    Parameters
    ----------
    X
        Featurized data set to generate predictions for.
    nn_id
        Identifier of a trained model to use for generating predictions.
        
    Returns
    -------
    Array with predictions for X.
    """
    sherlock_model, _ = model_helpers.construct_sherlock_model(nn_id, with_weights=True)
    feature_cols_dict = model_helpers.categorize_features()
    y_pred = sherlock_model.predict(
        [
            X[feature_cols_dict['char']].values,
            X[feature_cols_dict['word']].values,
            X[feature_cols_dict['par']].values,
            X[feature_cols_dict['rest']].values
        ]
    )
    
    return _transform_predictions_to_classes(y_pred, nn_id)
