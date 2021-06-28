from typing import List

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.pipeline import Pipeline

from sherlock import defaults
from sherlock.features import preprocessing


class SherlockTransformer(TransformerMixin):
    def __init__(self):
        self.feature_dict = defaults.DEFAULT_FEATURES_DICT

    def transform(self, X: pd.DataFrame):
        X = preprocessing.extract_features(X).astype("float32")
        return [X[cols].values for cols in self.feature_dict.values()]


class SherlockModel(ClassifierMixin):
    def __init__(self):
        self.encoder = defaults.DEFAULT_ENCODER
        self.model = defaults.DEFAULT_MODEL

    def predict_proba(self, X: List[pd.DataFrame]) -> np.array:
        return self.model.predict(X)

    def predict_log_proba(self, X: List[pd.DataFrame]) -> np.array:
        return np.log(self.model.predict(X))

    def predict(self, X: List[pd.DataFrame]) -> np.array:
        y_pred = self.predict_proba(X)
        y_pred_int = np.argmax(y_pred, axis=1)
        return self.encoder.inverse_transform(y_pred_int)

    def fit(self, X: List[pd.DataFrame], y: pd.Series) -> np.array:
        self.model.fit(X, y)


class SherlockPipeline(Pipeline):
    def __init__(self):
        steps = [("transformer", SherlockTransformer()), ("model", SherlockModel())]
        super().__init__(steps)

    def named_proba(self, X: pd.DataFrame, top_n=None):
        y_pred = self.predict_proba(X)
        result = dict()
        for i, col in enumerate(X.columns):
            temp = sorted(
                zip(self.steps[-1].encoder.classes_, y_pred[i]),
                key=lambda item: item[1],
                reverse=True,
            )
            result[col] = temp[0:top_n] if top_n else temp
        return result
