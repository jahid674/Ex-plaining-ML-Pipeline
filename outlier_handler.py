from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, method="iqr", factor=1.5, return_mask=True):
        self.method = method
        self.factor = factor
        self.return_mask = return_mask

    def fit(self, X, y=None):
        X = np.asarray(X)
        if self.method == "iqr":
            Q1 = np.nanpercentile(X, 25, axis=0)
            Q3 = np.nanpercentile(X, 75, axis=0)
            self.lower_bound_ = Q1 - self.factor * (Q3 - Q1)
            self.upper_bound_ = Q3 + self.factor * (Q3 - Q1)
        return self

    def transform(self, X):
        X = np.asarray(X)
        if self.return_mask:
            mask = (X < self.lower_bound_) | (X > self.upper_bound_)
            return mask
        else:
            return X