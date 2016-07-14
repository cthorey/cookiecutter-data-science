'''

Store general pipeline that can be reused in many different experiments


'''


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA
import xgboost as xgb
import pandas as pd
import os
import sys


class FeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, cols=[]):
        self.cols = cols

    def transform(self, X, **transform_params):
        tmp = X[self.cols]
        return tmp

    def fit(self, X, y=None, **fit_params):
        return self


class Inputer(TransformerMixin, BaseEstimator):
    '''
    Custom imputer function. Use fillna method from pandas instead
    of scikit learn.
    '''

    def __init__(self, fill=0, method=0):
        self.fill = fill
        self.method = method

    def transform(self, X, **transform_params):
        if (self.fill != 0) and (self.method == 0):
            return pd.DataFrame(X).fillna(value=self.fill)
        elif (self.fill == 0) and (self.method != 0):
            return pd.DataFrame(X).fillna(method=self.method)
        elif (self.fill == 0) and (self.method == 0):
            return pd.DataFrame(X).fillna(value=self.fill)
        else:
            raise ValueError('Error in the inputer args.')

    def fit(self, X, y=None, **fit_params):
        return self
