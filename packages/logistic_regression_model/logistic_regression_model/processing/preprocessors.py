import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalImputer(BaseEstimator, TransformerMixin):
    """Categorical data missing value imputer."""

    def __init__(self, variables=None, replace_with='Missing') -> None:
        self.replace_with = replace_with
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "CategoricalImputer":
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for var in self.variables:   
            X[var] = X[var].str.strip()    
            X[var] = np.where(X[var]=='', self.replace_with, X[var])
            X[var] = np.where(X[var]=='?', self.replace_with, X[var])
            X[var] = X[var].fillna(self.replace_with)

        return X