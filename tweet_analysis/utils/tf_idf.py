from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator


class TfidfTransformer(BaseEstimator):

    def __init__(self):
        self._model = TfidfVectorizer()

    def fit(self, df_x, df_y=None):
        self._model.fit(df_x)
        return self

    def transform(self, df_x):
        return self._model.transform(df_x)
