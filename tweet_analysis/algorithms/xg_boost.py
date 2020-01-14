import xgboost as xgb
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class XGBoost:
    def __init__(self):
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('xgboost', xgb.XGBClassifier(objective='multi:softmax'))
            ]
        )
