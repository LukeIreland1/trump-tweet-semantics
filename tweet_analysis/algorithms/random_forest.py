from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class RandomForest:
    def __init__(self):
        self.name = "Random Forest"
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('random_forest', rf())
            ]
        )
        self.colour = 'r'