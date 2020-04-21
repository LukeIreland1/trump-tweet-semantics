from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class NaiveBayes:
    def __init__(self):
        self.name = "Naive Bayes"
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('xgboost', MultinomialNB())
            ]
        )
        self.colour = 'm'
