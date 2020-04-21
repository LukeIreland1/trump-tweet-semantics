from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class StochasticGD:
    def __init__(self):
        self.name = "Stochastic Gradient Descent"
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('sgd', SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                        max_iter=5, tol=None))
            ]
        )
        self.colour = 'y'
