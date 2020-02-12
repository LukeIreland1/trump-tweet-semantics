from sklearn.neural_network import MLPClassifier as mlp
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class MultilayerPerceptron:
    def __init__(self):
        self.name = "Multilayer Perceptron"
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('xgboost', mlp(hidden_layer_sizes=(10, 10, 10), max_iter=1000))
            ]
        )
