from sklearn.linear_model import LogisticRegression as lp
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class LogisticRegression:
    def __init__(self):
        self.name = "Logistic Regression"
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('log_reg', lp(
                    multi_class='multinomial', solver='saga', max_iter=100))
            ]
        )
