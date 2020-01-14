from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from utils.tf_idf import TfidfTransformer


class LatentSA:
    def __init__(self):
        latent_model = TruncatedSVD(n_components=500,
                                    algorithm='randomized',
                                    n_iter=10, random_state=42)
        self.pipeline = Pipeline(
            steps=[
                ('tfidf', TfidfTransformer()),
                ('latent_sa', latent_model())
            ]
        )
