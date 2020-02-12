from timeit import default_timer as timer

import numpy as np
from sklearn.model_selection import cross_val_score

from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.random_forest import RandomForest
from algorithms.xg_boost import XGBoost
from algorithms.naive_bayes import NaiveBayes
from algorithms.stochastic_gradient_descent import StochasticGD
from algorithms.latent_sa import LatentSA
from utils.tweet_getter import TweetGetter


def avg(scores):
    return np.average(scores)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def run_algorithm(function):
    start = timer()
    accuracy = function()
    time = timer() - start
    return accuracy, time


def evaluate(algorithm):
    function = wrapper(
        cross_val_score, algorithm.pipeline, X, y, cv=5, scoring='accuracy')
    accuracy, time = run_algorithm(function)
    algorithm.accuracy = accuracy
    algorithm.time = time


tweets = TweetGetter().get_clean_tweets_with_scores()
X = tweets.tweet
y = tweets.polarity

algorithms = [XGBoost(), LogisticRegression(), RandomForest(),
              MultilayerPerceptron(), NaiveBayes(), StochasticGD()]

for algorithm in algorithms:
    evaluate(algorithm)

algorithms = sorted(algorithms, key=lambda algorithm: algorithm.accuracy, reverse=True)

for algorithm in algorithms:
    print("Name:\t{}\tAccuracy:\t{}\tTime:\t{}".format(
        algorithm.name, algorithm.accuracy, algorithm.time))
