from timeit import default_timer as timer

import numpy as np
from sklearn.model_selection import cross_val_score

from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.random_forest import RandomForest
from algorithms.xg_boost import XGBoost
from utils.tweet_getter import TweetGetter


def avg(scores):
    return np.average(scores)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def time(function):
    start = timer()
    score = function()
    time = timer() - start
    return score, time


tweets = TweetGetter().get_clean_tweets_with_scores()
X = tweets.tweet
y = tweets.polarity

xg_boost = XGBoost()
logistic_regression = LogisticRegression()
random_forest = RandomForest()
multilayer_perceptron = MultilayerPerceptron()

xg_boost_func = wrapper(
    cross_val_score, xg_boost.pipeline, X, y, cv=5, scoring='accuracy')
logistic_regression_func = wrapper(
    cross_val_score, logistic_regression.pipeline, X, y, cv=5, scoring='accuracy')
random_forest_func = wrapper(
    cross_val_score, random_forest.pipeline, X, y, cv=5, scoring='accuracy')
multilayer_perceptron_func = wrapper(
    cross_val_score, multilayer_perceptron.pipeline, X, y, cv=5, scoring='accuracy')

xg_boost_score, xg_boost_time = time(xg_boost_func)
logistic_regression_score, logistic_regression_time = time(
    logistic_regression_func)
random_forest_score, random_forest_time = time(random_forest_func)
multilayer_perceptron_score, multilayer_perceptron_time = time(
    multilayer_perceptron_func)

times = dict()
times["XG Boost"] = xg_boost_time
times["Logistic Regression"] = logistic_regression_time
times["Random Forest"] = random_forest_time
times["Multilayer Perceptron"] = multilayer_perceptron_time
times = {k: v for k, v in sorted(times.items(), key=lambda item: item[1])}

scores = dict()
scores["XG Boost"] = avg(xg_boost_score)
scores["Logistic Regression"] = avg(logistic_regression_score)
scores["Random Forest"] = avg(random_forest_score)
scores["Multilayer Perceptron"] = avg(multilayer_perceptron_score)
scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

for name, score in scores.items():
    print("Name:\t{}\Accuracy:\t{}\tTime:\t{}s".format(name, score, times[name]))
