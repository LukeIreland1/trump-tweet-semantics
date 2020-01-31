from sklearn.model_selection import cross_val_score

from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.random_forest import RandomForest
from algorithms.xg_boost import XGBoost
from utils.tweet_getter import TweetGetter

import numpy as np

def avg(scores):
    return np.average(scores)

tweets = TweetGetter().get_clean_tweets_with_scores()
X = tweets.tweet
y = tweets.polarity

xg_boost = XGBoost()
logistic_regression = LogisticRegression()
random_forest = RandomForest()
multilayer_perceptron = MultilayerPerceptron()

xg_boost_score = cross_val_score(
    xg_boost.pipeline, X, y, cv=5, scoring='accuracy')
logistic_regression_score = scores = cross_val_score(
    logistic_regression.pipeline, X, y, cv=5, scoring='accuracy')
random_forest_score = cross_val_score(
    random_forest.pipeline, X, y, cv=5, scoring='accuracy')
multilayer_perceptron_score = cross_val_score(
    multilayer_perceptron.pipeline, X, y, cv=5, scoring='accuracy')

scores = dict()
scores["XG Boost"] = avg(xg_boost_score)
scores["Logistic Regression"] = avg(logistic_regression_score)
scores["Random Forest"] = avg(random_forest_score)
scores["Multilayer Perceptron"] = avg(multilayer_perceptron_score)
scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

for name, score in scores.items():
    print("Name:\t{}\tScore:\t{}".format(name, avg(score)))
