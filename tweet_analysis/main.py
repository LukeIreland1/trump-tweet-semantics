from sklearn.model_selection import cross_val_score

from utils.tweet_getter import TweetGetter
from algorithms.xg_boost import XGBoost
from algorithms.logistic_regression import LogisticRegression
from algorithms.random_forest import RandomForest
from algorithms.multilayer_perceptron import MultilayerPerceptron

tweets = TweetGetter().get_clean_tweets_with_scores()
X = tweets.tweet
y = tweets.polarity

xg_boost = XGBoost()
logistic_regression = LogisticRegression()
random_forest = RandomForest()
multilayer_perceptron = MultilayerPerceptron()
xg_boost_score = scores = cross_val_score(xg_boost.pipeline, X, y, cv=5,scoring='accuracy')
logistic_regression_score = scores = cross_val_score(logistic_regression.pipeline, X, y, cv=5,scoring='accuracy')
random_forest_score = scores = cross_val_score(random_forest.pipeline, X, y, cv=5,scoring='accuracy')
multilayer_perceptron_score = cross_val_score(multilayer_perceptron.pipeline, X, y, cv=5,scoring='accuracy')