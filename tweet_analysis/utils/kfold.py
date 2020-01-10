from numpy import array
from sklearn.model_selection import KFold as kf
from tweet_analysis.utils.tweet_getter import TweetGetter

class KFold():
    def __init__(self):
        tweets = TweetGetter().get_clean_tweets_with_scores()
        kfold = kf(2, True, 1)
        X = tweets.tweet
        y = tweets.polarity

        for train_index, test_index in kfold.split(tweets):
            self.X_train, self.X_test = X[train_index], X[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]