import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tweet_analysis.utils.tweet_getter import TweetGetter
from tweet_analysis.utils.kfold import KFold


class LatentSA:
    def __init__(self, kfold=True):
        if kfold:
            kf = KFold()
            X_train = kf.X_train
            X_test = kf.X_test
            y_train = kf.y_train
            y_test = kf.y_test
        else:
            tweets = TweetGetter().get_clean_tweets_with_scores()
            X = tweets.tweet
            y = tweets.polarity
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=0)
        print(
            "Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(
                len(X_train), (len(
                    X_train[y_train == 0]) / (len(X_train)*1.))*100,
                (len(X_train[y_train == 1]) / (len(X_train)*1.))*100)
        )

    def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
        sentiment_fit = pipeline.fit(X_train, y_train)
        y_pred = sentiment_fit.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy score: {0:.2f}%".format(accuracy*100))
        return accuracy

    def train(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
        result = []
        for n in n_features:
            vectorizer.set_params(stop_words=stop_words,
                                  max_features=n, ngram_range=ngram_range)
            checker_pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', classifier)
            ])
            print("Test result for {} features".format(n))
            nfeature_accuracy = accuracy_summary(
                checker_pipeline, X_train, y_train, X_test, y_test)
            result.append((n, nfeature_accuracy))
        return result

    def get_accuracy_report():
        cv = CountVectorizer()
        rf = RandomForestClassifier(class_weight="balanced")
        n_features = np.arange(10000, 30001, 10000)

        tfidf = TfidfVectorizer()
        print("Result for trigram with stop words (Tfidf)\n")
        feature_result_tgt = train(
            vectorizer=tfidf, ngram_range=(1, 3))

    def get_classification_report():
        cv = CountVectorizer(max_features=30000, ngram_range=(1, 3))
        pipeline = Pipeline([
            ('vectorizer', cv),
            ('classifier', rf)
        ])
        sentiment_fit = pipeline.fit(X_train, y_train)
        y_pred = sentiment_fit.predict(X_test)
        print(classification_report(y_test, y_pred,
                                    target_names=['negative', 'neutral', 'positive']))
