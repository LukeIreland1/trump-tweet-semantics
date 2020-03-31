import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from nltk import WordPunctTokenizer

from . import root
from .lukifier import Lukifier


class TweetGetter:
    def __init__(self):
        self.tweets = []
        self.cleaned_tweets = []
        self.tweets_with_scores = []
        self.tweet_file = root / "resources" / "tweets.json"

    def get_tweets(self, filename):
        print("Getting tweets")
        with filename.open() as read_file:
            tweets = json.load(read_file)
        new_tweets = []
        for tweet in tweets:
            if tweet:
                new_tweets.append(tweet["text"])
        print("Got {} tweets".format(len(new_tweets)))
        self.tweets = new_tweets
        return new_tweets

    def clean_tweet(self, tweet):
        user_removed = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
        number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
        lower_case_tweet = number_removed.lower()
        tok = WordPunctTokenizer()
        words = tok.tokenize(lower_case_tweet)
        clean_tweet = (' '.join(words)).strip()
        return clean_tweet

    def clean_tweets(self, tweets):
        cleaned_tweets = []
        for tweet in tweets:
            cleaned_tweets.append(self.clean_tweet(tweet))
        return cleaned_tweets

    def get_clean_tweets(self):
        if not self.cleaned_tweets:
            print("Getting clean tweets")
            if not self.tweets:
                self.get_tweets(self.tweet_file)
            self.cleaned_tweets = self.clean_tweets(self.tweets)
        print("Got {} clean tweets".format(len(self.cleaned_tweets)))
        return self.cleaned_tweets

    def get_clean_tweets_with_scores(self):
        if not self.tweets_with_scores:
            print("Getting scores")
            if not self.cleaned_tweets:
                self.cleaned_tweets = self.get_clean_tweets()
            data = {
                'tweet': [],
                'score': [],
                'polarity': []
            }
            for tweet in self.cleaned_tweets:
                classifier = Lukifier(tweet)
                data['tweet'].append(tweet)
                data['score'].append(classifier.score)
                data['polarity'].append(classifier.polarity)
            self.tweets_with_scores = pd.DataFrame(data)
        print("Got {} scores".format(len(self.cleaned_tweets)))
        return self.tweets_with_scores

    def standardise(self, x, std, mean):
        return (x-mean)/std

    def classify(self, sentiment):
        if sentiment >= 0.125:
            return 1

        elif sentiment <= -0.125:
            return -1

        return 0

    def get_standardised_tweets(self):
        if not self.tweets_with_scores:
            self.get_clean_tweets_with_scores()
        std = self.tweets_with_scores.score.std()
        mean = self.tweets_with_scores.score.mean()
        self.tweets_with_scores.score = [self.standardise(
            x, std, mean) for x in self.tweets_with_scores.score]
        self.tweets_with_scores.polarity = [
            self.classify(x) for x in self.tweets_with_scores.score]
        return self.tweets_with_scores
