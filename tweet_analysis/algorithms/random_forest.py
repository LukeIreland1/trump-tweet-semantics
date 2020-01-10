import json
import os
import pprint
import re
import string
import time
from collections import defaultdict

import numpy as np
from nltk import WordPunctTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from lukifier import Lukifier


DIR_PATH = os.path.dirname(__file__)
TWEET_PATH = os.path.join(DIR_PATH, "tweets.json")

tweet_features = []

class RandomForest:
    def __init__(self):
        self.features = []
        self.labels = []

    def clean_tweets(self):
        tweets = []
        stemmer = WordNetLemmatizer()

        for tweet in range(0, len(self.features)):
            # Remove all the special characters
            tweet = re.sub(r'\W', ' ', str(self.features[tweet]))
            # remove all single characters
            tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', tweet)
            # Remove single characters from the start
            tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', tweet)
            # Substituting multiple spaces with single space
            tweet = re.sub(r'\s+', ' ', tweet, flags=re.I)
            # Removing prefixed 'b'
            tweet = re.sub(r'^b\s+', '', tweet)
            # Converting to Lowercase
            tweet = tweet.lower()
            # Lemmatization
            tweet = tweet.split()
            tweet = [stemmer.lemmatize(word) for word in tweet]
            tweet = ' '.join(tweet)
            tweets.append(tweet)
        self.features = tweets

    def text_to_numerical_features(self):
        vectorizer = CountVectorizer(
            max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
        self.X = vectorizer.fit_transform(self.features).toarray()

    def tfidf_transform(self):
        tfidfconverter = TfidfTransformer()
        self.X = tfidfconverter.fit_transform(self.features).toarray()

    def split_data(self):
        self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(
            self.features, self.labels, test_size=0.2, random_state=0)

    def preprocess(self):
        self.clean_tweets()
        self.text_to_numerical_features()
        self.tfidf_transform()


def get_tweets(filename=TWEET_PATH):
    with open(filename, "r", encoding="utf8") as read_file:
        tweets = json.load(read_file)
    new_tweets = []
    for tweet in tweets:
        if tweet:
            new_tweets.append(tweet["text"])
    return new_tweets

def get_labelled_tweets(tweets, total_tweets=1000):
    tweet_count = 0
    features = []
    labels = []
    for tweet in tweets[:total_tweets]:
        sen_class = Lukifier(tweet).polarity
        features.append(tweet)
        labels.append(sen_class)
        tweet_count += 1
        if tweet_count % 10 == 0:
            print("Processed {}/{} tweets".format(tweet_count, total_tweets))
    return features, labels

features, labels = get_labelled_tweets(get_tweets)
rf = RandomForest()
rf.features = features
rf.labels = labels
rf.preprocess()
rf.train()
rf.predict()

