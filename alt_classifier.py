# Make Predictions with Naive Bayes On The Iris Dataset
import json
import os
import re
from csv import reader
from math import exp, pi, sqrt

from nltk import WordPunctTokenizer

from lukifier import Lukifier

DIR_PATH = os.path.dirname(__file__)
TWEET_PATH = os.path.join(DIR_PATH, "tweets.json")
labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
word_features = []
word_labels = []
tweet_features = []
tweet_labels = []


def get_tweets(filename=TWEET_PATH):
    with open(filename, "r", encoding="utf8") as read_file:
        tweets = json.load(read_file)
    new_tweets = []
    for tweet in tweets:
        if tweet:
            new_tweets.append(tweet["text"])
    return new_tweets


def clean_tweet(tweet):
    user_removed = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet = number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet


def get_clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        clean_tweets.append(clean_tweet(tweet))
    return clean_tweets


def get_labelled_words(tweets):
    tweet_count = 0
    total_tweets = 1000
    for tweet in tweets[:total_tweets]:
        for word in tweet.split():
            word = word.strip().lower()
            sen_class = Lukifier(word).classification
            word_features.append(word)
            word_labels.append(sen_class)
        tweet_count += 1
        if tweet_count % 10 == 0:
            print("Processed {}/{} tweets for words".format(tweet_count, total_tweets))


def get_labelled_tweets(tweets):
    tweet_count = 0
    total_tweets = 1000
    for tweet in tweets[:total_tweets]:
        sen_class = Lukifier(tweet).classification
        tweet_features.append(tweet)
        tweet_labels.append(sen_class)
        tweet_count += 1
        if tweet_count % 10 == 0:
            print("Processed {}/{} tweets".format(tweet_count, total_tweets))


def get_word_sentiment_prob(word):
    word = clean_tweet(word)


print("Getting tweets")
#tweets = get_tweets()
#tweets = get_clean_tweets(tweets)
print("Got tweets")
print("Preparing dataset")
get_labelled_tweets(tweets)
get_labelled_words(tweets)
print("Prepared dataset")
