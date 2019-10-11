import nltk
import json
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re


def get_tweets(filename="tweets.json"):
    with open(filename, "r") as read_file:
        tweets = json.load(read_file)
    return tweets


def print_tweets(tweets):
    for tweet in tweets:
        print(tweet["text"])


def get_words(tweets):
    words = []
    for tweet in tweets:
        for word in tweet["text"].split():
            words.append(word.lower())
    return words


def get_phrases(words, length=2):
    phrases = []
    for index in range(len(words)):
        if index > 0 and index < len(words)-1:
            phrase = words[index-(length//2):index+(length-(length//2))]
        elif index >= 0:
            phrase = words[index-length:index]
        else:
            phrase = words[index:index+length]
        phrase = " ".join(phrase)
        phrases.append(phrase)
    return(phrases)


tweets = get_tweets()
words = get_words(tweets)
phrases = get_phrases(words, length=5)
words = nltk.FreqDist(words)
# print(words.most_common(50))
print(nltk.FreqDist(phrases).most_common(50))
