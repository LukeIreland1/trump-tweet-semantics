import nltk
import json
import random
import pickle
import re
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from statistics import mode
from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
from textblob import TextBlob


def get_tweets(filename="tweets.json"):
    with open(filename, "r") as read_file:
        tweets = json.load(read_file)
    new_tweets = []
    for tweet in tweets:
        if tweet:
            new_tweets.append(tweet["text"])
    return new_tweets


def print_tweets(tweets):
    for tweet in tweets:
        print(tweet)


def get_words(tweets):
    words = []
    for tweet in tweets:
        for word in tweet.split():
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
# print(nltk.FreqDist(phrases).most_common(50))


def clean_tweet(tweet):
    user_removed = re.sub(r'@[A-Za-z0-9]+', '', tweet)
    link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet = number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet


def get_sentiment_score(tweet):
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text
    analysis = TextBlob(clean_tweet(tweet))
    # set sentiment
    return analysis.sentiment.polarity


score = 0


def analyze_tweets(tweets):
    score = 0
    for tweet in tweets:
        cleaned_tweet = clean_tweet(tweet)
        sentiment_score = get_sentiment_score(cleaned_tweet)
        score += sentiment_score
        if cleaned_tweet:
            print('Tweet: {}'.format(cleaned_tweet))
            print('Score: {}\n'.format(sentiment_score))
        final_score = round((score / float(len(tweets))), 2)
    return final_score


final_score = analyze_tweets(tweets)

if final_score <= -0.25:
    status = 'NEGATIVE ❌'
elif final_score <= 0.25:
    status = 'NEUTRAL ?'
else:
    status = 'POSITIVE ✅'
