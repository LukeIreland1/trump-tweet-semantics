import json
import os
import pprint
import string
import time
import re
from nltk import WordPunctTokenizer
from collections import defaultdict
from lukifier import Lukifier

import numpy as np

word_features = []
word_labels = []
tweet_features = []
tweet_labels = []


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = defaultdict(int)
        self.logprior = {}
        self.bigdoc = defaultdict(list)
        self.loglikelihoods = defaultdict(defaultdict)
        self.V = []
        self.n = n_gram

    def compute_prior_and_bigdoc(self, training_set, training_labels):
        '''
        Computes the prior and the bigdoc (from the book's algorithm)
        :param training_set:
            a list of all documents of the training set
        :param training_labels:
            a list of labels corresponding to the documents in the training set
        :return:
            None
        '''
        for x, y in zip(training_set, training_labels):
            grams = x.split(" ")

            self.prior[y] += len(grams)
            if x:
                self.bigdoc[y].append(x)

    def compute_vocabulary(self, documents):
        vocabulary = set()

        for doc in documents:
            for word in doc.split(" "):
                vocabulary.add(word.lower())

        return vocabulary

    def count_word_in_classes(self):
        counts = {}
        for c in list(self.bigdoc.keys()):
            docs = self.bigdoc[c]
            counts[c] = defaultdict(int)
            for doc in docs:
                words = doc.split(" ")
                for word in words:
                    counts[c][word] += 1

        return counts

    def train(self, training_set, training_labels, alpha=1):
        # Get number of documents
        N_doc = len(training_set)

        # Get vocabulary used in training set
        self.V = self.compute_vocabulary(training_set)

        # Create bigdoc
        for x, y in zip(training_set, training_labels):
            if x:
                self.bigdoc[y].append(x)

        # Get set of all classes
        all_classes = set(training_labels)

        # Compute a dictionary with all word counts for each class
        self.word_count = self.count_word_in_classes()

        # For each class
        for c in all_classes:
            # Get number of documents for that class
            N_c = training_labels.count(c)

            # Compute logprior for class
            self.logprior[c] = np.log(N_c / N_doc)

            # Calculate the sum of counts of words in current class
            total_count = 0
            for word in self.V:
                total_count += self.word_count[c][word]

            # For every word, get the count and compute the log-likelihood for this class
            for word in self.V:
                count = self.word_count[c][word]
                self.loglikelihoods[c][word] = np.log(
                    (count + alpha) / (total_count + alpha * len(self.V)))

    def predict(self, test_doc):
        sums = {
            -1: 0,
            0: 0,
            1: 0,
        }
        for c in self.bigdoc.keys():
            sums[c] = self.logprior[c]
            words = test_doc.split(" ")
            for word in words:
                if word in self.V:
                    sums[c] += self.loglikelihoods[c][word]

        return sums


sentiment_numerical_val = {
    'NEG': -1,
    'NEUT': 0,
    'POS': 1
}

DIR_PATH = os.path.dirname(__file__)
TWEET_PATH = os.path.join(DIR_PATH, "tweets.json")


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
    words = [word for word in words if word != "amp"]
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet


def get_clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        clean_tweets.append(clean_tweet(tweet))
    return clean_tweets


def get_labelled_words(tweets, total_tweets=1000):
    tweet_count = 0
    for tweet in tweets[:total_tweets]:
        for word in tweet.split():
            word = word.strip().lower()
            sen_class = Lukifier(word).polarity
            word_features.append(word)
            word_labels.append(sen_class)
        tweet_count += 1
        if tweet_count % 10 == 0:
            print("Processed {}/{} tweets for words".format(tweet_count, total_tweets))


def get_labelled_tweets(tweets, total_tweets=1000):
    tweet_count = 0
    for tweet in tweets[:total_tweets]:
        sen_class = Lukifier(tweet).polarity
        tweet_features.append(tweet)
        tweet_labels.append(sen_class)
        tweet_count += 1
        if tweet_count % 10 == 0:
            print("Processed {}/{} tweets".format(tweet_count, total_tweets))


def evaluate_predictions(validation_set, validation_labels, trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -2

    for tweet, label in zip(validation_set, validation_labels):
        if not tweet:
            continue

        probabilities = trained_classifier.predict(tweet)
        for prob in probabilities:
            probabilities[prob] = abs(probabilities[prob])
        if (probabilities[0] - probabilities[-1])*2 < (probabilities[0] - probabilities[1]):
            prediction = 1
        if (probabilities[0] - probabilities[1])*2 < (probabilities[0] - probabilities[-1]):
            prediction = -1
        else:
            prediction = max(probabilities, key=probabilities.get)
        print("{} is the prediction for {}".format(prediction, tweet))

        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            print("Expected {} | Actual {} | Probabilities {}".format(
                label, prediction, probabilities))
            predictions_list.append("-")

    print("Predicted correctly {} out of {} ({}%)".format(correct_predictions, len(
        validation_labels), round(correct_predictions/len(validation_labels)*100, 5)))
    return predictions_list, round(correct_predictions/len(validation_labels)*100)


def split_data(features, labels):
    size = len(features)
    feature_size = round(size*0.7)
    return (
        features[:feature_size], labels[:feature_size],
        features[feature_size:], labels[feature_size:]
    )


tweets = get_clean_tweets(get_tweets())
get_labelled_tweets(tweets)
training_set, training_labels, validation_set, validation_labels = split_data(
    tweet_features, tweet_labels)


start = time.time()

NBclassifier = NaiveBayesClassifier()
NBclassifier.train(training_set, training_labels, alpha=1)
results, acc = evaluate_predictions(
    validation_set, validation_labels, NBclassifier)

end = time.time()
print('Ran in {} seconds'.format(round(end - start, 3)))
