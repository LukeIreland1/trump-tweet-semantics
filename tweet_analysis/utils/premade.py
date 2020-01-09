import nltk
import re
import os
import json
from nltk import WordPunctTokenizer
from string import punctuation
from nltk.corpus import stopwords

dir_path = os.path.dirname(__file__)
tweet_path = os.path.join(dir_path, "tweets.json")


def get_tweets(filename=tweet_path):
    with open(filename, "r", encoding="utf8") as read_file:
        tweets = json.load(read_file)
    new_tweets = []
    for tweet in tweets:
        if tweet:
            new_tweets.append(tweet["text"])
    return new_tweets


tweets = get_tweets()
testData = tweets[:100]
trainingData = tweets[100:]


class PreProcessTweets:

    def processTweets(self, list_of_tweets):
        processedTweets = []
        for tweet in list_of_tweets:
            processedTweets.append(
                (self.clean_tweet(tweet["text"]), tweet["label"]))
        return processedTweets

    def clean_tweet(self, tweet):
        user_removed = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
        number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
        lower_case_tweet = number_removed.lower()
        tok = WordPunctTokenizer()
        words = tok.tokenize(lower_case_tweet)
        clean_tweet = (' '.join(words)).strip()
        return clean_tweet


tweetProcessor = PreProcessTweets()
preprocessedTrainingSet = tweetProcessor.processTweets(trainingData)
preprocessedTestSet = tweetProcessor.processTweets(testData)


def buildVocabulary(preprocessedTrainingData):
    all_words = []

    for (words, sentiment) in preprocessedTrainingData:
        all_words.extend(words)

    wordlist = nltk.FreqDist(all_words)
    word_features = wordlist.keys()

    return word_features


def extract_features(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# Now we can extract the features and train the classifier
word_features = buildVocabulary(preprocessedTrainingSet)
trainingFeatures = nltk.classify.apply_features(
    extract_features, preprocessedTrainingSet)


NBayesClassifier = nltk.NaiveBayesClassifier.train(trainingFeatures)


NBResultLabels = [NBayesClassifier.classify(
    extract_features(tweet[0])) for tweet in preprocessedTestSet]


# get the majority vote
if NBResultLabels.count('positive') > NBResultLabels.count('negative'):
    print("Overall Positive Sentiment")
    print("Positive Sentiment Percentage = " +
          str(100*NBResultLabels.count('positive')/len(NBResultLabels)) + "%")
else:
    print("Overall Negative Sentiment")
    print("Negative Sentiment Percentage = " +
          str(100*NBResultLabels.count('negative')/len(NBResultLabels)) + "%")
