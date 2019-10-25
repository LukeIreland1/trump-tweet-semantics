import json
import nltk
import re
import os
from nltk import WordPunctTokenizer
from textblob import TextBlob

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

class tweet_analyser():
    def __init__(self):
        self.score = 0
        self.pos_tweets = []
        self.pos_count = 0
        self.neg_tweets = []
        self.neg_count = 0
        self.neut_tweets = []
        self.neut_count = 0

    def clean_tweet(self, tweet):
        user_removed = re.sub(r'@[A-Za-z0-9]+', '', tweet)
        link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
        number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
        lower_case_tweet = number_removed.lower()
        tok = WordPunctTokenizer()
        words = tok.tokenize(lower_case_tweet)
        clean_tweet = (' '.join(words)).strip()
        return clean_tweet

    def get_sentiment_score(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        return analysis.sentiment.polarity

    def analyze_tweets(self, tweets):
        for tweet in tweets:
            cleaned_tweet = self.clean_tweet(tweet)
            if cleaned_tweet:
                sentiment_score = self.get_sentiment_score(cleaned_tweet)
                if sentiment_score <= -0.25:
                    self.neg_count += 1
                    self.neg_tweets.append(tweet)
                elif sentiment_score <= 0.25:
                    self.neut_count += 1
                    self.neut_tweets.append(tweet)
                else:
                    self.pos_count += 1
                    self.pos_tweets.append(tweet)
                self.score += sentiment_score
                print('Tweet: {}'.format(cleaned_tweet))
                print('Score: {}\n'.format(sentiment_score))
        final_score = round((self.score / float(len(tweets))), 2)
        return final_score

    def sentiment_split(self):
        total = self.neg_count + self.pos_count + self.neut_count
        neg_split = format(self.neg_count*100/total, ".3g")
        pos_split = format(self.pos_count*100/total, ".3g")
        neut_split = format(self.neut_count*100/total, ".3g")
        return neg_split, neut_split, pos_split

def run_analysis():
    analyser = tweet_analyser()
    final_score = analyser.analyze_tweets(tweets[:100])

    if final_score <= -0.25:
        status = 'NEGATIVE ❌'
    elif final_score <= 0.25:
        status = 'NEUTRAL ?'
    else:
        status = 'POSITIVE ✅'

    print(final_score, status)
    neg, neut, pos = analyser.sentiment_split()
    print(
        "{}% negative\n{}% neutral\n{}% positive".format(
            neg, neut, pos
        )
    )


run_analysis()
