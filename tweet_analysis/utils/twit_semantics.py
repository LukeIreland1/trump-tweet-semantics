import json
import nltk
import re
import os
import markovify
import matplotlib.pyplot as plt
from nltk import WordPunctTokenizer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

DIR_PATH = os.path.dirname(__file__)
TWEET_PATH = os.path.join(DIR_PATH, "tweets.json")

# TODO: Incoroporate new custom semantic/classifer from lukifier

def get_tweets(filename=TWEET_PATH):
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


# Needs adding to a function or class
tweets = get_tweets()
words = get_words(tweets)
phrases = get_phrases(words, length=4)
words = nltk.FreqDist(words)
print(words.most_common(50))
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


def get_clean_tweets(tweets):
    clean_tweets = []
    for tweet in tweets:
        clean_tweets.append(clean_tweet(tweet))
    return clean_tweets

def display_cloud():
    wordcloud = PhraseCloud(width=800, height=800,
                            background_color='white',
                            stopwords=STOPWORDS,
                            min_font_size=10).generate(phrases)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()


def get_word_freq(word):
    return words[word]


def make_tweets(tweets):
    # Build the model.
    text_model = markovify.Text(tweets)

    # Print five randomly-generated sentences
    for i in range(5):
        print(text_model.make_short_sentence(140))


def make_tweets_from_word(tweets, word):
    results = [tweet for tweet in tweets if word.lower() in tweet.lower()]
    # Build the model.
    text_model = TrumpModel(results, word)

    # Print five randomly-generated sentences
    for i in range(5):
        print(text_model.make_short_sentence())


class TrumpModel(markovify.Text):
    def __init__(self, results, word):
        self.word = word
        super().__init__(results)

    def make_short_sentence(self):
        sentence = ""
        while True:
            sentence = self.make_sentence()
            if sentence:
                if len(sentence) > 0 and len(sentence) < 140:
                    if word in sentence:
                        return sentence


class TweetAnalyser():
    def __init__(self):
        self.score = 0
        self.pos_tweets = []
        self.pos_count = 0
        self.neg_tweets = []
        self.neg_count = 0
        self.neut_tweets = []
        self.neut_count = 0

    def get_sentiment_score(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(clean_tweet(tweet))
        # set sentiment
        return analysis.sentiment.polarity

    def analyze_tweets(self, tweets):
        for tweet in tweets:
            cleaned_tweet = clean_tweet(tweet)
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


class PhraseCloud(WordCloud):
    def process_text(self, phrases):
        phrases = get_clean_tweets(phrases)
        return nltk.FreqDist(phrases)

def run_analysis():
    analyser = TweetAnalyser()
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
# display_cloud()
word = "obama"
count = words[word]
print("Trump said {} {} times".format(word, count))
# make_tweets(tweets)
make_tweets_from_word(tweets, word)
