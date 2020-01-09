import numpy as np
import pandas as pd
from tweet_analysis.algorithms.utils import tweets
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

tfidf = TfidfVectorizer()
tfidf.fit(tweets['text'])
X = tfidf.transform(tweets['text'])

tweets.dropna(inplace=True)
cols = ['text','score','polarity']
tweets.drop(cols, axis=1, inplace=True)

X = tweets.text
y = tweets.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(
    "Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(
        len(X_train), (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
        (len(X_train[y_train == 1]) / (len(X_train)*1.))*100)
)


def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000, 30001, 10000)


def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
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


tfidf = TfidfVectorizer()
print("Result for trigram with stop words (Tfidf)\n")
feature_result_tgt = nfeature_accuracy_checker(
    vectorizer=tfidf, ngram_range=(1, 3))

cv = CountVectorizer(max_features=30000, ngram_range=(1, 3))
pipeline = Pipeline([
    ('vectorizer', cv),
    ('classifier', rf)
])
sentiment_fit = pipeline.fit(X_train, y_train)
y_pred = sentiment_fit.predict(X_test)
print(classification_report(y_test, y_pred,
                            target_names=['negative', 'positive']))
