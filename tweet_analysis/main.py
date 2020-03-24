import concurrent.futures
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.naive_bayes import NaiveBayes
from algorithms.random_forest import RandomForest
from algorithms.stochastic_gradient_descent import StochasticGD
from algorithms.xg_boost import XGBoost
from utils.tweet_getter import TweetGetter


colours = {
    'b': False,
    'g': False,
    'r': False,
    'c': False,
    'm': False,
    'y': False
}


def get_colour_code():
    for colour in colours:
        if not colours[colour]:
            colours[colour] = True
            return colour


def avg(scores):
    return np.average(scores)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def run_algorithm(function):
    start = timer()
    accuracy = avg(function())
    time = timer() - start
    return accuracy, time


def evaluate(algorithm):
    function = wrapper(
        cross_val_score, algorithm.pipeline, X, y, cv=10, scoring='accuracy')
    accuracy, time = run_algorithm(function)
    algorithm.accuracy = accuracy
    algorithm.time = time


tweets = TweetGetter().get_standardised_tweets()
X = tweets.tweet
y = tweets.polarity

algorithms = [XGBoost(), LogisticRegression(), RandomForest(),
              MultilayerPerceptron(), NaiveBayes(), StochasticGD()]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = {executor.submit(evaluate, algorithm): algorithm
               for algorithm in algorithms}
    for future in concurrent.futures.as_completed(futures):
        algorithm = futures[future]
        print("Training {}".format(algorithm.name))

algorithms.sort(key=lambda x: x.accuracy, reverse=True)

data = {
    "Name": [algorithm.name for algorithm in algorithms],
    "Accuracy": [algorithm.accuracy for algorithm in algorithms],
    "Time (s)": [algorithm.time for algorithm in algorithms]
}

df = pd.DataFrame(data)
print(df)

for algorithm in algorithms:
    plt.plot(algorithm.accuracy, algorithm.time,
                "{}o".format(get_colour_code()), label=algorithm.name)
    plt.xlabel("Accuracy")
    plt.ylabel("Time (s)")
    plt.legend(loc="upper right")

plt.savefig("output.svg")

