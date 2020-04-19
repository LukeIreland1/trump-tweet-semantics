import concurrent.futures
from pathlib import Path
from random import randint
from timeit import default_timer as timer
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.naive_bayes import NaiveBayes
from algorithms.random_forest import RandomForest
from algorithms.stochastic_gradient_descent import StochasticGD
from algorithms.xg_boost import XGBoost
from sklearn.model_selection import cross_val_score
from utils.tweet_getter import TweetGetter

RESTART = False
SAVE_PATH = Path("graphs")
if SAVE_PATH.exists():
    if RESTART:
        shutil.rmtree(SAVE_PATH)
else:
    SAVE_PATH.mkdir()


COLOURS = {
    'b': False,
    'g': False,
    'r': False,
    'c': False,
    'm': False,
    'y': False
}


def get_colour_code():
    global COLOURS
    for colour in COLOURS:
        if not COLOURS[colour]:
            COLOURS[colour] = True
            return colour
    COLOURS = {key: False for key in COLOURS.keys()}
    return get_colour_code()


def get_range(size, orig_size):
    length = int(size*orig_size)
    start = randint(0, orig_size)
    if (start + length) > orig_size:
        start = (start + length) % orig_size
    end = start + length
    return start, end


def progressive_train(X, y):
    orig_size = len(X)
    sizes = [i*0.125 for i in range(1, 9)]
    for size in sizes:
        start, end = get_range(size, orig_size)
        print("Training on {} tweets".format(end-start))
        train(X[start:end], y[start:end])


def train(X, y):
    save_file = SAVE_PATH.joinpath("size{}.svg".format(len(X)))
    if save_file.exists():
        return

    algorithms = [XGBoost(), LogisticRegression(), RandomForest(),
                  MultilayerPerceptron(), NaiveBayes(), StochasticGD()]

    start = timer()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(evaluate, algorithm): algorithm
                   for algorithm in algorithms}
        for future in concurrent.futures.as_completed(futures):
            algorithm = futures[future]
            print("Training {}".format(algorithm.name))
    print("Total training time: {}s".format(timer() - start))

    algorithms.sort(key=lambda x: x.accuracy, reverse=True)

    accuracies = [algorithm.accuracy for algorithm in algorithms]
    times = [algorithm.time for algorithm in algorithms]
    data = {
        "Name": [algorithm.name for algorithm in algorithms],
        "Accuracy": accuracies,
        "Time (s)": [algorithm.time for algorithm in algorithms]
    }

    df = pd.DataFrame(data)
    print(df)
    print("Accuracy (mean):\t{}\nAccuracy (std):\t{}".format(
        np.mean(accuracies), np.std(accuracies)))
    print("Time (mean):\t{}\nTime (std):\t{}".format(
        np.mean(times), np.std(times)))

    for algorithm in algorithms:
        plt.plot(algorithm.accuracy, algorithm.time,
                 "{}o".format(get_colour_code()), label=algorithm.name)
        plt.xlabel("Accuracy")
        plt.ylabel("Time (s)")
        plt.legend()

    plt.savefig(save_file)
    plt.clf()


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

progressive_train(X, y)
