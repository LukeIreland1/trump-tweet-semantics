import multiprocessing
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
from itertools import product

np.set_printoptions(precision=2)

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


def progressive_train(X, y, save_path):
    orig_size = len(X)
    sizes = [i*0.125 for i in range(1, 9)]
    for size in sizes:
        start, end = get_range(size, orig_size)
        train(X[start:end], y[start:end], save_path)


def train(X, y, save_path):
    save_file = save_path.joinpath("size{}.svg".format(len(X)))
    print("Searching for results at {}".format(save_file))
    if save_file.exists():
        return
    print("Training on {} tweets".format(len(X)))

    algorithms = [XGBoost(), LogisticRegression(), RandomForest(),
                  MultilayerPerceptron(), NaiveBayes(), StochasticGD()]

    start = timer()
    args = [(algorithm, X, y) for algorithm in algorithms]
    with multiprocessing.Pool() as pool:
        algorithms = pool.starmap(evaluate, args)
    print("Total training time for size {}: {}s".format(len(X), timer() - start))

    algorithms.sort(key=lambda x: x.accuracy, reverse=True)

    accuracies = [algorithm.accuracy for algorithm in algorithms]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    times = [algorithm.time for algorithm in algorithms]
    mean_time = np.mean(times)
    std_time = np.std(times)

    data = {
        "Name": [algorithm.name for algorithm in algorithms],
        "Accuracy": accuracies,
        "Accuracy (Mean)": mean_acc,
        "Accuracy (std)": std_acc,
        "Time (s)": times,
        "Time (s) (mean)": mean_time,
        "Time (s) (std)": std_time
    }

    df = pd.DataFrame(data)
    print(df)

    for algorithm in algorithms:
        plt.plot(algorithm.accuracy, algorithm.time,
                 "{}o".format(get_colour_code()), label=algorithm.name)
        plt.xlabel("Accuracy")
        plt.ylabel("Time (s)")
        plt.legend()

    print("Saved results to {}".format(str(save_file)))
    plt.savefig(str(save_file))
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


def evaluate(algorithm, X, y):
    function = wrapper(
        cross_val_score, algorithm.pipeline, X, y, cv=10, scoring='accuracy')
    accuracy, time = run_algorithm(function)
    algorithm.accuracy = accuracy
    algorithm.time = time
    return algorithm


if __name__ == "__main__":
    restart = False
    save_path = Path("graphs")
    if save_path.exists():
        if restart:
            try:
                shutil.rmtree(str(save_path.resolve()))
            except:
                print("Could not delete {}".format(save_path))
    else:
        save_path.mkdir()

    tweets = TweetGetter().get_standardised_tweets()
    X = tweets.tweet
    y = tweets.polarity

    start = timer()
    progressive_train(X, y, save_path)
    print("Total training time for all sizes: {}s".format(timer()-start))
