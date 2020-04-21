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

TEMPLATE = """
Accuracy (Mean): {}
Accuracy (std): {}
Time (s) (mean): {}
Time (s) (std): {}
Tweets: {}

"""


def save_dict(d, size, save_path):
    lines = [
        "| Name                        | Accuracy | Time (s)   |",
        "| --------------------------- | -------- | ---------- |"
    ]
    for i in range(len(d["Name"])):
        lines.append(
            "|{}|{}|{}|\n".format(
                d["Name"][i],
                format(d["Accuracy"][i], '.3g'),
                format(d["Time (s)"][i], '.3g')
            )
        )
    lines.append(
        TEMPLATE.format(
            format(d["Accuracy (Mean)"], '.3g'),
            format(d["Accuracy (std)"], '.3g'),
            format(d["Time (s) (mean)"], '.3g'),
            format(d["Time (s) (std)"], '.3g'),
            size
        )
    )
    save_file = Path("results.txt")
    if save_file.exists():
        with save_file.open('a') as write_file:
            write_file.writelines(lines)
    else:
        with save_file.open('w') as write_file:
            write_file.writelines(lines)
    print("Added results to {}".format(str(save_file)))


def get_range(length, orig_size):
    start = randint(0, orig_size)
    if (start + length) > orig_size:
        start = (start + length) % orig_size
    end = start + length
    return start, end


def progressive_train(X, y, save_path):
    results = dict()
    colours = dict()

    orig_size = len(X)
    sizes = [i*0.125 for i in range(1, 9)]
    lengths = [int(size*orig_size) for size in sizes]

    print("Training on tweets of sizes: {}".format(lengths))
    print("Original size is: {}".format(orig_size))

    for length in lengths:
        start, end = get_range(length, orig_size)
        results[length] = train(X[start:end], y[start:end], save_path)

    algorithm_results = {
        algorithm.name: {"Accuracy": [], "Time": []}
        for algorithm in algorithms for _, algorithms in results
    }
    for length, algorithms in results:
        for algorithm in algorithms:
            algorithm_results[algorithm.name]["Accuracy"].append(
                algorithm.accuracy)
            algorithm_results[algorithm.name]["Time"].append(algorithm.time)
            colours[algorithm.name] = algorithm.colour

    for algorithm in algorithm_results:
        plt.plot(algorithm["Accuracy"], algorithm["Time"], colours[algorithm["Name"]], label=algorithm["Name"])
        plt.xlabel("Accuracy")
        plt.ylabel("Time (s)")
        plt.legend()

    save_file = save_path.joinpath("combined.svg")
    plt.savefig(str(save_file))

def train(X, y, save_path):
    size = len(X)
    save_file = save_path.joinpath("size{}.svg".format(size))
    print("Searching for results at {}".format(save_file))
    if save_file.exists():
        return
    print("Training on {} tweets".format(size))

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

    save_dict(data, size, save_path)

    for algorithm in algorithms:
        plt.plot(algorithm.accuracy, algorithm.time,
                 "{}o".format(algorithm.colour), label=algorithm.name)
        plt.xlabel("Accuracy")
        plt.ylabel("Time (s)")
        plt.legend()

    print("Saved graph to {}".format(str(save_file)))
    plt.savefig(str(save_file))
    plt.clf()

    return algorithms


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
    restart = True
    save_path = Path("graphs")
    if save_path.exists():
        if restart:
            try:
                shutil.rmtree(str(save_path.resolve()))
                save_path.mkdir()
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
