import multiprocessing
import re
import shutil
import time
import warnings
from itertools import product
from pathlib import Path
from random import randint
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.exceptions
from sklearn.model_selection import cross_validate

from algorithms.algorithm import Algorithm
from algorithms.logistic_regression import LogisticRegression
from algorithms.multilayer_perceptron import MultilayerPerceptron
from algorithms.naive_bayes import NaiveBayes
from algorithms.random_forest import RandomForest
from algorithms.stochastic_gradient_descent import StochasticGD
from algorithms.xg_boost import XGBoost
from utils.tweet_getter import TweetGetter

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.UndefinedMetricWarning)


np.set_printoptions(precision=2)

TEMPLATE = """
Accuracy (Mean): {}
Accuracy (std): {}
Precision (Mean): {}
Precision (std): {}
Recall (Mean): {}
Recall (std): {}
Time (s) (mean): {}
Time (s) (std): {}
Tweets: {}

"""

COLOURS = ['r', 'g', 'b', 'c', 'y', 'm']

def save_graph_individual(algorithms, size):
    save_file = Path("graphs/size{}.svg".format(size))
    for algorithm in algorithms:
        plt.plot(algorithm.accuracy, algorithm.time,
                 "{}o".format(algorithm.colour),
                 label="{} (Accuracy)".format(algorithm.name))
        plt.plot(algorithm.precision, algorithm.time,
                 "{}v".format(algorithm.colour),
                 label="{} (Precision)".format(algorithm.name))
        plt.plot(algorithm.recall, algorithm.time,
                 "{}s".format(algorithm.colour),
                 label="{} (Recall)".format(algorithm.name))

    plt.xlabel("Score")
    plt.ylabel("Time (s)")
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.savefig(str(save_file),
                dpi=300,
                format='svg',
                bbox_extra_artists=(lg,),
                bbox_inches='tight')
    plt.clf()

def get_colour(colours, name):
    for colour, value in colours.items():
        if value == name:
            return colour
        elif not value:
            colours[colour] = name
            return colour

def read_dict(f):
    results = []
    colours = {colour:"" for colour in COLOURS}
    length_dict = dict()
    with f.open() as read_file:
        lines = read_file.read()
        results = lines.split("Name")
    results = [result.splitlines() for result in results]
    results = [result for result in results if len(result) > 1]
    for result in results:
        length = -1
        std_time = -1
        std_acc = -1
        std_pre = -1
        std_acc = -1
        mean_time = -1
        mean_acc = -1
        mean_pre = -1
        mean_rec = -1
        algorithms = []
        for line in result:
            if re.match(r"\|[a-zA-Z]", line):
                data = line.split("|")
                data = [d for d in data if d]
                a = Algorithm(data)
                a.colour = get_colour(colours, a.name)
                algorithms.append(a)
            elif "Accuracy" in line:
                if "mean" in line:
                    mean_acc = line.split()[-1]
                elif "std" in line:
                    std_acc = line.split()[-1]
            elif "Precision" in line:
                if "mean" in line:
                    mean_pre = line.split()[-1]
                elif "std" in line:
                    std_pre = line.split()[-1]
            elif "Recall" in line:
                if "mean" in line:
                    mean_rec = line.split()[-1]
                elif "std" in line:
                    std_rec = line.split()[-1]
            elif "Time" in line:
                if "mean" in line:
                    mean_time = line.split()[-1]
                elif "std" in line:
                    std_time = line.split()[-1]
            elif "Tweets" in line:
                length = line.split()[-1]
        length_dict[length] = {
            "std_acc": std_acc, "mean_acc": mean_acc, "std_pre": std_pre,
            "mean_pre": mean_pre, "std_rec": std_rec, "mean_rec": mean_rec,
            "std_time": std_time, "mean_time": mean_time, "algorithms": algorithms
        }

    algorithms = []
    for length, value in length_dict.items():
        save_graph_individual(value["algorithms"], length)
        if not algorithms:
            algorithms = value["algorithms"]
            for result in algorithms:
                result.time = [result.time]
                result.accuracy = [result.accuracy]
                result.precision = [result.precision]
                result.recall = [result.recall]
        else:
            for i in range(len(algorithms)):
                algorithms[i].time.append(value["algorithms"][i].time)
                algorithms[i].accuracy.append(value["algorithms"][i].accuracy)
                algorithms[i].precision.append(
                    value["algorithms"][i].precision)
                algorithms[i].recall.append(value["algorithms"][i].recall)

    plt.autoscale(True)
    for algorithm in algorithms:
        print(algorithm)
        plt.plot(algorithm.accuracy, algorithm.time, "{}o".format(
            algorithm.colour), label="{} (Accuracy)".format(algorithm.name))
        plt.plot(algorithm.precision, algorithm.time, "{}v".format(
            algorithm.colour), label="{} (Precision)".format(algorithm.name))
        plt.plot(algorithm.recall, algorithm.time, "{}s".format(
            algorithm.colour), label="{} (Recall)".format(algorithm.name))

    plt.xlabel("Score")
    plt.ylabel("Time (s)")
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    save_file = Path("graphs/combined.svg")
    plt.savefig(str(save_file),
                dpi=300,
                format='svg',
                bbox_extra_artists=(lg,),
                bbox_inches='tight')


def save_dict(d, size, save_path):
    lines = [
        "| Name                        | Accuracy | Precision | Recall | Time (s) |\n",
        "| --------------------------- | -------- | --------- | ------ | -------- |\n"
    ]
    for i in range(len(d["Name"])):
        lines.append(
            "|{}|{}|{}|{}|{}|\n".format(
                d["Name"][i],
                format(d["Accuracy"][i], '.3g'),
                format(d["Precision"][i], '.3g'),
                format(d["Recall"][i], '.3g'),
                format(d["Time (s)"][i], '.3g')
            )
        )
    lines.append(
        TEMPLATE.format(
            format(d["Accuracy (Mean)"], '.3g'),
            format(d["Accuracy (std)"], '.3g'),
            format(d["Precision (Mean)"], '.3g'),
            format(d["Precision (std)"], '.3g'),
            format(d["Recall (Mean)"], '.3g'),
            format(d["Recall (std)"], '.3g'),
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


def get_slice(X, y, length):
    size = len(X)
    if length == size:
        return X, y

    X_slice = []
    y_slice = []

    while len(X_slice) != length:
        start = randint(0, size)
        if (start + length) > size:
            start = (start + length) % size
        end = start + length
        X_slice = X[start:end]
        y_slice = y[start:end]

    return X_slice, y_slice


def progressive_train(X, y, save_path):
    results = []

    orig_size = len(X)
    sizes = [i*0.125 for i in range(1, 9)]
    lengths = [int(size*orig_size) for size in sizes]

    print("Training on tweets of sizes: {}".format(lengths))
    print("Original size is: {}".format(orig_size))

    for length in lengths:
        X_train, y_train = get_slice(X, y, length)
        print(length, len(X), len(X_train), len(y_train))
        if length in lengths:
            algorithms = train(X_train, y_train, save_path, length)
            if algorithms:
                if not results:
                    results = algorithms
                    for result in results:
                        result.time = [result.time]
                        result.accuracy = [result.accuracy]
                        result.precision = [result.precision]
                        result.recall = [result.recall]
                else:
                    for i in range(len(results)):
                        results[i].time.append(algorithms[i].time)
                        results[i].accuracy.append(algorithms[i].accuracy)
                        results[i].precision.append(algorithms[i].precision)
                        results[i].recall.append(algorithms[i].recall)

        time_string = time.strftime("%H:%M:%S", time.localtime())
        print("Training for size {} finished at {}".format(length, time_string))

    plt.autoscale(True)
    for algorithm in algorithms:
        print(algorithm)
        plt.plot(algorithm.accuracy, algorithm.time, "{}o".format(
            algorithm.colour), label="{} (Accuracy)".format(algorithm.name))
        plt.plot(algorithm.precision, algorithm.time, "{}v".format(
            algorithm.colour), label="{} (Precision)".format(algorithm.name))
        plt.plot(algorithm.recall, algorithm.time, "{}s".format(
            algorithm.colour), label="{} (Recall)".format(algorithm.name))

    plt.xlabel("Score")
    plt.ylabel("Time (s)")
    lg = plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    save_file = Path("graphs/combined.svg")
    plt.savefig(str(save_file),
                dpi=300,
                format='svg',
                bbox_extra_artists=(lg,),
                bbox_inches='tight')


def train(X, y, save_path, length):
    size = len(X)
    if size != length:
        return

    save_file = save_path.joinpath("size{}.svg".format(size))
    print("Searching for results at {}".format(save_file))
    if save_file.exists():
        return
    print("Training on {} tweets".format(size))

    algorithms = [XGBoost(), LogisticRegression(), RandomForest(),
                  MultilayerPerceptron(), NaiveBayes(), StochasticGD()]

    start = timer()
    args = [(algorithm, X, y) for algorithm in algorithms]
    with multiprocessing.Pool() as pool_inner:
        algorithms = pool_inner.starmap(evaluate, args)
    print("Total training time for size {}: {}s".format(size, timer() - start))

    algorithms.sort(key=lambda x: x.accuracy, reverse=True)

    accuracies = [algorithm.accuracy for algorithm in algorithms]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    precisions = [algorithm.precision for algorithm in algorithms]
    mean_pre = np.mean(precisions)
    std_pre = np.std(precisions)

    recalls = [algorithm.recall for algorithm in algorithms]
    mean_rec = np.mean(recalls)
    std_rec = np.std(recalls)

    times = [algorithm.time for algorithm in algorithms]
    mean_time = np.mean(times)
    std_time = np.std(times)

    data = {
        "Name": [algorithm.name for algorithm in algorithms],
        "Accuracy": accuracies,
        "Accuracy (Mean)": mean_acc,
        "Accuracy (std)": std_acc,
        "Precision": precisions,
        "Precision (Mean)": mean_pre,
        "Precision (std)": std_pre,
        "Recall": recalls,
        "Recall (Mean)": mean_rec,
        "Recall (std)": std_rec,
        "Time (s)": times,
        "Time (s) (mean)": mean_time,
        "Time (s) (std)": std_time
    }

    save_dict(data, size, save_path)

    save_graph_individual(algorithms, size)

    return algorithms


def avg(scores):
    return np.average(scores)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def run_algorithm(function):
    results = function()
    scores = (avg(results['test_accuracy']), avg(
        results['test_precision_weighted']), avg(results['test_recall_weighted']))
    return scores, avg(results['score_time'])


def evaluate(algorithm, X, y):
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted']
    function = wrapper(
        cross_validate, algorithm.pipeline, X, y, cv=10, scoring=scoring)
    scores, time = run_algorithm(function)
    algorithm.accuracy, algorithm.precision, algorithm.recall = scores
    algorithm.time = time
    return algorithm


if __name__ == "__main__":
    # restart = True
    # save_path = Path("graphs")
    # if restart:
    #     try:
    #         shutil.rmtree(str(save_path.resolve()))
    #         save_path.mkdir()
    #         Path("results.txt").unlink()
    #     except:
    #         print("Failed to clear files from previous run")
    # else:
    #     save_path.mkdir()

    # tweets = TweetGetter().get_standardised_tweets()
    # X = tweets.tweet
    # y = tweets.polarity

    # start = timer()
    # progressive_train(X, y, save_path)
    # print("Total training time for all sizes: {}s".format(timer()-start))
    read_dict(Path("results.txt"))
