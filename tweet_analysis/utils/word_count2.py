import multiprocessing
from pathlib import Path
from nltk import download
from nltk.corpus import words
import time

def process_token(token):
    if token in words.words():
        return token

if __name__ == "__main__":
    download('words')

    report = Path.cwd().joinpath("docs","report.md")
    
    my_words = []
    tokens = []
    with report.open() as read_file:
        tokens = read_file.read().split()

    print("Unprocessed tokens:\t{}".format(len(tokens)))
    start = time.time()
    with multiprocessing.Pool() as pool:
        my_words = pool.map(process_token, tokens)
    print("Time 1: {}s".format(time.time()-start))
    my_words = [word for word in my_words if word]
    print("Valid words:\t{}".format(len(my_words)))