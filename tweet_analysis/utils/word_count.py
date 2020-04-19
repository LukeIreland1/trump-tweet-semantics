import asyncio
import multiprocessing
from pathlib import Path
from nltk import download
from nltk.corpus import words
import time

download('words')

report = Path.cwd().joinpath("docs","report.md")

tokens = []
with report.open() as read_file:
    tokens = read_file.read().split()

my_words = []

async def main():
    async def process_token(token):
        if token in words.words():
            my_words.append(token)

    coros = [process_token(token) for token in tokens]
    await asyncio.gather(*coros)

print("Unprocessed tokens:\t{}".format(len(tokens)))
start = time.time()
asyncio.run(main())
print("Time 1: {}s".format(time.time()-start))
print("Valid words:\t{}".format(len(my_words)))