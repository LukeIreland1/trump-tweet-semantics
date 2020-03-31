from pathlib import Path
from nltk import download
from nltk.corpus import words

download('words')

report = Path.cwd().joinpath("docs","report.md")

lines = []
with report.open() as read_file:
    lines = read_file.readlines()

my_words = []
word_count = 0
line_count = 0
for line in lines:
    print("Lines Processed: {}".format(line_count))
    for word in line.split():
        if word in words.words():
            my_words.append(word)
    line_count +=  1

print(my_words, len(my_words))