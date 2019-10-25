# Trump Tweet Semantics Report

## Table of Contents
1. [Introduction](#introduction)
    * [Background](#background)
    * [Aim](#aim)
    * [Objectives](#objectives)
2. [Literature Review](#literature-review)
3. [Parsing](#parsing)
4. [Tweet Cleaner](#tweet-cleaner)
5. [Sentiment Scoring](#sentiment-scoring)
6. [Sentiment Classfication](#sentiment-classification)
7. [Word Cloud](#word-cloud)
8. [Frequency Distribution](#frequency-distribution)
9.  [Tweet Generation](#tweet-generation)
10. [Evaluation](#evaluation)
11. [Conclusion](#conclusion)



## Introduction

This project revolves around analysing Donald Trump's twitter in various ways to provide interesting insights to his narrative.

Analysis methods include:
* Frequency Distribution
* Word Clouds
* Sentiment Analysis
* Sentiment Analysis
* Tweet Generation

These methods will be presented as a web page via PonyORM

### Background

I was interested analyzing Trump's tweets for semantics due to his controversial nature. It would be interesting to see, given what the media often say about him, if he truly is a bad/negative person.

### Aim

The point of this project will be to perform various types of analysis on the language used in Trump's tweet to see if any interesting trends arise.

### Objectives

1. Perform frequency distribution on variable length phrases
2. Render word clouds of phrases
3. Analyse whole tweets for semantics

## Literature Review

### Guide (To-Do)
* Technology Justifications
* Background Reading
* Justify using anything
* Be critical of decisions/guides/opinions

I decided to use Python as it's my strongest language, plus it's flexibility across platforms and level/variety of API support makes it an obvious choice.

I originally planned to use Twitter's API via [Twitter Search](https://github.com/ckoepp/TwitterSearch), but I couldn't use it due to being unable to apply for a Twitter Developer Account. I instead opted for someone else's collected tweets at [Trump Twitter Archive](http://www.trumptwitterarchive.com/archive). The export format wasn't great, as you had to wait a while for the page to compile all the tweets into the correct format (When it would be useful to have it precompiled) and the page doesn't actually give you a JSON file, just a text output in JSON format, that you have to slowly copy and paste into a file and use programs to format the JSON into readable format.

I saw guides such as [Basic Binary Sentiment Analysis using NLTK](https://towardsdatascience.com/basic-binary-sentiment-analysis-using-nltk-c94ba17ae386), [Text Classification using NLTK](https://pythonprogramming.net/text-classification-nltk-tutorial/) and [Creating a Twtter Sentiment Analysis program using NLTK's Naive Bayes Classifier](https://towardsdatascience.com/creating-the-twitter-sentiment-analysis-program-in-python-with-naive-bayes-classification-672e5589a7ed) using NLTK's Naive Bayes Classifier, but they used pre-processed data meaning I can't use them for my tweets. This [guide](https://www.freecodecamp.org/news/how-to-make-your-own-sentiment-analyzer-using-python-and-googles-natural-language-api-9e91e1c493e/) used Google's Natural Language API to perform Sentiment Analysis but this method required internet access, and wasn't particularly fast.

Eventually, I fell upon this [article](https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/) which used TextBlob to perform sentiment analysis instead.
TextBlob is a simplified text processing library for Python, and provides a simple API for performing Natural Language Processing tasks, such as speech tagging, noun extraction, classification, translation and, most importantly, sentiment analysis.


## Parsing

## Tweet Cleaner

## Sentiment Scoring

## Sentiment Classification

## Word Cloud

## Frequency Distribution

## Evaluation

Results and findings

## Conclusion