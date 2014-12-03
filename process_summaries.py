from __future__ import division
from collections import namedtuple
from multiprocessing import Pool
import math
import string

import nltk
from nltk.corpus import stopwords

from config import FREQUENCY

ALPHA = 1
BASE_LOG_PROBABILITY = math.log10(0.0001)
STOP_WORDS = frozenset(stopwords.words('english'))
PUNCTUATION = frozenset(string.punctuation)
MovieResult = namedtuple('MovieResult', 'year, wordcounts, total')


def tokenize(plot):
    raw_tokens = [''.join(ch for ch in token if ch not in PUNCTUATION).lower() for token in
                  nltk.word_tokenize(plot.decode('utf_8', errors='ignore'))]
    return raw_tokens


def process_one_plot(movie):
    # Word occurence
    plot_counts = dict()
    tokens = tokenize(movie.get('summary'))
    total = 0
    for word in tokens:
        if word == '':
            continue
        plot_counts[word] = plot_counts.setdefault(word, 0) + 1
        total += 1
    return MovieResult(movie.get('year'), plot_counts, total)


def process_one_plot_freq(movie):
    # Word frequency
    result = process_one_plot(movie)
    freq_count = {word: count/result.total for (word, count) in result.wordcounts.iteritems()}
    return MovieResult(result.year, freq_count, result.total)


def add_plot_counts(decade_word_counts, plot_counts):
    for word, count in plot_counts.iteritems():
        decade_word_counts[word] = decade_word_counts.setdefault(word, 0) + count


def process_plots(word_counts, movies):
    for movie in movies:
        one = process_one_plot(movie)
        add_plot_counts(word_counts.setdefault(one.year, dict()), one.wordcounts)


def process_plots_mp(movies, frequency=FREQUENCY):
    pool = Pool(4)
    if frequency:
        results = pool.map(process_one_plot_freq, movies)
    else:
        results = pool.map(process_one_plot, movies)
    pool.close()
    pool.join()
    return results


def get_movie_features(movies):
    results = process_plots_mp(movies)
    word_counts = dict()
    for movie in results:
        add_plot_counts(word_counts.setdefault(movie.year, dict()), movie.wordcounts)
    return word_counts


def get_training_classifier(movie_features):
    return {decade: calculate_decade_cond_probs(movie_features.get(decade))
            for (decade, features) in movie_features.iteritems()}


def calculate_decade_cond_probs(decade_features):
    total_wordcount = sum([v for k, v in decade_features.iteritems()])
    return {word: math.log10(count/total_wordcount) for (word, count) in decade_features.iteritems()}
