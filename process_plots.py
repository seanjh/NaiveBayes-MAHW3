from collections import namedtuple
from multiprocessing import Pool
from math import log
import string

import nltk
from nltk.corpus import stopwords

import parse_movies_example as pme
from config import FILE_NAME

BASE_LOG_PROBABILITY = log(0.0001)
STOP_WORDS = frozenset(stopwords.words('english'))
PUNCTUATION = frozenset(string.punctuation)
MovieResult = namedtuple('MovieResult', 'year, wordcounts')


def tokenize(plot):
    raw_tokens = [''.join(ch for ch in token if ch not in PUNCTUATION).lower() for token in
                  nltk.word_tokenize(plot.decode('utf_8', errors='ignore'))]
    #return [token for token in raw_tokens if token not in STOP_WORDS]
    return raw_tokens


def process_one_plot(movie):
    plot_counts = dict()
    tokens = tokenize(movie.get('summary'))
    for word in tokens:
        plot_counts[word] = plot_counts.setdefault(word, 0) + 1
    return MovieResult(movie.get('year'), plot_counts)


def add_plot_counts(decade_word_counts, plot_counts):
    for word, count in plot_counts.iteritems():
        decade_word_counts[word] = decade_word_counts.setdefault(word, 0) + count


def process_plots(word_counts, movies):
    for movie in movies:
        one = process_one_plot(movie)
        add_plot_counts(word_counts.setdefault(one.year, dict()), one.wordcounts)


def process_plots_mp(movies):
    pool = Pool(4)
    results = pool.map(process_one_plot, movies)
    pool.close()
    pool.join()
    return results


def get_movie_features(movies):
    results = process_plots_mp(movies)
    word_counts = dict()
    for movie in results:
        #word_set = word_set.union(movie.wordcounts.items())
        add_plot_counts(word_counts.setdefault(movie.year, dict()), movie.wordcounts)
    return word_counts


def calculate_cond_prob(movie_features):
    word_probs = dict()
    word_set = get_full_feature_set(movie_features)
    for decade in movie_features.keys():
        word_probs[decade] = calculate_decade_cond_probs(movie_features.get(decade), word_set)
    return word_probs


def get_full_feature_set(movie_features):
    word_set = set()
    for decade in movie_features.keys():
        word_set = word_set.union(movie_features.get(decade).keys())
    return word_set


def calculate_decade_cond_probs(decade_features, word_set):
    decade_word_probs = dict()
    total_wordcount = sum([v for k, v in decade_features.iteritems()])
    for word in word_set:
        if decade_features.get(word) is None:
            decade_word_probs[word] = BASE_LOG_PROBABILITY
        else:
            decade_word_probs[word] = log(decade_features.get(word) / total_wordcount)
    return decade_word_probs


def print_top_features(word_counts, num):
    for year, words in word_counts.iteritems():
        ordered = sorted(words.items(), key=lambda t: t[1], reverse=True)
        print 'Decade %s' % str(year)
        for word in ordered[:num]:
            print '\t%s' % str(word)


def main():
    movies = pme.load_all_movies(FILE_NAME)
    features = get_movie_features(movies)
    print_top_features(features, 10)


if __name__ == '__main__':
    main()
