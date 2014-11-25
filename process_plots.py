__author__ = 'muhammadkhadafi'

from collections import namedtuple
from multiprocessing import Pool
import string

import nltk
from nltk.corpus import stopwords

import parse_movies_example as pme
from config import FILE_NAME

STOP_WORDS = frozenset(stopwords.words('english'))
PUNCTUATION = frozenset(string.punctuation)
MovieResult = namedtuple('MovieResult', 'year, wordcounts')


def tokenize(plot):
    # raw_tokens = [''.join(ch for ch in token if ch not in PUNCTUATION).lower() for token in
    #               nltk.word_tokenize(plot.decode('utf_8', errors='ignore'))]
    raw_tokens = [''.join(ch for ch in token if ch not in PUNCTUATION).lower() for token in
                  nltk.word_tokenize(plot.decode('utf_8'))]
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


def report(word_counts, num):
    for year, words in word_counts.iteritems():
        ordered = sorted(words.items(), key=lambda t: t[1], reverse=True)
        print 'Decade %s' % str(year)
        for word in ordered[:num]:
            print '\t%s' % str(word)


def main():
    movies = pme.load_all_movies(FILE_NAME)
    word_counts = dict()
    results = process_plots_mp(movies)
    for movie in results:
        #word_set = word_set.union(movie.wordcounts.items())
        add_plot_counts(word_counts.setdefault(movie.year, dict()), movie.wordcounts)
    report(word_counts, 10)


if __name__ == '__main__':
    main()