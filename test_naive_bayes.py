from __future__ import division
import logging
import datetime

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer

import naive_bayes as nb
import parse_movies_example as pme
from process_plots import get_movie_features, get_training_classifier, process_plots_mp
from config import FILE_NAME, BALANCE_NUM, LOGGER, set_file_logger

set_file_logger("test_naivebayes")


def print_top_features(features, num):
    for year, words in features.iteritems():
        ordered = sorted(words.items(), key=lambda t: t[1], reverse=True)
        LOGGER.info('Decade %s' % str(year))
        for word in ordered[:num]:
            LOGGER.info('\t%s' % str(word))


def test_homegrown_nb(balanced):
    test_movies = [balanced[i] for i in range(len(balanced)) if i % 3 == 0]
    training_movies = [balanced[i] for i in range(len(balanced)) if i % 3 != 0]

    features = get_movie_features(training_movies)
    classifier = get_training_classifier(features)

    results = nb.rank_classification(test_movies, classifier)
    correct = sum([1 for result in results if result[0][0] == result[1]])
    LOGGER.info("NB classifier predicted %d/%d correctly (%0.3f%% accuracy)" % (
        correct, len(results), correct / len(results) * 100
    ))


def test_sklearn_nb(balanced):
    movie_words = process_plots_mp(balanced)

    training_movies = [movie_words[i] for i in range(len(movie_words)) if i % 3 != 0]
    test_movies = [movie_words[i] for i in range(len(movie_words)) if i % 3 == 0]

    vec = DictVectorizer()
    training_features = vec.fit_transform([movie.wordcounts for movie in training_movies]).toarray()
    training_labels = np.array([movie.year for movie in training_movies])
    #LOGGER.debug("Original size of feature vectors: %d (issparse: %s)" % (
        #csr_matrix(training_features[-1]).toarray().size, str(issparse(training_features))
    #))

    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(training_features, training_labels)

    test_features = vec.transform([movie.wordcounts for movie in test_movies])
    test_labels = np.array([movie.year for movie in test_movies])

    results = mnb_classifier.predict(test_features)

    correct = sum([1 for i, result in enumerate(results) if result == test_labels[i]])
    LOGGER.info("skleanrn's MultinomialNB classifier predicted %d/%d correctly (%0.3f%% accuracy)" % (
        correct, len(test_labels), correct / len(test_labels) * 100
    ))


def main():
    movies = list(pme.load_all_movies(FILE_NAME))
    balanced = nb.balance_dataset(movies, BALANCE_NUM)
    print "Balanced samples: %d" % len(balanced)
    test_homegrown_nb(balanced)
    test_sklearn_nb(balanced)


if __name__ == '__main__':
    main()