import re
import random
import math

import numpy
import process_summaries as pm
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB

from config import TRAIN_TEST_RATIO

# constants
BINARY = True
NONWORDS = re.compile('[\W_]+')
STOPWORDS = stopwords.words('english')


def freq(lst):
    freq_dict = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq_dict:
            freq_dict[ele] = 0
        freq_dict[ele] += 1
    return freq_dict, length


def get_unigram(summary):
    return freq(summary.split())


def balance_dataset(movies, movies_each_decade):

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])

    balanced_movies = list([])

    for decade in decades:
        decade_movies = list([])
        for movie in movies:
            if movie['year'] == decade:
                decade_movies.append(movie)
        balanced_movies.extend(random.sample(decade_movies, movies_each_decade))

    # print(len(balanced_movies))
    return balanced_movies


def likelihood_word_per_decade(word, decade_unigram):

    drichlet_prior = 0.000001

    if word in decade_unigram[0].keys():
        return float(decade_unigram[0][word]) / float(decade_unigram[1])
    else:
        return drichlet_prior


def rank_classification(test_movies, list_of_decade_features):

    # correct_classification = 0
    count_movies = 0
    pairs_of_guesses_and_actuals = list([])

    for test_movie in test_movies:
        guessed_decade = naive_bayes(test_movie, 'all', list_of_decade_features)
        actual_decade = test_movie['year']

        guessed_decade = sorted(guessed_decade, key=lambda tup: tup[1], reverse=True)
        guessed_decade = [x[0] for x in guessed_decade]
        location_of_correct_answer = guessed_decade.index(actual_decade)

        # print(str(count_movies) + " out of " + str(len(test_movies)))
        count_movies += 1

        pairs_of_guesses_and_actuals.append((guessed_decade, actual_decade, location_of_correct_answer))

    return pairs_of_guesses_and_actuals


def naive_bayes(movie, return_type, list_of_decade_features):

    summary_words = pm.process_one_plot(movie)[1]

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    drichlet_prior = math.log10(0.000001)

    decade_value_pair = list([])
    guessed_decade = 0
    max_value = 0
    for decade in decades:
        # print('NB in decade ' + str(decade))
        sum_log_likelihood = 0
        # decade_unigram = list_of_decade_features[(decade - 1930) / 10][1]
        for word in summary_words.keys():
            word_likelihood = list_of_decade_features[decade].get(word)
            # print(word_likelihood)
            if word_likelihood is not None:
                sum_log_likelihood += word_likelihood
            else:
                sum_log_likelihood += drichlet_prior

        # print(decade_value_pair)
        decade_value_pair.append((decade, sum_log_likelihood))

        if max_value == 0 or max_value < sum_log_likelihood:
            max_value = sum_log_likelihood
            guessed_decade = decade

    # print(decade_value_pair)

    if return_type == 'best':
        return guessed_decade
    else:
        return decade_value_pair


def get_decade_word_probs(movies, features, number_of_words_a, number_of_words_b):
    results = pm.process_plots_mp(movies)

    word_counts = dict()
    number_of_films_per_decade = numpy.zeros(9)
    for movie in results:
        movie_ones = dict.fromkeys( movie[1].iterkeys(), 1 )
        movieResult_ones = pm.MovieResult(movie.year, movie_ones, movie.total)
        number_of_films_per_decade[(movie.year-1930) / 10] += 1
        pm.add_plot_counts(
            word_counts.setdefault(movieResult_ones.year, dict()),
            movieResult_ones.wordcounts)
    print(number_of_films_per_decade)

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    all_word_ever_probs = dict()
    for d1 in decades:
        # print(feature)
        all_word_ever_probs = dict(all_word_ever_probs.items() + features[d1].items())

    all_word_ever_probs = dict.fromkeys(all_word_ever_probs, 10000000)

    word_counts_probs = dict()
    for d2 in decades:
        word_probs = {word: (float(count) / number_of_films_per_decade[(d2-1930)/10]) for
                      (word, count) in word_counts.get(d2).iteritems()}
        word_counts_probs[d2] = word_counts_probs.setdefault(d2, word_probs)

        for decade_word_key in word_counts_probs.get(d2).keys():
            if word_counts_probs.get(d2)[decade_word_key] < all_word_ever_probs[decade_word_key]:
                all_word_ever_probs[decade_word_key] = word_counts_probs.get(d2)[decade_word_key]

    iconicity_of_words = dict()
    word_counts_iconicity_sorted_a = dict()
    word_counts_iconicity_sorted_b = dict()
    for d3 in decades:
        word_iconicity = {word: (count / all_word_ever_probs.get(word)) for
                          (word, count) in word_counts_probs.get(d3).iteritems()}
        iconicity_of_words[d3] = iconicity_of_words.setdefault(d3, word_iconicity)

        iconicity_of_words_sorted = sorted(iconicity_of_words[d3].items(), key=lambda x: x[1], reverse=True)
        top_words_tuple = iconicity_of_words_sorted[:number_of_words_b]
        top_words_b = [w_b[0] for w_b in top_words_tuple]
        top_words_a = [w_a[0] for w_a in top_words_tuple[:number_of_words_a]]
        word_counts_iconicity_sorted_a[d3] = word_counts_iconicity_sorted_a.setdefault(d3, top_words_a)
        word_counts_iconicity_sorted_b[d3] = word_counts_iconicity_sorted_b.setdefault(d3, top_words_b)

    return [word_counts_iconicity_sorted_a, word_counts_iconicity_sorted_b]


def test_sklearn_nb(train, test):
    training_movies = pm.process_plots_mp(train)
    test_movies = pm.process_plots_mp(test)

    vec = DictVectorizer()
    training_features = vec.fit_transform([movie.wordcounts for movie in training_movies]).toarray()
    training_labels = numpy.array([movie.year for movie in training_movies])

    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(training_features, training_labels)

    test_features = vec.transform([movie.wordcounts for movie in test_movies])
    test_labels = numpy.array([movie.year for movie in test_movies])

    results = mnb_classifier.predict(test_features)

    correct = sum([1 for i, result in enumerate(results) if result == test_labels[i]])
    print("sklearn's MultinomialNB classifier predicted %d/%d correctly (%0.3f%% accuracy)" % (
        correct, len(test_labels), float(correct) / len(test_labels) * 100
    ))


def split_list(full_list, ratio=TRAIN_TEST_RATIO):
    assert ratio > 0
    print('splitting %d training/test movies (%d/1 train/test ratio)' % (len(full_list), ratio))
    return ([full_list[i] for i in range(len(full_list)) if i % ratio != 0],
            [full_list[i] for i in range(len(full_list)) if i % ratio == 0])


def predict_nb(test_movies, classifier):
    classification_results = rank_classification(test_movies, classifier)
    guess_number = [x[2] for x in classification_results]
    guesses_dict = dict((i, guess_number.count(i)) for i in guess_number)
    guesses_dict[9] = 0

    correct_num = guesses_dict[0]
    print("The Naive-Bayes Classification correctly classifies %d out of %d (%0.3f%% accuracy)" % (
        correct_num, len(test_movies), float(correct_num) / len(test_movies) * 100
    ))
    return guesses_dict, classification_results


def get_iconic_words(movies):
    list_of_decade_features_all = pm.get_training_classifier(pm.get_movie_features(movies))
    iconic_words = get_decade_word_probs(movies, list_of_decade_features_all, 10, 100)
    iconic_words_10 = iconic_words[0]

    print("getting 100 most iconic words per decade")
    iconic_words_100 = iconic_words[1]

    return iconic_words_10, iconic_words_100


def remove_iconic_words(movies, iconic_words_100):
    for movie_q3 in movies:
        for word in iconic_words_100[movie_q3['year']]:
            movie_q3['summary'] = movie_q3['summary'].replace(word.encode('ascii', 'ignore'), '')
