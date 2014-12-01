__author__ = 'muhammadkhadafi'

import re
import parse_movies_example as pme
from config import FILE_NAME
from nltk.corpus import stopwords
import numpy
import matplotlib.pyplot as pyp
import random
import math
import process_plots as pp
import datetime as dt
# import process_plots_2 as pp2

# constants
BINARY = True
NONWORDS = re.compile('[\W_]+')
STOPWORDS = stopwords.words('english')


def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return (freq, length)

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

def plot_pmf(word, movies):

    years = ''
    for movie in movies:
        summary = ' '.join(re.split(NONWORDS, movie['summary'])).lower()
        if word in summary:
            years += ' ' + str(movie['year'])

    unigram_years = get_unigram(years)

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

    totals = numpy.array([], dtype=float)
    for decade in decades:
        if decade != 2020:
            totals = numpy.append(totals, unigram_years[0][str(decade)])
        else:
            totals = numpy.append(totals, 0.)

    print(totals)

    pyp.hist(decades, decades, weights=(totals / totals.sum()), align='left')
    pyp.xticks(decades[:-1])
    pyp.show()

    print(get_unigram(years))

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



def plot_movie_classification(movie_name, all_movies, list_of_decade_features):

    movie_to_classify = dict()
    for movie in all_movies:
        if movie['title'] == movie_name:
            movie_to_classify = movie
            break

    decade_value_tuples = naive_bayes(movie_to_classify, 'all', list_of_decade_features)
    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

    value = numpy.array([])

    for decade_value_tuple in decade_value_tuples:
        value = numpy.append(value, decade_value_tuple[1])
    value = numpy.append(value, 0.)

    pyp.hist(decades, decades, weights=value, align='left')
    pyp.xticks(decades[:-1])
    pyp.ylim(value.min() - 100, value[:-1].max() + 100)
    pyp.show()


def naive_bayes(movie, return_type, list_of_decade_features):

    summary_words = pp.process_one_plot(movie)[1]

    # print(pp.process_one_plot(movie)[1])
    # print(pp.process_one_plot(movie)[1].get(u'the'))

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


def get_decade_word_probs(movies, number_of_words):
    results = pp.process_plots_mp(movies)
    print("finish pp process plots")
    word_counts = dict()
    number_of_films_per_decade = numpy.zeros(9)
    for movie in results:
        movie_ones = dict.fromkeys( movie[1].iterkeys(), 1 )
        movieResult_ones = pp.MovieResult(movie.year, movie_ones, movie.total)
        # pp.add_plot_counts(word_counts.setdefault(movie.year, dict()), movie.wordcounts)
        # print(movieResult_ones)
        number_of_films_per_decade[(movie.year-1930) / 10] += 1
        pp.add_plot_counts(word_counts.setdefault(movieResult_ones.year, dict()), movieResult_ones.wordcounts)
    print("finish word_counts")
    print(number_of_films_per_decade)
    print(word_counts.get(1930).get('are'))

    word_counts_probs = dict()
    word_counts_probs_sorted = dict()
    all_word_ever_probs = dict()
    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    for decade in decades:
        word_probs = {word: (float(count) / number_of_films_per_decade[(decade-1930)/10]) for (word, count) in word_counts.get(decade).iteritems()}
        word_counts_probs[decade] = word_counts_probs.setdefault(decade, word_probs)
        # print("finish " + str(decade) + " word_probs")

        for decade_word_key in word_counts_probs.get(decade).keys():
            if decade_word_key in all_word_ever_probs.keys():
                if word_counts_probs.get(decade)[decade_word_key] < all_word_ever_probs[decade_word_key]:
                    all_word_ever_probs[decade_word_key] = word_counts_probs.get(decade)[decade_word_key]
            else:
                all_word_ever_probs[decade_word_key] = word_counts_probs.get(decade)[decade_word_key]

        # print("finish adding " + str(decade) + " words to all words ever")

        # for key
        # word_probs_sorted = sorted(word_probs.items(), key=lambda x: x[1])
        # word_counts_probs_sorted[decade] = word_counts_probs.setdefault(decade, word_probs_sorted[:number_of_words])
    print("finish word_counts_probs")
    print(word_counts_probs.get(1930).get('are'))
    print(all_word_ever_probs.get('are'))

    iconicity_of_words = dict()
    word_counts_iconicity_sorted = dict()
    for decadeAgain in decades:
        word_iconicity = {word: (count / all_word_ever_probs.get(word)) for (word, count) in word_counts_probs.get(decadeAgain).iteritems()}
        iconicity_of_words[decadeAgain] = iconicity_of_words.setdefault(decadeAgain, word_iconicity)


        sorted(iconicity_of_words.items(), key=lambda x: x[1])
        word_counts_iconicity_sorted[decade] = word_counts_probs.setdefault(decade, iconicity_of_words[:number_of_words])

    print("finish iconicity_of_words")
    # print(iconicity_of_words.get(1930).get('are'))

    print(word_counts_iconicity_sorted.get(1930))



    # print(word_counts_probs_sorted.get(1930))

    # return word_counts_probs_sorted


def main():

    movies = list(pme.load_all_movies(FILE_NAME))
    print('finish loading movies')

    # plot_pmf('radio', movies)
    # plot_pmf('beaver', movies)
    # plot_pmf('the', movies)

    # balanced_movies = balance_dataset(movies, 6000)
    balanced_movies = balance_dataset(movies, 100)
    print('finish balancing movies')

    # print(pp.process_plots_mp(balanced_movies)[0])
    # print(pp.process_plots_mp(balanced_movies)[1])

    # print(pp.get_training_classifier(pp.get_movie_features(balanced_movies))[1930])
    # plot_pmf('radio', balanced_movies)
    # plot_pmf('beaver', balanced_movies)
    # plot_pmf('the', balanced_movies)

    count = 0
    test_movies = list([])
    training_movies = list([])

    for balanced_movie in balanced_movies:
        if count % 3 == 0:
            test_movies.append(balanced_movie)
        else:
            training_movies.append(balanced_movie)
        count += 1

    print('finish splitting training/test movies')

    list_of_decade_features = pp.get_training_classifier(pp.get_movie_features(training_movies))

    print('finish getting all decade unigrams from process plots')

    # print(pp.process_one_plot(movies[0]))
    # print(pp.get_movie_features(training_movies).get(1930))

    # plot_movie_classification("Finding Nemo", movies, list_of_decade_features)
    # plot_movie_classification("The Matrix", movies, list_of_decade_unigrams)
    # plot_movie_classification("Gone with the Wind", movies, list_of_decade_unigrams)
    # plot_movie_classification("Harry Potter and the Goblet of Fire", movies, list_of_decade_unigrams)
    # plot_movie_classification("Avatar", movies, list_of_decade_unigrams)

    print(dt.datetime.now())
    classification_result = rank_classification(test_movies, list_of_decade_features)
    guess_number = [x[2] for x in classification_result]
    guesses_dict = dict((i, guess_number.count(i)) for i in guess_number)
    guesses_dict[9] = 0

    print("The Naive-Bayes Classification 2 correctly classifies " + str(guesses_dict[0]) + " out of " +
          str(len(test_movies)))

    print(guesses_dict)

    print(dt.datetime.now())

    # ones = numpy.ones(10)
    # n, bins, patches = pyp.hist(numpy.array(guesses_dict.keys()) + ones, numpy.array(guesses_dict.keys()) + ones,
    #                             weights=(numpy.array(guesses_dict.values()) / float(sum(guesses_dict.values()))),
    #                             align='left', cumulative=True, color='w')
    #
    # pyp.plot(bins[:-1], n, '-o')
    # pyp.xticks((numpy.array(guesses_dict.keys()) + ones)[:-1])
    # pyp.show()


    confusionMatrix = numpy.zeros ((9, 9))
    # confusionMatrix[0][0] = 1


    for result in classification_result:
        actualYear = (result[1] - 1930) / 10
        guessedYear = (result[0][0] - 1930) / 10
        if actualYear != guessedYear:
            confusionMatrix[actualYear][guessedYear] += 1

    print(confusionMatrix)
    print("The decades most confused with each other is " + str((confusionMatrix.argmax() % 9) * 10 + 1930) + " to " +
          str((confusionMatrix.argmax() / 9) * 10 + 1930))



if __name__ == '__main__':
    # main()

    movies = list(pme.load_all_movies(FILE_NAME))
    balanced_movies = balance_dataset(movies, 200)
    get_decade_word_probs(balanced_movies, 10)