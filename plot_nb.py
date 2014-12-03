import re

import numpy
import matplotlib.pyplot as pyp

from naive_bayes import get_unigram, naive_bayes, NONWORDS


def plot_pmf(word, movies, dataset_type):

    years = ''
    for movie in movies:
        summary = ' '.join(re.split(NONWORDS, movie['summary'])).lower()
        if word != '':
            if word in summary:
                years += ' ' + str(movie['year'])
        else:
            years += ' ' + str(movie['year'])

    unigram_years = get_unigram(years)

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

    totals = numpy.array([], dtype=float)
    for decade in decades:
        if str(decade) in unigram_years[0].keys():
            totals = numpy.append(totals, unigram_years[0][str(decade)])
        else:
            totals = numpy.append(totals, 0.)

    pyp.clf()
    pyp.hist(decades, decades, weights=(totals / totals.sum()), align='left')
    pyp.xticks(decades[:-1])
    pyp.ylabel("PMF")
    pyp.xlabel("Decades")

    if word != '':
        pyp.title('PMF of movies containing "' + word + '" across ' + dataset_type + ' dataset')
        pyp.savefig("plots/PMF_" + word + "_" + dataset_type + ".png")
    else:
        pyp.title('PMF of all movies across ' + dataset_type + ' dataset')
        pyp.savefig("plots/PMF_all_" + dataset_type + ".png")


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

    pyp.clf()
    pyp.hist(decades, decades, weights=value, align='left')
    pyp.xticks(decades[:-1])
    pyp.ylim(value.min() - 100, value[:-1].max() + 100)
    pyp.title('Prediction for "' + movie_name + '" (actual decade ' + str(movie_to_classify['year']) + ')')
    pyp.xlabel("Decades")
    pyp.ylabel("Sum of log likelihood")
    pyp.savefig('plots/Prediction_' + movie_name.replace(' ', '_') + ".png")


def plot2ad(movies):
    plot_pmf('', movies, 'entire')
    plot_pmf('radio', movies, 'entire')
    plot_pmf('beaver', movies, 'entire')
    plot_pmf('the', movies, 'entire')


def plot2eg(balanced_movies):
    plot_pmf('radio', balanced_movies, 'balanced')
    plot_pmf('beaver', balanced_movies, 'balanced')
    plot_pmf('the', balanced_movies, 'balanced')


def plot_2j(movies, list_of_decade_features):
    print('plotting question 2j')
    plot_movie_classification("Finding Nemo", movies, list_of_decade_features)
    plot_movie_classification("The Matrix", movies, list_of_decade_features)
    plot_movie_classification("Gone with the Wind", movies, list_of_decade_features)
    plot_movie_classification("Harry Potter and the Goblet of Fire", movies, list_of_decade_features)
    plot_movie_classification("Avatar", movies, list_of_decade_features)


def plot_2l(guesses_dict):
    ones = numpy.ones(10)
    n, bins, patches = pyp.hist(numpy.array(guesses_dict.keys()) + ones, numpy.array(guesses_dict.keys()) + ones,
                                weights=(numpy.array(guesses_dict.values()) / float(sum(guesses_dict.values()))),
                                align='left', cumulative=True, color='w')

    pyp.clf()
    pyp.plot(bins[:-1], n, '-o')
    pyp.xticks((numpy.array(guesses_dict.keys()) + ones)[:-1])
    pyp.title("Cumulative match curve")
    pyp.xlabel("Number of guesses")
    pyp.ylabel("Guesses right")
    pyp.savefig("plots/Cumulative_match_curve.png")
    # pyp.show()


def plot_confusion_matrix(predictions):
    confusion_matrix = numpy.zeros((9, 9))

    for result in predictions:
        actual_year = (result[1] - 1930) / 10
        guessed_year = (result[0][0] - 1930) / 10
        if actual_year != guessed_year:
            confusion_matrix[actual_year][guessed_year] += 1

    print(confusion_matrix)
    print("The decades most confused with each other is " + str((confusion_matrix.argmax() % 9) * 10 + 1930) + " to " +
          str((confusion_matrix.argmax() / 9) * 10 + 1930))