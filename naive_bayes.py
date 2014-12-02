__author__ = 'muhammadkhadafi'

import re
import parse_movies_example as pme
from config import FILE_NAME, BALANCE_NUM
from nltk.corpus import stopwords
import numpy
import matplotlib.pyplot as pyp
import random
import math
import process_plots as pp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pprint

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
        pyp.savefig("PMF_" + word + "_" + dataset_type + ".png")
    else:
        pyp.title('PMF of all movies across ' + dataset_type + ' dataset')
        pyp.savefig("PMF_all_" + dataset_type + ".png")


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

    pyp.clf()
    pyp.hist(decades, decades, weights=value, align='left')
    pyp.xticks(decades[:-1])
    pyp.ylim(value.min() - 100, value[:-1].max() + 100)
    pyp.title('Prediction for movie "' + movie_name + '" across each decade (in log likelihood)')
    pyp.xlabel("Decades")
    pyp.ylabel("Sum of log likelihood")
    pyp.savefig('Prediction_' + movie_name.replace(' ', '_') + ".png")


def naive_bayes(movie, return_type, list_of_decade_features):

    summary_words = pp.process_one_plot(movie)[1]

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

def main():

    print('loading movies')
    movies = list(pme.load_all_movies(FILE_NAME))

    print("================================================")
    print("START OF QUESTION 2")
    print('plotting question 2a to 2d')
    plot_pmf('', movies, 'entire')
    plot_pmf('radio', movies, 'entire')
    plot_pmf('beaver', movies, 'entire')
    plot_pmf('the', movies, 'entire')

    print('balancing movie data (%d per decade)' % BALANCE_NUM)
    balanced_movies = balance_dataset(movies, BALANCE_NUM)

    print('plotting question 2e to 2g')
    plot_pmf('radio', balanced_movies, 'balanced')
    plot_pmf('beaver', balanced_movies, 'balanced')
    plot_pmf('the', balanced_movies, 'balanced')

    print('splitting training/test movies')
    count = 0
    test_movies = list([])
    training_movies = list([])

    for balanced_movie in balanced_movies:
        if count % 3 == 0:
            test_movies.append(balanced_movie)
        else:
            training_movies.append(balanced_movie)
        count += 1

    print('getting all decade features from process plots')
    list_of_decade_features = pp.get_training_classifier(pp.get_movie_features(training_movies))

    print('plotting question 2j')
    plot_movie_classification("Finding Nemo", movies, list_of_decade_features)
    plot_movie_classification("The Matrix", movies, list_of_decade_features)
    plot_movie_classification("Gone with the Wind", movies, list_of_decade_features)
    plot_movie_classification("Harry Potter and the Goblet of Fire", movies, list_of_decade_features)
    plot_movie_classification("Avatar", movies, list_of_decade_features)

    print("starting classifier")
    classification_result = rank_classification(test_movies, list_of_decade_features)
    guess_number = [x[2] for x in classification_result]
    guesses_dict = dict((i, guess_number.count(i)) for i in guess_number)
    guesses_dict[9] = 0

    correct_num = guesses_dict[0]
    print("The Naive-Bayes Classification correctly classifies %d out of %d (%0.3f%% accuracy)" % (
        correct_num, len(test_movies), float(correct_num) / len(test_movies) * 100
    ))

    print('plotting cumulative match curve')
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
    pyp.savefig("Cumulative_match_curve.png")
    # pyp.show()

    print('plotting confusion matrix')
    confusionMatrix = numpy.zeros((9, 9))

    for result in classification_result:
        actualYear = (result[1] - 1930) / 10
        guessedYear = (result[0][0] - 1930) / 10
        if actualYear != guessedYear:
            confusionMatrix[actualYear][guessedYear] += 1

    print(confusionMatrix)
    print("The decades most confused with each other is " + str((confusionMatrix.argmax() % 9) * 10 + 1930) + " to " +
          str((confusionMatrix.argmax() / 9) * 10 + 1930))
    print("END OF QUESTION 2")
    print("================================================")
    print("START OF QUESTION 3")
    print("getting 10 most iconic words per decade")
    list_of_decade_features_all = pp.get_training_classifier(pp.get_movie_features(balanced_movies))
    iconic_words = get_decade_word_probs(balanced_movies, list_of_decade_features_all, 10, 100)
    iconic_words_10 = iconic_words[0]
    pprint.pprint(iconic_words_10)

    print("getting 100 most iconic words per decade")
    iconic_words_100 = iconic_words[1]

    print("removing iconic words from balanced movies")
    balanced_movies_q3 = balanced_movies
    for movie_q3 in balanced_movies_q3:
        for word in iconic_words_100[movie_q3['year']]:
            movie_q3['summary'] = movie_q3['summary'].replace(word.encode('ascii','ignore'), '')

    print("classifying movies without iconic words")
    print('splitting training/test movies')
    count = 0
    test_movies_q3 = list([])
    training_movies_q3 = list([])

    for balanced_movie_q3 in balanced_movies_q3:
        if count % 3 == 0:
            test_movies_q3.append(balanced_movie_q3)
        else:
            training_movies_q3.append(balanced_movie_q3)
        count += 1

    print('getting all decade features from process plots')
    list_of_decade_features_q3 = pp.get_training_classifier(pp.get_movie_features(training_movies_q3))

    print("starting classifier")
    classification_result = rank_classification(test_movies_q3, list_of_decade_features_q3)
    guess_number = [x[2] for x in classification_result]
    guesses_dict = dict((i, guess_number.count(i)) for i in guess_number)
    guesses_dict[9] = 0

    correct_num = guesses_dict[0]
    print("The Naive-Bayes Classification without iconic words correctly classifies %d out of %d (%0.3f%% accuracy)" % (
        correct_num, len(test_movies_q3), float(correct_num) / len(test_movies_q3) * 100
    ))

    print("END OF QUESTION 3")
    print("================================================")
    print("START OF QUESTION 4")
    print("classifying movies using sklearn")

    balanced_movies = balance_dataset(movies, BALANCE_NUM)

    v = CountVectorizer(decode_error='ignore')
    summary_list = numpy.array([movie_dict['summary'] for movie_dict in balanced_movies])
    year_list = numpy.array([movie_dict2['year'] for movie_dict2 in balanced_movies])

    summary_vectorized = v.fit_transform(summary_list).toarray()

    test_x = []
    train_x = []
    test_y = []
    train_y = []

    for i in range(0, len(summary_list)):
        if i % 3 == 0:
            test_x.append(summary_list[i])
            test_y.append(year_list[i])
        else:
            train_x.append(summary_list[i])
            train_y.append(year_list[i])

    summary_vectorized_train = v.fit_transform(numpy.array(train_x)).toarray()
    summary_vectorized_test = v.fit_transform(numpy.array(test_x)).toarray()

    clf = MultinomialNB()
    clf.fit(summary_vectorized_train, train_y)
    pred_y = clf.predict(summary_vectorized_test)

    correct_num = (pred_y == test_y).sum()
    print("The SKLearn Naive-Bayes Classification correctly classifies %d out of %d (%0.3f%% accuracy)" % (
        correct_num, len(test_y), float(correct_num) / len(test_y) * 100
    ))
    print("END OF QUESTION 4")
    print("================================================")


def get_decade_word_probs(movies, features, number_of_words_a, number_of_words_b):
    results = pp.process_plots_mp(movies)

    word_counts = dict()
    number_of_films_per_decade = numpy.zeros(9)
    for movie in results:
        movie_ones = dict.fromkeys( movie[1].iterkeys(), 1 )
        movieResult_ones = pp.MovieResult(movie.year, movie_ones, movie.total)
        number_of_films_per_decade[(movie.year-1930) / 10] += 1
        pp.add_plot_counts(word_counts.setdefault(movieResult_ones.year, dict()), movieResult_ones.wordcounts)
    print(number_of_films_per_decade)

    decades = numpy.array([1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010])
    all_word_ever_probs = dict()
    for d1 in decades:
        # print(feature)
        all_word_ever_probs = dict(all_word_ever_probs.items() + features[d1].items())

    all_word_ever_probs = dict.fromkeys(all_word_ever_probs, 10000000)

    word_counts_probs = dict()
    for d2 in decades:
        word_probs = {word: (float(count) / number_of_films_per_decade[(d2-1930)/10]) for (word, count) in word_counts.get(d2).iteritems()}
        word_counts_probs[d2] = word_counts_probs.setdefault(d2, word_probs)

        for decade_word_key in word_counts_probs.get(d2).keys():
            if word_counts_probs.get(d2)[decade_word_key] < all_word_ever_probs[decade_word_key]:
                all_word_ever_probs[decade_word_key] = word_counts_probs.get(d2)[decade_word_key]

    iconicity_of_words = dict()
    word_counts_iconicity_sorted_a = dict()
    word_counts_iconicity_sorted_b = dict()
    for d3 in decades:
        word_iconicity = {word: (count / all_word_ever_probs.get(word)) for (word, count) in word_counts_probs.get(d3).iteritems()}
        iconicity_of_words[d3] = iconicity_of_words.setdefault(d3, word_iconicity)

        iconicity_of_words_sorted = sorted(iconicity_of_words[d3].items(), key=lambda x:x[1], reverse=True)
        top_words_tuple = iconicity_of_words_sorted[:number_of_words_b]
        top_words_b = [w_b[0] for w_b in top_words_tuple]
        top_words_a = [w_a[0] for w_a in top_words_tuple[:number_of_words_a]]
        word_counts_iconicity_sorted_a[d3] = word_counts_iconicity_sorted_a.setdefault(d3, top_words_a)
        word_counts_iconicity_sorted_b[d3] = word_counts_iconicity_sorted_b.setdefault(d3, top_words_b)

    return [word_counts_iconicity_sorted_a, word_counts_iconicity_sorted_b]


if __name__ == '__main__':
    main()
