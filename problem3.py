import copy
import pprint

from naive_bayes import (balance_dataset, split_list,
                         predict_nb, get_iconic_words,
                         remove_iconic_words)
from config import FILE_NAME, BALANCE_NUM
import parse_movies_example as pme
import process_summaries as pm
import problem4 as p4


def problem3(balanced_movies, training_movies, test_movies):
    print("START OF QUESTION 3")
    print("getting 10 most iconic words per decade")
    iconic_words_10, iconic_words_100 = get_iconic_words(balanced_movies)
    pprint.pprint(iconic_words_10)

    print("removing iconic words from balanced movies")
    balanced_movies_q3 = copy.deepcopy(balanced_movies)
    remove_iconic_words(balanced_movies_q3, iconic_words_100)

    print("classifying movies without iconic words")
    training_movies_q3, test_movies_q3 = split_list(balanced_movies_q3)
    print('getting all decade features from process plots (i.e., training classifier)')
    list_of_decade_features_q3 = pm.get_training_classifier(pm.get_movie_features(training_movies_q3))

    print("applying classifier")
    predict_nb(test_movies_q3, list_of_decade_features_q3)

    print("END OF QUESTION 3")
    print("================================================")

    p4.problem4(training_movies, test_movies)

if __name__ == '__main__':
    movies = list(pme.load_all_movies(FILE_NAME))
    balanced = balance_dataset(movies, BALANCE_NUM)
    train, test = split_list(balanced)
    problem3(balanced, train, test)