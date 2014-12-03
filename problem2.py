from naive_bayes import balance_dataset, split_list, predict_nb
from config import FILE_NAME, BALANCE_NUM
import parse_movies_example as pme
import process_summaries as pm
import plot_nb as pnb
import problem3 as p3


def problem2(movies):
    print("START OF QUESTION 2")

    print('plotting question 2a to 2d')
    pnb.plot2ad(movies)

    print('balancing movie data (%d per decade)' % BALANCE_NUM)
    balanced_movies = balance_dataset(movies, BALANCE_NUM)

    print('plotting question 2e to 2g')
    pnb.plot2eg(balanced_movies)

    training_movies, test_movies = split_list(balanced_movies)

    print('getting all decade features from process plots (i.e., training classifier)')
    list_of_decade_features = pm.get_training_classifier(pm.get_movie_features(training_movies))

    pnb.plot_2j(movies, list_of_decade_features)

    print("applying classifier")
    guesses_dict, classification_results = predict_nb(test_movies, list_of_decade_features)

    print('plotting cumulative match curve')
    pnb.plot_2l(guesses_dict)

    print('plotting confusion matrix')
    pnb.plot_confusion_matrix(classification_results)

    print("END OF QUESTION 2")
    print("================================================")

    p3.problem3(balanced_movies, training_movies, test_movies)

if __name__ == '__main__':
    print('loading movies')
    problem2(list(pme.load_all_movies(FILE_NAME)))
    print("================================================")