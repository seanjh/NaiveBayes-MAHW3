from naive_bayes import test_sklearn_nb, balance_dataset, split_list
from config import FILE_NAME, BALANCE_NUM
import parse_movies_example as pme
import problem5 as p5


def problem4(training_movies, test_movies):
    print("START OF QUESTION 4")
    print("classifying movies using sklearn")

    test_sklearn_nb(training_movies, test_movies)

    print("END OF QUESTION 4")
    print("================================================")

    p5.problem5(training_movies, test_movies)

if __name__ == '__main__':
    movies = list(pme.load_all_movies(FILE_NAME))
    balanced = balance_dataset(movies, BALANCE_NUM)
    train, test = split_list(balanced)
    problem4(train, test)