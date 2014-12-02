from __future__ import division
import logging
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import issparse, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

import parse_movies_example as pme
import naive_bayes as nb
from config import FILE_NAME, N_FEATURES, TARGET_CUM_VAR_RATIO, BALANCE_NUM, LOGGER

logStart = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename='logs/%s_problem5.log' % logStart,
    level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s'
)


def split_list(full_list, ratio=4):
    return ([full_list[i] for i in range(len(full_list)) if i % ratio != 0],
            [full_list[i] for i in range(len(full_list)) if i % ratio == 0])


def feature_extraction_sklearn(vectorizer, train, test):
    train_features = vectorizer.fit_transform([movie.get('summary') for movie in train])
    train_labels = np.array([movie.get('year') for movie in train])
    test_features = vectorizer.transform([movie.get('summary') for movie in test])
    test_labels = np.array([movie.get('year') for movie in test])
    return train_features, train_labels, test_features, test_labels


def feature_decomposition(transformer, train_features, test_features):
    LOGGER.info("Beginning Dimensionality reduction using truncated SVD (%d features)" % transformer.n_components)
    train_dfeatures = transformer.fit_transform(train_features)
    LOGGER.debug(["%6f " % transformer.explained_variance_ratio_[i] for i in range(5)])
    LOGGER.debug("%0.4f%% of total variance in %d features\n" % (
        100 * transformer.explained_variance_ratio_.sum(), transformer.n_components))
    return train_dfeatures, transformer.transform(test_features)


def decompose_tsvd_target(transformer, train_features, test_features, target_cuml_var_ratio=0.9):
    LOGGER.info("Aiming for %.3f%% cumulative total sum of variance" % (target_cuml_var_ratio * 100))
    #transformer = TruncatedSVD(n_components=n_features)
    train_d, test_d = feature_decomposition(transformer, train_features, test_features)
    if sum(transformer.explained_variance_ratio_) < target_cuml_var_ratio:
        return decompose_tsvd_target(
            TruncatedSVD(n_components=(transformer.n_components*2)),
            train_features, test_features,
            target_cuml_var_ratio)
    LOGGER.debug("Reduced feature vectors size: %d" % csr_matrix(train_features[-1]).toarray().size)
    return transformer, train_d, test_d


def get_correct_num(predictions, labels):
    return sum([1 for i, result in enumerate(predictions) if result == labels[i]])


def classify(classifier, train_features, train_labels, test_features,
             test_labels, desc="Linear classifer"):
    LOGGER.info("Beginning %s" % desc)
    classifier.fit(train_features, train_labels)
    results = classifier.predict(test_features)
    correct = get_correct_num(results, test_labels)
    LOGGER.info("%s predicted %d/%d correctly (%0.3f%% accuracy)\n" % (
        desc, correct, len(test_labels), correct / len(test_labels) * 100))
    return results


def rescale_features(train, test):
    LOGGER.info("Rescaling feature matrices")
    if issparse(train):
        LOGGER.info("Converting feature matrices from sparse to dense")
        train = csr_matrix(train).todense()
        test = csr_matrix(test).todense()
    scaler = StandardScaler(with_mean=False)
    train_features_rs = scaler.fit_transform(train)
    return train_features_rs, scaler.transform(test)


def prepare_features(train_movies, test_movies):
    LOGGER.debug("Training samples: %d" % len(train_movies))
    # Extract
    vectorizer = CountVectorizer(decode_error=u'replace')
    (train_features, train_labels, test_features, test_labels) = feature_extraction_sklearn(
        vectorizer, train_movies, test_movies
    )
    LOGGER.debug("Original feature vectors size: %d" % csr_matrix(train_features[-1]).toarray().size)
    return train_features, train_labels, test_features, test_labels


def five_ab(train_features, train_labels, test_features, test_labels):
    # Reduce feature dimensions
    transformer = TruncatedSVD(n_components=N_FEATURES)
    transformer, train_features, test_features = decompose_tsvd_target(
        transformer, train_features, test_features, TARGET_CUM_VAR_RATIO
    )
    #train_features, test_features = feature_decomposition(transformer, train_features, test_features)
    LOGGER.debug("Reduced feature vectors size: %d" % csr_matrix(train_features[-1]).toarray().size)

    # Rescale features
    train_features, test_features = rescale_features(train_features, test_features)
    return train_features, train_labels, test_features, test_labels


def five_c(train_features, train_labels, test_features, test_labels):
    classify(SGDClassifier(),
             train_features, train_labels, test_features, test_labels,
             "Stochastic Gradient Descent")
    classify(Perceptron(penalty='l1'),
             train_features, train_labels, test_features, test_labels,
             "Perceptron (L1)")
    classify(Perceptron(penalty='l2', n_iter=25),
             train_features, train_labels, test_features, test_labels,
             "Perceptron (L2, 25 epochs/passes)")
    classify(LinearSVC(),
             train_features, train_labels, test_features, test_labels,
             "Linear Support Vector Classification")
    classify(SVC(kernel='rbf'),
             train_features, train_labels, test_features, test_labels,
             "C-Support Vector Classification (rbf kernel)")
    classify(KNeighborsClassifier(),
             train_features, train_labels, test_features, test_labels,
             "k-nearest neighbors vote classification (k=5)")
    classify(LogisticRegression(),
             train_features, train_labels, test_features, test_labels,
             "Logistic Regression classification")


def five_f(train_features, train_labels, test_features, test_labels):
    n_features = [10, 100, 1000, 10000]
    accuracy = []
    for num in n_features:
        transformer = TruncatedSVD(n_components=num)
        d_train_feat, d_test_feat = feature_decomposition(transformer, train_features, test_features)
        d_train_feat, d_test_feat = rescale_features(d_train_feat, d_test_feat)
        classifier = LogisticRegression()
        results = classify(classifier,
                 d_train_feat, train_labels, d_test_feat, test_labels,
                 "Logistic Regression classification")
        accuracy.append(get_correct_num(results, test_labels) / len(test_labels))
    LOGGER.debug(["%d: %.4f%%" % (n_features[i], accuracy[i] * 100) for i in range(len(n_features))])
    plot_feature_decomposition(n_features, accuracy)


def plot_feature_decomposition(n_features, accuracy):
    plt.plot(accuracy, n_features)
    plt.ylabel("# of Features")
    plt.ylabel("Model Accuracy")
    plt.show()


def main():
    movies = list(pme.load_all_movies(FILE_NAME))
    balanced = nb.balance_dataset(movies, BALANCE_NUM)
    train_movies, test_movies = split_list(balanced, 3)

    train_features, train_labels, test_features, test_labels = prepare_features(train_movies, test_movies)
    original_train_features, original_test_features = train_features, test_features

    train_features, train_labels, test_features, test_labels = five_ab(train_features, train_labels, test_features, test_labels)
    five_c(train_features, train_labels, test_features, test_labels)
    five_f(original_train_features, train_labels, original_test_features, test_labels)


if __name__ == '__main__':
    main()