from __future__ import division

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import TruncatedSVD, RandomizedPCA
from sklearn.linear_model import SGDClassifier, Perceptron, LogisticRegression
from sklearn.preprocessing import StandardScaler

import parse_movies_example as pme
from process_plots import process_plots_mp
from config import FILE_NAME, N_FEATURES

TARGET_CUML_VAR_RATIO = 0.9


def split_list(full_list, ratio=4):
    return ([full_list[i] for i in range(len(full_list)) if i % ratio != 0],
            [full_list[i] for i in range(len(full_list)) if i % ratio == 0])


def feature_extraction(vectorizer, train_results, test_results):
    train_features = vectorizer.fit_transform([movie.wordcounts for movie in train_results])
    train_labels = np.array([movie.year for movie in train_results])
    test_features = vectorizer.transform([movie.wordcounts for movie in test_results])
    test_labels = np.array([movie.year for movie in test_results])
    return train_features, train_labels, test_features, test_labels


def feature_extraction_sklearn(vectorizer, movies):
    train, test = split_list(movies)
    train_features = vectorizer.fit_transform([movie.get('summary') for movie in train])
    train_labels = np.array([movie.get('year') for movie in train])
    test_features = vectorizer.transform([movie.get('summary') for movie in test])
    test_labels = np.array([movie.get('year') for movie in test])
    return train_features, train_labels, test_features, test_labels


def feature_hashing(vectorizer, movies):
    train, test = split_list(movies)
    train_features = vectorizer.fit_transform([movie.wordcounts for movie in train])
    train_labels = np.array([movie.year for movie in train])
    test_features = vectorizer.transform([movie.wordcounts for movie in test])
    test_labels = np.array([movie.year for movie in test])
    return train_features, train_labels, test_features, test_labels


def feature_decomposition(transformer, train_features, test_features, print_var_ratio=False):
    train_dfeatures = transformer.fit_transform(train_features)
    if print_var_ratio:
        for i, variance in enumerate(transformer.explained_variance_ratio_):
            if i % 10 == 0:
                print
            print "%f " % variance,
        print
        print "%0.4f%% of total" % (sum(transformer.explained_variance_ratio_) * 100)
    return train_dfeatures, transformer.transform(test_features)


def decompose_tsvd_target(train_features, test_features, n_features, target_cuml_var_ratio=TARGET_CUML_VAR_RATIO):
    print "Beginning Dimensionality reduction using truncated SVD (%d features)" % n_features
    transformer = TruncatedSVD(n_components=n_features)
    train_d, test_d = feature_decomposition(transformer, train_features, test_features, print_var_ratio=True)
    if sum(transformer.explained_variance_ratio_) < target_cuml_var_ratio:
        decompose_tsvd_target(train_features, test_features, n_features*2, target_cuml_var_ratio)
    return transformer, train_d, test_d


def classify(classifier, train_features, train_labels, test_features,
             test_labels, desc="Linear classifer", rescale=False):
    print "Beginning %s" % desc
    if rescale:
        print "Rescaling feature matrices"
        if issparse(train_features):
            print "Converting sparse feature matrices to dense"
            train_features = csr_matrix(train_features).todense()
            test_features = csr_matrix(test_features).todense()
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
    classifier.fit(train_features, train_labels)
    results = classifier.predict(test_features)
    correct = sum([1 for i, result in enumerate(results) if result == test_labels[i]])
    print "%s predicted %d/%d correctly (%0.3f%% accuracy)" % (
        desc, correct, len(test_labels), correct / len(test_labels) * 100
    )
    print


def run_linear_classifiers(train_features, train_labels, test_features, test_labels):
    print "Beginning Standardize features by removing the mean and scaling to unit variance"
    if CLASSIFIER_CONFIG.get("SGDC"):
        classify(SGDClassifier(loss="hinge", penalty="l2"),
                 train_features, train_labels, test_features, test_labels,
                 "Stochastic Gradient Descent", rescale=True)
    if CLASSIFIER_CONFIG.get("PERCEPTRON-L1"):
        classify(Perceptron(penalty='l1'),
                 train_features, train_labels, test_features, test_labels,
                 "Perceptron (L1)", rescale=True)
    if CLASSIFIER_CONFIG.get("PERCEPTRON-L2"):
        classify(Perceptron(penalty='l2', n_iter=25),
                 train_features, train_labels, test_features, test_labels,
                 "Perceptron (L2, 25 epochs/passes)", rescale=True)
    if CLASSIFIER_CONFIG.get("Linear-SVC"):
        classify(LinearSVC(),
                 train_features, train_labels, test_features, test_labels,
                 "Linear Support Vector Classification", rescale=True)
    if CLASSIFIER_CONFIG.get("SVC"):
        classify(SVC(kernel='rbf'),
                 train_features, train_labels, test_features, test_labels,
                 "C-Support Vector Classification (rbf kernel)", rescale=True)
    if CLASSIFIER_CONFIG.get("LOGISTIC"):
        classify(LogisticRegression(),
                 train_features, train_labels, test_features, test_labels,
                 "Logistic Regression classification")


def run_clustering_classifiers(train_features, train_labels, test_features, test_labels):
    if train_features[-1].size > 10:
        pass
    if CLASSIFIER_CONFIG.get("NN"):
        classify(KNeighborsClassifier(),
                 train_features, train_labels, test_features, test_labels,
                 "C-Support Vector Classification (rbf kernel)")


EXTRACT_CONFIG = {
    "HOMEGROWN-DENSE": False,
    "HOMEGROWN": False,
    "FHASING": True,
    "COUNTVEC": False,
    "TFIDFVEC": False
}

FEATURE_CONFIG = {
    "TSVD": False,
    "RPCA": False
}

CLASSIFIER_CONFIG = {
    "SGDC": True,
    "PERCEPTRON-L1": True,
    "PERCEPTRON-L2": True,
    "Linear-SVC": False,
    "SVC": False,
    "LOGISTIC": True,
    "NN": True
}


def main():
    movies = list(pme.load_all_movies(FILE_NAME))

    if EXTRACT_CONFIG.get("COUNTVEC"):
        print "Extracting features using CountVectorizer"
        vectorizer = CountVectorizer(decode_error=u'replace')
        (train_features, train_labels, test_features, test_labels) = feature_extraction_sklearn(vectorizer, movies)
    elif EXTRACT_CONFIG.get("TFIDFVEC"):
        if FEATURE_CONFIG.get("TSVD"):
            print "Extracting features using TfidfVectorizer for TSVD decomposition"
            vectorizer = TfidfVectorizer(smooth_idf=True, sublinear_tf=True, decode_error=u'replace')
        else:
            print "Extracting features using TfidfVectorizer"
            vectorizer = TfidfVectorizer()
        (train_features, train_labels, test_features, test_labels) = feature_extraction_sklearn(vectorizer, movies)
    elif EXTRACT_CONFIG.get("HOMEGROWN-DENSE"):
        print "Extracting features using homegrown method with DictVectorizer (DENSE)"
        word_counts = process_plots_mp(movies)
        train_counts, test_counts = split_list(word_counts)
        vectorizer = DictVectorizer(sparse=False)
        (train_features, train_labels,
         test_features, test_labels) = feature_extraction(vectorizer, train_counts, test_counts)
    elif EXTRACT_CONFIG.get("FHASING"):
        print "Extracting features using feature hashing, aka the hashing trick"
        vectorizer = HashingVectorizer(n_features=N_FEATURES, decode_error=u'replace')
        (train_features, train_labels, test_features, test_labels) = feature_extraction_sklearn(vectorizer, movies)
    else:
        print "Extracting features using homegrown method with DictVectorizer (sparse)"
        word_counts = process_plots_mp(movies)
        train_counts, test_counts = split_list(word_counts)
        vectorizer = DictVectorizer()
        (train_features, train_labels,
         test_features, test_labels) = feature_extraction(vectorizer, train_counts, test_counts)


    original_train_features, original_test_features = train_features, test_features

    if not EXTRACT_CONFIG.get("FHASING") and FEATURE_CONFIG.get("TSVD"):
        transformer, train_features, test_features = decompose_tsvd_target(original_train_features,
                                                                           original_test_features, 500)
    #elif not EXTRACT_CONFIG.get("FHASING") and FEATURE_CONFIG.get("RPCA"):
        #print "Beginning Principal component analysis (PCA) using randomized SVD"
        #transformer = RandomizedPCA(n_components=N_FEATURES)
        #train_features, test_features = feature_decomposition(transformer, train_features, test_features)
    elif EXTRACT_CONFIG.get("FHASING"):
        print "No decomposition for hashed features"
    else:
        print "No decomposition method selected"

    print "Feature dimensions: %d" % train_features[-1].size
    run_linear_classifiers(train_features, train_labels, test_features, test_labels)
    run_clustering_classifiers(train_features, train_labels, test_features, test_labels)


if __name__ == '__main__':
    main()