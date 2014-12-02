import os
import os.path
import logging
import logging.handlers

FILE_NAME = '../plot.list.gz'

BALANCE_NUM = 6000

# FREQUENCY controls whether to build features from
# word occurrence (word count) or word frequency
# (word count/total count) for each summary
FREQUENCY = False

# Number of features to preserve following decomposition for
# linear classifiers (problem 5)
N_FEATURES = 500

# Target cumulative sum of variance ratio.
TARGET_CUM_VAR_RATIO = 0.65


LOG_LEVEL = logging.DEBUG


def _setup_logger():
    logger = logging.getLogger('HW3')
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    return logger

LOGGER = _setup_logger()