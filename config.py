import os
import os.path
import logging
import datetime

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

# Logging config
LOG_LEVEL = logging.DEBUG

LOGGER = logging.getLogger('HW3')
LOGGER.setLevel(LOG_LEVEL)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
_ch.setLevel(logging.DEBUG)
LOGGER.addHandler(_ch)


def set_file_logger(filename):
    cur_path = os.path.abspath(os.path.abspath(__file__))
    log_path = os.path.abspath(os.path.join(cur_path, '..', 'logs'))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename='logs/%s_%s.log' % (timestamp, filename),
        level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s'
    )