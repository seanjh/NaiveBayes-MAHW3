import os
import os.path
import logging
import datetime

FILE_NAME = '../plot.list.gz'

# Number of movies to include for each decade in the
# balanced sample set. This number should be less than
# or equal to the minimum number of summaries available
# for any single decade.
BALANCE_NUM = 6000

# Ratio of training to test moves
TRAIN_TEST_RATIO = 3

# FREQUENCY controls whether to build features from
# word occurrence (word count) or word frequency
# (word count/total count) for each summary
FREQUENCY = False

# Number of features to preserve following decomposition for
# linear classifiers.
N_FEATURES = 500

# Target cumulative sum of variance ratio.
TARGET_CUM_VAR_RATIO = 0.75

# Set up some helper directories (if necessary)
_cur_path = os.path.abspath(os.path.abspath(__file__))
_log_path = os.path.abspath(os.path.join(_cur_path, '..', 'logs'))
_plot_path = os.path.abspath(os.path.join(_cur_path, '..', 'plots'))
if not os.path.exists(_log_path):
        os.mkdir(_log_path)
if not os.path.exists(_plot_path):
    os.mkdir(_plot_path)

# Logging config
_log_level = logging.DEBUG

LOGGER = logging.getLogger('HW3')
LOGGER.setLevel(_log_level)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_ch = logging.StreamHandler()
_ch.setFormatter(_formatter)
_ch.setLevel(logging.DEBUG)
LOGGER.addHandler(_ch)


def set_file_logger(filename):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        filename='logs/%s_%s.log' % (timestamp, filename),
        level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s'
    )