import os
import sys

#############################################################
# DIRECTORIES
#############################################################

# GRID directory
GRID_DIR = os.path.dirname(os.path.realpath(__file__))
if GRID_DIR not in sys.path:
    sys.path.append(GRID_DIR)

# Root directory with common functions
ROOT_DIR = os.path.normpath(os.path.join(GRID_DIR, '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# GRIDcorpus dataset directory
# GRID_DATA_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/GRIDcorpus'
GRID_DATA_DIR = '/home/voletiv/Datasets/GRIDcorpus'

#############################################################
# IMPORT
#############################################################

from common_params import *

#############################################################
# PARAMETERS FOR GRIDCORPUS
#############################################################

FRAMES_PER_VIDEO = 75

WORDS_PER_VIDEO = 6

FRAMES_PER_WORD = 14

NUM_OF_MOUTH_PIXELS = 1600

MOUTH_W = 40

MOUTH_H = 40

NUM_OF_UNIQUE_WORDS = 53     # including silent and short pause

# excluding 'sil' and 'sp'
# wordsVocabSize = (nOfUniqueWords - 2) + 1
GRID_WORDS_VOCAB_SIZE = NUM_OF_UNIQUE_WORDS - 2

# Actually, GRID_VOCAB_SIZE = 51,
# but 'a' is missing from word2vec vocab,
# so GRID_VOCAB_SIZE is made 50
GRID_VOCAB_SIZE = 50

# grid_vocab list
GRID_VOCAB_LIST_FILE = os.path.join(GRID_DIR, 'grid_vocabulary.txt')


#############################################################
# LOAD VOCAB LIST
#############################################################

def load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE):
    grid_vocab = []
    with open(GRID_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().split()[-1]
            grid_vocab.append(word)
    return grid_vocab

GRID_VOCAB = load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE)
