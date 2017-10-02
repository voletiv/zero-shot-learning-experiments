import os
import sys

#############################################################
# DIRECTORIES
#############################################################

# LRW directory
if 'LRW_DIR' not in dir():
    LRW_DIR = os.path.dirname(os.path.realpath(__file__))

if LRW_DIR not in sys.path:
    sys.path.append(LRW_DIR)

# Root directory with common functions
if 'ROOT_DIR' not in dir():
    ROOT_DIR = os.path.normpath(os.path.join(LRW_DIR, '..'))

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

#############################################################
# IMPORT
#############################################################

from common_params import *

#############################################################
# PARAMETERS FOR LRW
#############################################################

LRW_VOCAB_SIZE = 500

LRW_VOCAB_LIST_FILE = os.path.join(LRW_DIR, 'lrw_vocabulary.txt')

#############################################################
# LOAD VOCAB LIST
#############################################################


def load_lrw_vocab_list(LRW_VOCAB_LIST_FILE):
    grid_vocab = []
    with open(LRW_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().lower()
            grid_vocab.append(word)
    return grid_vocab

LRW_VOCAB = load_lrw_vocab_list(LRW_VOCAB_LIST_FILE)
