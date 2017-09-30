import numpy as np
import os
import sys

#############################################################
# DIRECTORIES
#############################################################

# Current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# Lip reader directory
sys.path.append(os.path.normpath(os.path.join(CURR_DIR, '../Lipreader')))

# GRIDcorpus dataset directory
GRID_DIR = '/media/voletiv/01D2BF774AC76280/Datasets/GRIDcorpus'

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

# # wordIdx dictionary
# WORD_IDX = np.load(os.path.join(CURR_DIR, "wordIdx.npy")).item()
