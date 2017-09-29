# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

########################################
# Make the embedding matrix (words -> word vectors)
# for the vocabulary of LRW dataset
########################################

import numpy as np
import os

from gensim.models import KeyedVectors

from common_functions import *

########################################
## Params
########################################

# Directory where the file
# 'GoogleNews-vectors-negative300.bin' is saved
WORD2VEC_BIN_SAVED_DIR = '/media/voletiv/01D2BF774AC76280/Word2Vec'

LRW_VOCAB_LIST_FILE = 'lrw_vocabulary.txt'

LRW_VOCAB_SIZE = 500

EMBEDDING_DIM = 300

########################################
# Load word2vec binary file
########################################

word2vecBinFile = os.path.join(WORD2VEC_BIN_SAVED_DIR,
    'GoogleNews-vectors-negative300.bin')

word2vec = KeyedVectors.load_word2vec_format(word2vecBinFile, binary=True)

########################################
# Load vocabulary
########################################

lrw_vocab = load_lrw_vocab_list(LRW_VOCAB_LIST_FILE)

########################################
# Make embedding matrix
########################################

embedding_matrix = np.zeros((LRW_VOCAB_SIZE, EMBEDDING_DIM))

for i, word in enumerate(lrw_vocab):
    embedding_matrix[i] = word2vec.word_vec(word)

########################################
# Save embedding matrix
########################################

np.save("lrw_vocab_embedding_matrix", embedding_matrix)
