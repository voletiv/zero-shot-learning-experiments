# PERFORM ZERO SHOT LEARNING ON GRIDCORPUS
import numpy as np

from grid_params import *
from grid_functions import *

########################################
# ZSL equations
########################################

# minimise L(X^T * V * S, Y) + Ω(V)
# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# === dxa = (dxd) . dxm . mxz . zxa . (axa)

########################################
# Read embedding matrices
########################################

# class_to_attr_matrix ===  class_dim x attr_dim

# word2vec
class_to_attr_matrix_word2vec = np.load(os.path.join(CURR_DIR,
                                                     'grid_embedding_matrix_word2vec.npy'))

# fasttext
class_to_attr_matrix_fasttext = np.load(os.path.join(CURR_DIR,
                                                     'grid_embedding_matrix_fasttext.npy'))

########################################
# Load LipReader
########################################

LSTMLipreaderEncoder = load_LSTM_lipreader_encoder()

########################################
# Vars
########################################

# Number of training classes (words)
z_vals = np.array([10, 20, 30, 40])

# Number of test classes (OOV words)
t_vals = GRID_VOCAB_SIZE - z_vals

# Number of atributes === word2vec dim
attr_dim = matrix_class_to_attr.shape[1]

train_num_of_words = z_vals[0]
test_num_of_words = t_vals[0]

########################################
# Make Training and Testing Data
########################################

training_dirs, training_word_numbers, training_words, \
    training_word_idx, testing_dirs, testing_word_numbers, testing_words, \
    testing_word_idx, si_in_vocab_dirs, si_in_vocab_word_numbers, \
    si_in_vocab_words, si_in_vocab_word_idx, si_oov_dirs, \
    si_oov_word_numbers, si_oov_words, si_oov_word_idx \
    = make_train_test_siInV_siOOV_data(train_num_of_words)



