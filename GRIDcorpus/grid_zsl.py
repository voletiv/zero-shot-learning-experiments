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

# word_to_attr_matrix ===  class_dim x attr_dim

# word2vec
word_to_attr_matrix_word2vec = np.load(os.path.join(GRID_DIR,
                                                    'grid_embedding_matrix_word2vec.npy'))

# fasttext
word_to_attr_matrix_fasttext = np.load(os.path.join(GRID_DIR,
                                                    'grid_embedding_matrix_fasttext.npy'))

# WORD2VEC or FASTTEXT
word_to_attr_matrix = word_to_attr_matrix_word2vec
# word_to_attr_matrix = word_to_attr_matrix_fasttext

########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

########################################
# Vars
########################################

# Number of training classes (words)
z_vals = np.array([10, 20, 30, 40])

# Number of test classes (OOV words)
t_vals = GRID_VOCAB_SIZE - z_vals

# Number of atributes === word2vec dim
attr_dim = word_to_attr_matrix.shape[1]

train_num_of_words = z_vals[0]
test_num_of_words = t_vals[0]

########################################
# Make Training and Testing Data
########################################

training_words_idx, train_dirs, train_word_numbers, train_word_idx, \
    test_dirs, test_word_numbers, test_word_idx, \
    si_in_vocab_dirs, si_in_vocab_word_numbers, si_in_vocab_word_idx, \
    si_oov_dirs, si_oov_word_numbers, si_oov_word_idx \
    = make_train_test_siInV_siOOV_data(train_num_of_words)

########################################
# Make Inputs and Outpus to LEARN on
########################################

# train_features === mxd
# train_attributes === mxz
train_features, train_attributes = make_features_and_attributes(
    train_dirs, train_word_numbers, train_word_idx, LSTMLipreaderEncoder,
    word_to_attr_matrix)

########################################
# EMBARRASSINGLY SIMPLE LEARNING
########################################

# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# === dxa = (dxd) . dxm . mxz . zxa . (axa)

pred_feature_to_attribute_matrix = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
    train_features.T, train_features) + optG * np.eye(train_features.shape[1])),
    train_features.T), train_attributes), word_to_attr_matrix),
    np.linalg.inv(np.dot(word_to_attr_matrix.T, word_to_attr_matrix) + optL * np.eye(word_to_attr_matrix.shape[1])))



