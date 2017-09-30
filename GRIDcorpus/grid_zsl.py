# PERFORM ZERO SHOT LEARNING ON GRIDCORPUS
import numpy as np

from grid_params import *
from grid_functions import *

########################################
## ZSL equations
########################################

# minimise L(X^T * V * S, Y) + Ω(V)
# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# === dxa = (dxd) . dxm . mxz . zxa . (axa)

########################################
## Read embedding matrices
########################################

# class_to_attr_matrix ===  class_dim x attr_dim

# word2vec
class_to_attr_matrix_word2vec = np.load(os.path.join(CURR_DIR,
    'grid_embedding_matrix_word2vec.npy'))

# fasttext
class_to_attr_matrix_fasttext = np.load(os.path.join(CURR_DIR,
    'grid_embedding_matrix_fasttext.npy'))

########################################
## Read LipReader
########################################

LSTMLipreaderEncoder = load_LSTM_lipreader_encoder()

########################################
## Vars
########################################

# Number of training classes (words)
z_vals = np.array([10, 20, 30, 40])

# Number of test classes (OOV words)
t_vals = 50 - z_vals

# Number of atributes === word2vec dim
attr_dim = matrix_class_to_attr.shape[1]

train_num_of_words = z_vals[0]
test_num_of_words = t_vals[0]

########################################
## Read GRIDcorpus directories
########################################

train_val_dirs, train_val_word_numbers, train_val_words, \
        si_dirs, si_word_numbers, si_words \
    = load_speakerdirs_wordnums_words_lists(trainValSpeakersList \
                                            = [1, 2, 3, 4, 5, 6, 7, 10],
                                        siList = [13, 14])

train_val_dirs = np.array(train_val_dirs)
train_val_word_numbers = np.array(train_val_word_numbers)
train_val_words = np.array(train_val_words)

si_dirs = np.array(si_dirs)
si_word_numbers = np.array(si_word_numbers)
si_words = np.array(si_words)

########################################
## Make Word Idx
## - map words to their index in vocab
########################################

train_val_word_idx = -np.ones((len(train_val_words)))
for i in range(len(train_val_words)):
    if train_val_words[i] in GRID_VOCAB:
        train_val_word_idx[i] = GRID_VOCAB.index(train_val_words[i])

si_word_idx = -np.ones((len(si_words)))
for i in range(len(si_words)):
    if si_words[i] in GRID_VOCAB:
        si_word_idx[i] = GRID_VOCAB.index(si_words[i])

########################################
## Remove rows corresponding to words not in vocab ('a')
########################################

train_val_rows_to_keep = train_val_word_idx != -1

train_val_dirs = train_val_dirs[train_val_rows_to_keep]
train_val_word_numbers = train_val_word_numbers[train_val_rows_to_keep]
train_val_words = train_val_words[train_val_rows_to_keep]
train_val_word_idx = train_val_word_idx[train_val_rows_to_keep]

si_rows_to_keep = si_word_idx != -1

si_dirs = si_dirs[si_rows_to_keep]
si_word_numbers = si_word_numbers[si_rows_to_keep]
si_words = si_words[si_rows_to_keep]
si_word_idx = si_word_idx[si_rows_to_keep]

########################################
## Assign Training and Testing Directories
########################################

# Train data - speakers speaking certain words
# Test data - same speakers speaking other words
# SI_Test data - different speakers speaking other words

all_word_idx = np.arange(GRID_VOCAB_SIZE)

# Choose words to keep in training data - training words
np.random.seed(29)
training_word_idx = np.random.choice(all_word_idx, train_num_of_words)

# Make the rest of the words as part of testing data - testing words
testing_word_idx = np.delete(all_word_idx, train_word_idx)

# Choose those rows in data that contain training words
training_data_idx = np.array([i for i in range(len(train_val_words)) if train_val_word_idx[i] in training_word_idx])

# Choose the rest of the rows as testing data
testing_data_idx = np.delete(np.arange(len(train_val_words)), training_data_idx)

# Choose those rows in data that contain training words
si_in_vocab_data_idx = np.array([i for i in range(len(si_words)) if si_word_idx[i] in training_word_idx])

# Choose the rest of the rows as testing data
si_oov_data_idx = np.delete(np.arange(len(si_words)), si_in_vocab_data_idx)

# TRAINING DATA
training_dirs = train_val_dirs[training_data_idx]
training_word_numbers = train_val_word_numbers[training_data_idx]
training_words = train_val_words[training_data_idx]
training_word_idx = training_word_idx[training_data_idx]

# (SPEAKER-DEPENDENT) TESTING DATA
testing_dirs = train_val_dirs[testing_data_idx]
testing_word_numbers = train_val_word_numbers[testing_data_idx]
testing_words = train_val_words[testing_data_idx]
testing_word_idx = training_word_idx[testing_data_idx]

# SPEAKER-INDEPENDENT IN-VOCAB DATA
si_in_vocab_dirs = si_dirs[si_in_vocab_data_idx]
si_in_vocab_word_numbers = si_word_numbers[si_in_vocab_data_idx]
si_in_vocab_words = si_words[si_in_vocab_data_idx]
si_in_vocab_word_idx = si_word_idx[si_in_vocab_data_idx]

# SPEAKER-INDEPENDENT OOV DATA
si_oov_dirs = si_dirs[si_oov_data_idx]
si_oov_word_numbers = si_word_numbers[si_oov_data_idx]
si_oov_words = si_words[si_oov_data_idx]
si_oov_word_idx = si_word_idx[si_oov_data_idx]

########################################
## Make Training and Testing Data
########################################




