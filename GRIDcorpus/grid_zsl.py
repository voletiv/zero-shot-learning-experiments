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
# Get FULL Data
########################################

train_val_dirs, train_val_word_numbers, train_val_word_idx, \
    si_dirs, si_word_numbers, si_word_idx \
    = get_train_val_si_dirs_wordnumbers_wordidx()

########################################
# Make FULL features and one_hot_words
########################################

train_val_features, train_val_one_hot_words = make_features_and_one_hot_words(
    train_val_dirs, train_val_word_numbers, train_val_word_idx,
    LSTMLipreaderEncoder, word_to_attr_matrix)

si_features, si_one_hot_words = make_features_and_one_hot_words(
    si_dirs, si_word_numbers, si_word_idx,
    LSTMLipreaderEncoder, word_to_attr_matrix)

########################################
# Split into train and test (OOV) data
# EMBARRASSINGLY SIMPLE LEARNING
# Calc Acc
########################################


def find_accs(train_num_of_words_list):
    train_accs = []
    test_accs = []
    si_in_vocab_accs = []
    si_oov_accs = []
    for train_num_of_words in train_num_of_words_list:
        # Split
        train_features, train_one_hot_words, test_features, test_one_hot_words, \
            si_in_vocab_features, si_in_vocab_one_hot_words, \
            si_oov_features, si_oov_one_hot_words \
            = make_train_test_siI_siOOV_data(
                train_num_of_words,
                train_val_features, train_val_one_hot_words,
                si_features, si_one_hot_words)
        # Pred V
        optG = 1e-6
        optL = 1e-3
        pred_feature_to_attribute_matrix = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
            train_features.T, train_features) + optG * np.eye(train_features.shape[1])),
            train_features.T), train_one_hot_words), word_to_attr_matrix),
            np.linalg.inv(np.dot(word_to_attr_matrix.T, word_to_attr_matrix) + optL * np.eye(word_to_attr_matrix.shape[1])))
        # Acc
        y_train_pred = np.argmax(
            np.dot(np.dot(train_features, predV), word_to_attr_matrix.T), axis=1)
        train_acc.append(np.sum(y_train_pred == np.argmax(
            train_one_hot_words, axis=1)) / len(train_one_hot_words))
        train_accs.append(train_acc)
        y_test_pred = np.argmax(
            np.dot(np.dot(test_features, predV), word_to_attr_matrix.T), axis=1)
        test_acc.append(np.sum(y_test_pred == np.argmax(
            test_one_hot_words, axis=1)) / len(test_one_hot_words))
        test_accs.append(test_acc)
        y_si_in_vocab_pred = np.argmax(
            np.dot(np.dot(si_in_vocab_features, predV), word_to_attr_matrix.T), axis=1)
        si_in_vocab_acc.append(np.sum(y_si_in_vocab_pred == np.argmax(
            si_in_vocab_one_hot_words, axis=1)) / len(si_in_vocab_one_hot_words))
        si_in_vocab_accs.append(si_in_vocab_acc)
        y_si_oov_pred = np.argmax(
            np.dot(np.dot(si_oov_features, predV), word_to_attr_matrix.T), axis=1)
        si_oov_acc.append(np.sum(y_si_oov_pred == np.argmax(
            si_oov_one_hot_words, axis=1)) / len(si_oov_one_hot_words))
        si_oov_accs.append(si_oov_acc)
    return train_accs, test_accs, si_in_vocab_accs, si_oov_accs

train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)

train_accs, test_accs, si_in_vocab_accs, si_oov_accs = find_accs(
    train_num_of_words_list)

print(train_accs, test_accs, si_in_vocab_accs, si_oov_accs)





########################################
# Split into train and test (OOV) data
########################################

train_features, train_one_hot_words, test_features, test_one_hot_words, \
    si_in_vocab_features, si_in_vocab_one_hot_words, \
    si_oov_features, si_oov_one_hot_words \
    = make_train_test_siI_siOOV_data(
        train_num_of_words,
        train_val_features, train_val_one_hot_words,
        si_features, si_one_hot_words)

########################################
# EMBARRASSINGLY SIMPLE LEARNING
########################################

# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# === dxa = (dxd) . dxm . mxz . zxa . (axa)

optG = 1e-6
optL = 1e-3

pred_feature_to_attribute_matrix = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
    train_features.T, train_features) + optG * np.eye(train_features.shape[1])),
    train_features.T), train_one_hot_words), word_to_attr_matrix),
    np.linalg.inv(np.dot(word_to_attr_matrix.T, word_to_attr_matrix) + optL * np.eye(word_to_attr_matrix.shape[1])))


########################################
# ACCURACY CALC
########################################

y_train_pred = np.argmax(
    np.dot(np.dot(train_features, predV), word_to_attr_matrix.T), axis=1)
train_acc.append(np.sum(y_train_pred == np.argmax(
    train_one_hot_words, axis=1)) / len(train_one_hot_words))

y_test_pred = np.argmax(
    np.dot(np.dot(test_features, predV), word_to_attr_matrix.T), axis=1)
test_acc.append(np.sum(y_test_pred == np.argmax(
    test_one_hot_words, axis=1)) / len(test_one_hot_words))

y_si_in_vocab_pred = np.argmax(
    np.dot(np.dot(si_in_vocab_features, predV), word_to_attr_matrix.T), axis=1)
si_in_vocab_acc.append(np.sum(y_si_in_vocab_pred == np.argmax(
    si_in_vocab_one_hot_words, axis=1)) / len(si_in_vocab_one_hot_words))

y_si_oov_pred = np.argmax(
    np.dot(np.dot(si_oov_features, predV), word_to_attr_matrix.T), axis=1)
si_oov_acc.append(np.sum(y_si_oov_pred == np.argmax(
    si_oov_one_hot_words, axis=1)) / len(si_oov_one_hot_words))


# ########################################
# # VAL to find optimal g and l
# ########################################

# train_idx = np.arange(len(train_features))
# np.random.shuffle(train_idx)
# tf = np.array(train_features[train_idx])
# vf = tf[:int(0.2 * len(tf))]
# tf = tf[int(0.2 * len(tf)):]
# to = np.array(train_one_hot_words[train_idx])
# vo = to[:int(0.2 * len(to))]
# to = to[int(0.2 * len(to)):]

# trainAcc = []
# valAcc = []
# testAcc = []
# siInAcc = []
# siOOVAcc = []
# b = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
# for gExp in tqdm.tqdm(b):
#     for lExp in b:
#         g = math.pow(10, gExp)
#         l = math.pow(10, lExp)
#         predV = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
#             tf.T, tf) + g * np.eye(tf.shape[1])),
#             tf.T), to), word_to_attr_matrix),
#             np.linalg.inv(np.dot(word_to_attr_matrix.T, word_to_attr_matrix) + l * np.eye(word_to_attr_matrix.shape[1])))
#         yTrPred = np.argmax(
#             np.dot(np.dot(tf, predV), word_to_attr_matrix.T), axis=1)
#         trainAcc.append(np.sum(yTrPred == np.argmax(to, axis=1)) / len(to))
#         yVPred = np.argmax(
#             np.dot(np.dot(vf, predV), word_to_attr_matrix.T), axis=1)
#         valAcc.append(np.sum(yVPred == np.argmax(vo, axis=1)) / len(vo))
#         yTestPred = np.argmax(
#             np.dot(np.dot(test_features, predV), word_to_attr_matrix.T), axis=1)
#         testAcc.append(np.sum(yTestPred == np.argmax(
#             test_one_hot_words, axis=1)) / len(test_one_hot_words))
#         ySiInPred = np.argmax(
#             np.dot(np.dot(si_in_vocab_features, predV), word_to_attr_matrix.T), axis=1)
#         siInAcc.append(np.sum(ySiInPred == np.argmax(
#             si_in_vocab_one_hot_words, axis=1)) / len(si_in_vocab_one_hot_words))
#         ySiOOVPred = np.argmax(
#             np.dot(np.dot(si_oov_features, predV), word_to_attr_matrix.T), axis=1)
#         siOOVAcc.append(np.sum(ySiOOVPred == np.argmax(
#             si_oov_one_hot_words, axis=1)) / len(si_oov_one_hot_words))
