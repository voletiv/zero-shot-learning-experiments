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

# ########################################
# # Get FULL Data
# ########################################

# train_val_dirs, train_val_word_numbers, train_val_word_idx, \
#     si_dirs, si_word_numbers, si_word_idx \
#     = get_train_val_si_dirs_wordnumbers_wordidx()

# ########################################
# # Make FULL features and one_hot_words
# ########################################

# train_val_features, train_val_one_hot_words = make_features_and_one_hot_words(
#     train_val_dirs, train_val_word_numbers, train_val_word_idx,
#     LSTMLipreaderEncoder)

# si_features, si_one_hot_words = make_features_and_one_hot_words(
#     si_dirs, si_word_numbers, si_word_idx, LSTMLipreaderEncoder)

all_vars = np.load(os.path.join(
    GRID_DIR, "train_val_si_features_onehotwords.npz"))
train_val_features = all_vars["train_val_features"]
train_val_one_hot_words = all_vars["train_val_one_hot_words"]
si_features = all_vars["si_features"]
si_one_hot_words = all_vars["si_one_hot_words"]

########################################
# Split into train and test (OOV) data
# EMBARRASSINGLY SIMPLE LEARNING
# Calc Acc
########################################

optG = 1e-4
optL = 1e-3

train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)

predVs, train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs \
    = learn_v_and_calc_accs(train_num_of_words_list, word_to_attr_matrix,
                            train_val_features, train_val_one_hot_words,
                            si_features, si_one_hot_words)

print(train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs)


# Reduce dimensions of word_to_attr_matrix

train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)

tr_a = []
te_a = []
sii_a = []
sio_a = []
si_a = []
for reduced_dim in tqdm.tqdm(range(1, 30)):
    reduced_word_to_attr_matrix = word_to_attr_matrix[:, :reduced_dim]
    predVs, train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs \
        = learn_v_and_calc_accs(train_num_of_words_list, word_to_attr_matrix,
                                train_val_features, train_val_one_hot_words,
                                si_features, si_one_hot_words)
    print(train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs)
    tr_a.append(train_accs)
    te_a.append(test_accs)
    sii_a.append(si_in_vocab_accs)
    sio_a.append(si_oov_accs)
    si_a.append(si_accs)

tr_a = np.array(tr_a)
te_a = np.array(te_a)
sii_a = np.array(sii_a)
sio_a = np.array(sio_a)
si_a = np.array(si_a)

plt.subplot(151)
plt.imshow(tr_a, cmap='gray', clim=(0., 1.))
plt.subplot(152)
plt.imshow(te_a, cmap='gray', clim=(0., 1.))
plt.subplot(153)
plt.imshow(sii_a, cmap='gray', clim=(0., 1.))
plt.subplot(154)
plt.imshow(sio_a, cmap='gray', clim=(0., 1.))
plt.subplot(155)
plt.imshow(si_a, cmap='gray', clim=(0., 1.))
plt.show()


########################################
# VAL to find optimal g and l
########################################

trainAcc = []
valAcc = []
testAcc = []
siInAcc = []
siOOVAcc = []
siAcc = []
train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)
b = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
for gExp in tqdm.tqdm(b):
    trainAcc.append([])
    valAcc.append([])
    testAcc.append([])
    siInAcc.append([])
    siOOVAcc.append([])
    siAcc.append([])
    for lExp in tqdm.tqdm(b):
        trainAcc[-1].append([])
        valAcc[-1].append([])
        testAcc[-1].append([])
        siInAcc[-1].append([])
        siOOVAcc[-1].append([])
        siAcc[-1].append([])
        for reduced_dim in tqdm.tqdm(range(1, 300, 20)):
            reduced_word_to_attr_matrix = word_to_attr_matrix[:, :reduced_dim]
            g = math.pow(10, gExp)
            l = math.pow(10, lExp)
            predVs, train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs \
                = learn_v_and_calc_accs(train_num_of_words_list, reduced_word_to_attr_matrix,
                                        train_val_features[:5000], train_val_one_hot_words[:5000],
                                        train_val_features[5000:7000], train_val_one_hot_words[5000:7000],
                                        optG=g, optL=l)
            trainAcc[-1][-1].append(train_accs)
            testAcc[-1][-1].append(test_accs)
            siInAcc[-1][-1].append(si_in_vocab_accs)
            siOOVAcc[-1][-1].append(si_oov_accs)
            siAcc[-1][-1].append(si_accs)

np.unravel_index(np.array(testAcc).argmax(), np.array(testAcc).shape)
