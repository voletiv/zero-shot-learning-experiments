# PERFORM ZERO SHOT LEARNING ON GRIDCORPUS

import matplotlib.pyplot as plt
import numpy as np
import os

from grid_params import *
from grid_functions import *

########################################
# ZSL equations
########################################

# minimise L(X^T * V * S, Y) + Ω(V)
# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
# === dxa = (dxd) . dxm . mxz . zxa . (axa)


########################################
# TO SAVE
########################################

# grid_embedding_accs = {}

grid_embedding_accs = np.load('grid_embedding_accs.npy').item()

########################################
# Set embedding type
########################################

# WORD2VEC or FASTTEXT or GLOVE
word_embedding = 'word2vec'
# word_embedding = 'fasttext'
# word_embedding = 'glove'
# word_embedding = 'ewpk'

########################################
# Read embedding matrices
########################################

# word_to_attr_matrix ===  class_dim x attr_dim

# word2vec
if word_embedding == 'word2vec':
    word_to_attr_matrix = np.load(os.path.join(GRID_DIR,
                                               'grid_embedding_matrix_word2vec.npy'))
# fasttext
elif word_embedding == 'fasttext':
    word_to_attr_matrix = np.load(os.path.join(GRID_DIR,
                                               'grid_embedding_matrix_fasttext.npy'))
# GloVe
elif word_embedding == 'glove':
    word_to_attr_matrix = np.load(os.path.join(GRID_DIR,
                                               'grid_embedding_matrix_glove.npy'))
# Eigenwords with Prior Knowledge
elif word_embedding == 'ewpk':
    word_to_attr_matrix = np.load(os.path.join(GRID_DIR,
                                               'grid_embedding_matrix_ewpk.npy'))

########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

# ########################################
# # Get FULL Data
# ########################################

# train_val_dirs, train_val_word_numbers, train_val_word_idx, \
#     = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(TRAIN_VAL_SPEAKERS_LIST)

# si_dirs, si_word_numbers, si_word_idx \
#     = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(SI_SPEAKERS_LIST)


# ########################################
# # Remove 'a' from data
# # (because some word embeddings like word2vec
# # dont contain it in their vocab)
# ########################################

# train_val_rows_to_keep = train_val_word_idx != -1

# train_val_dirs = train_val_dirs[train_val_rows_to_keep]
# train_val_word_numbers = train_val_word_numbers[train_val_rows_to_keep]
# train_val_word_idx = train_val_word_idx[train_val_rows_to_keep]

# si_rows_to_keep = si_word_idx != -1

# si_dirs = si_dirs[si_rows_to_keep]
# si_word_numbers = si_word_numbers[si_rows_to_keep]
# si_word_idx = si_word_idx[si_rows_to_keep]

# ########################################
# # Make FULL features and one_hot_words
# ########################################

# train_val_features, train_val_one_hot_words \
#     = make_GRIDcorpus_features_and_one_hot_words(
#         train_val_dirs, train_val_word_numbers, train_val_word_idx,
#         LSTMLipreaderEncoder)

# si_features, si_one_hot_words = make_GRIDcorpus_features_and_one_hot_words(
#     si_dirs, si_word_numbers, si_word_idx, LSTMLipreaderEncoder)

# np.savez("train_val_si_LSTM256_features_onehotwords",
#     train_val_LSTM256_features=train_val_LSTM256_features,
#     train_val_LSTM256_one_hot_words=train_val_LSTM256_one_hot_words,
#     si_LSTM256_features=si_LSTM256_features,
#     si_LSTM256_one_hot_words=si_LSTM256_one_hot_words)

all_vars = np.load(os.path.join(
    GRID_DIR, "train_val_si_features_onehotwords.npz"))
train_val_features = all_vars["train_val_syncnet_features"]
train_val_one_hot_words = all_vars["train_val_syncnet_one_hot_words"]
si_features = all_vars["si_syncnet_features"]
si_one_hot_words = all_vars["si_syncnet_one_hot_words"]

########################################
# Split into train and test (OOV) data
# EMBARRASSINGLY SIMPLE LEARNING
# Calc Acc
########################################

optG = 1e1
optL = 1e-2

iv_accs = []
oov_accs = []
si_iv_accs = []
si_oov_accs = []
si_accs = []

# Number of words in the training list
train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)

number_of_iterations = 100
for iter in tqdm.tqdm(range(number_of_iterations)):
    iv_accs.append([])
    oov_accs.append([])
    si_iv_accs.append([])
    si_oov_accs.append([])
    si_accs.append([])
    # For each value of number of training classes
    for train_num_of_words in tqdm.tqdm(train_num_of_words_list):
        # Pred V and calc accs
        pred_V, iv_acc, oov_acc, si_iv_acc, si_oov_acc, si_acc, \
            = learn_by_ESZSL_and_calc_accs(train_num_of_words, word_to_attr_matrix,
                                           train_val_features, train_val_one_hot_words,
                                           si_features, si_one_hot_words,
                                           optG, optL, fix_seed=False)
        # Save
        iv_accs[-1].append(iv_acc)
        oov_accs[-1].append(oov_acc)
        si_iv_accs[-1].append(si_iv_acc)
        si_oov_accs[-1].append(si_oov_acc)
        si_accs[-1].append(si_acc)

# mean and std
iv_accs_mean = np.mean(iv_accs, axis=0)
oov_accs_mean = np.mean(oov_accs, axis=0)
si_iv_accs_mean = np.mean(si_iv_accs, axis=0)
si_oov_accs_mean = np.mean(si_oov_accs, axis=0)
si_accs_mean = np.mean(si_accs, axis=0)

iv_accs_std = np.std(iv_accs, axis=0)
oov_accs_std = np.std(oov_accs, axis=0)
si_iv_accs_std = np.std(si_iv_accs, axis=0)
si_oov_accs_std = np.std(si_oov_accs, axis=0)
si_accs_std = np.std(si_accs, axis=0)

# Print
for i in range(len(train_num_of_words_list)):
    print(iv_accs_mean[i], oov_accs_mean[i], si_iv_accs_mean[i],
          si_oov_accs_mean[i], si_accs_mean[i])

# Plot
plt.errorbar(train_num_of_words_list, iv_accs_mean, yerr=iv_accs_std,
             capsize=3, label='iv - speaker-dependent')
plt.errorbar(train_num_of_words_list, oov_accs_mean, yerr=oov_accs_std,
             capsize=3, label='oov - speaker-dependent')
plt.errorbar(train_num_of_words_list, si_iv_accs_mean, yerr=si_iv_accs_std,
             capsize=3, label='iv - speaker-INdependent')
plt.errorbar(train_num_of_words_list, si_oov_accs_mean, yerr=si_oov_accs_std,
             capsize=3, label='oov - speaker-INdependent')
plt.errorbar(train_num_of_words_list, si_accs_mean, yerr=si_accs_std,
             capsize=3, label='[iv+oov] - speaker-INdependent')
plt.legend(loc='upper right')
plt.ylim([0., 1.])
plt.gca().yaxis.grid(True, alpha=0.5)
plt.xlabel("Number of words in the training vocabulary, out of 50")
plt.ylabel("Accuracy")
plt.title("ESZSL - GRIDcorpus - " + word_embedding)
plt.show()

########################################
# Save
########################################

grid_embedding_accs[word_embedding] = {}

grid_embedding_accs[word_embedding]["iv_accs_mean"] = iv_accs_mean
grid_embedding_accs[word_embedding]["oov_accs_mean"] = oov_accs_mean
grid_embedding_accs[word_embedding]["si_iv_accs_mean"] = si_iv_accs_mean
grid_embedding_accs[word_embedding]["si_oov_accs_mean"] = si_oov_accs_mean
grid_embedding_accs[word_embedding]["si_accs_mean"] = si_accs_mean

grid_embedding_accs[word_embedding]["iv_accs_std"] = iv_accs_std
grid_embedding_accs[word_embedding]["oov_accs_std"] = oov_accs_std
grid_embedding_accs[word_embedding]["si_iv_accs_std"] = si_iv_accs_std
grid_embedding_accs[word_embedding]["si_oov_accs_std"] = si_oov_accs_std
grid_embedding_accs[word_embedding]["si_accs_std"] = si_accs_std

np.save("grid_embedding_accs", grid_embedding_accs)

########################################
# Mini plots
########################################

# OOV
plt.errorbar(train_num_of_words_list, grid_embedding_accs['word2vec']['oov_accs_mean'],
             yerr=grid_embedding_accs['word2vec']['oov_accs_std'], label='word2vec',
             linestyle='--', fmt='o', capsize=3)
plt.errorbar(train_num_of_words_list, grid_embedding_accs['fasttext']['oov_accs_mean'],
             yerr=grid_embedding_accs['fasttext']['oov_accs_std'], label='fasttext',
             linestyle='--', fmt='o', capsize=3)
plt.errorbar(train_num_of_words_list, grid_embedding_accs['glove']['oov_accs_mean'],
             yerr=grid_embedding_accs['glove']['oov_accs_std'], label='glove',
             linestyle='--', fmt='o', capsize=3)
plt.errorbar(train_num_of_words_list, grid_embedding_accs['ewpk']['oov_accs_mean'],
             yerr=grid_embedding_accs['glove']['oov_accs_std'], label='ewpk',
             linestyle='--', fmt='o', capsize=3)
plt.legend(fontsize=12)
plt.ylim([0., 1.])
plt.gca().yaxis.grid(True, alpha=0.5)
plt.xlabel("Number of words in the training vocabulary, out of 50")
plt.ylabel("Accuracy")
plt.title("ZSL OOV accuracies on GRIDcorpus")
plt.show()

# SI
plt.errorbar(train_num_of_words_list, grid_embedding_accs['word2vec']['si_iv_accs_mean'],
             yerr=grid_embedding_accs['word2vec']['si_iv_accs_std'], label='iv-word2vec',
             linestyle=':', fmt='o', capsize=3, c='C0')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['fasttext']['si_iv_accs_mean'],
             yerr=grid_embedding_accs['fasttext']['si_iv_accs_std'], label='iv-fasttext',
             linestyle=':', fmt='o', capsize=3, c='C1')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['glove']['si_iv_accs_mean'],
             yerr=grid_embedding_accs['glove']['si_iv_accs_std'], label='iv-glove',
             linestyle=':', fmt='o', capsize=3, c='C2')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['ewpk']['si_iv_accs_mean'],
             yerr=grid_embedding_accs['glove']['si_iv_accs_std'], label='iv-ewpk',
             linestyle=':', fmt='o', capsize=3, c='C3')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['word2vec']['si_oov_accs_mean'],
             yerr=grid_embedding_accs['word2vec']['si_oov_accs_std'], label='oov-word2vec',
             linestyle='-', fmt='o', capsize=3, c='C0')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['fasttext']['si_oov_accs_mean'],
             yerr=grid_embedding_accs['fasttext']['si_oov_accs_std'], label='oov-fasttext',
             linestyle='-', fmt='o', capsize=3, c='C1')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['glove']['si_oov_accs_mean'],
             yerr=grid_embedding_accs['glove']['si_oov_accs_std'], label='oov-glove',
             linestyle='-', fmt='o', capsize=3, c='C2')
plt.errorbar(train_num_of_words_list, grid_embedding_accs['ewpk']['si_oov_accs_mean'],
             yerr=grid_embedding_accs['glove']['si_oov_accs_std'], label='oov-ewpk',
             linestyle='-', fmt='o', capsize=3, c='C3')
plt.legend(fontsize=11, loc='upper right')
plt.ylim([0., 1.])
plt.gca().yaxis.grid(True, alpha=0.5)
plt.xlabel("Number of words in the training vocabulary, out of 50")
plt.ylabel("Accuracy")
plt.title("ZSL Speaker-Independent accuracies on GRIDcorpus")
plt.show()


# ########################################
# # VAL to find optimal g and l
# ########################################

# # CONCLUSIONS
# # 1. Let's keep dimension of word embedding 300 (max), because accs are
# # saturating by that, so no harm
# # 2. Let's choose gExp and lExp as 7 and 4 for all values of z
# # (train_num_of_words), since diff between acc at optimal value for each value
# # of z and this value is not significant => optG = 10, optL = .01

# iv_train_accs = []
# iv_val_accs = []

# train_num_of_words_list = np.arange(5, GRID_VOCAB_SIZE, 5)
# word_embedding_dimensions_list = np.arange(1, 300, 20)

# b = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]

# for train_num_of_words in tqdm.tqdm(train_num_of_words_list):
#     iv_train_accs.append([])
#     iv_val_accs.append([])
#     for gExp in tqdm.tqdm(b):
#         iv_train_accs[-1].append([])
#         iv_val_accs[-1].append([])
#         for lExp in tqdm.tqdm(b):
#             iv_train_accs[-1][-1].append([])
#             iv_val_accs[-1][-1].append([])
#             g = math.pow(10, gExp)
#             l = math.pow(10, lExp)
#             # Choose training words
#             training_words_idx = choose_words_for_training(
#                 train_num_of_words, GRID_VOCAB_SIZE)
#             # Split data into in_vocab and OOV
#             in_vocab_features, in_vocab_one_hot_words, oov_features, oov_one_hot_words \
#                 = split_data_into_in_vocab_and_oov(training_words_idx,
#                                                    train_val_features,
#                                                    train_val_one_hot_words)
#             # Choose only 10000 data points
#             num_of_data = 10000
#             in_vocab_features = in_vocab_features[:num_of_data]
#             in_vocab_one_hot_words = in_vocab_one_hot_words[:num_of_data]
#             # Choose train and val data within training data
#             np.random.seed(29)
#             val_data_idx = np.random.choice(
#                 len(in_vocab_features), int(.2 * len(in_vocab_features)), replace=False)
#             train_data_idx = np.delete(
#                 np.arange(len(in_vocab_features)), val_data_idx)
#             # Make train and val data from in_vocab_data
#             train_features = in_vocab_features[train_data_idx]
#             train_one_hot_words = in_vocab_one_hot_words[train_data_idx]
#             val_features = in_vocab_features[val_data_idx]
#             val_one_hot_words = in_vocab_one_hot_words[val_data_idx]
#             for reduced_dim in tqdm.tqdm(word_embedding_dimensions_list):
#                 # Split embedding matrix into in_vocab and oov
#                 reduced_word_to_attr_matrix = word_to_attr_matrix[
#                     :, :reduced_dim]
#                 in_vocab_word_to_attr_matrix, oov_word_to_attr_matrix \
#                     = split_embedding_matrix_into_in_vocab_and_oov(
#                         training_words_idx, reduced_word_to_attr_matrix)
#                 # Train
#                 pred_V = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
#                     in_vocab_features.T, in_vocab_features)
#                     + g * np.eye(train_features.shape[1])),
#                     train_features.T), train_one_hot_words), in_vocab_word_to_attr_matrix),
#                     np.linalg.inv(np.dot(in_vocab_word_to_attr_matrix.T,
#                                          in_vocab_word_to_attr_matrix)
#                                   + l * np.eye(in_vocab_word_to_attr_matrix.shape[1])))
#                 # Train Acc
#                 y_train_preds = np.argmax(np.dot(
#                     np.dot(train_features, pred_V), in_vocab_word_to_attr_matrix.T), axis=1)
#                 iv_train_accs[-1][-1][-1].append(np.sum(y_train_preds == np.argmax(
#                     train_one_hot_words, axis=1)) / len(train_one_hot_words))
#                 # Val Acc
#                 y_val_preds = np.argmax(np.dot(
#                     np.dot(val_features, pred_V), in_vocab_word_to_attr_matrix.T), axis=1)
#                 iv_val_accs[-1][-1][-1].append(np.sum(y_val_preds == np.argmax(
#                     val_one_hot_words, axis=1)) / len(val_one_hot_words))

# # Plots

# # For each z
# for i in range(len(train_num_of_words_list)):
#     opt_dims = np.unravel_index(np.array(iv_val_accs)[
#                                 i, :, :, -1].argmax(), np.array(iv_val_accs)[i, :, :, -1].shape)
#     my_image = np.reshape(np.array(iv_val_accs)[
#                           i, :, :, -1], (len(b), len(b)))
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(my_image, cmap='gray', clim=(0.5, 1.))
#     plt.scatter([opt_dims[1]], [opt_dims[0]], c='g')
#     plt.title(str(train_num_of_words_list[i]) + " words in training")
#     plt.xlabel("l exponent")
#     plt.ylabel("g exponent")

# plt.suptitle("Finding optimal g and l values for GRIDcorpus")
