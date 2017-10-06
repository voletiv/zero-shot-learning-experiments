# word2vec - http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
# word2vec - https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings
# fasttext - https://blog.manash.me/how-to-use-pre-trained-word-vectors-from-facebooks-fasttext-a71e6d55f27
# fasttext - https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

########################################
# Make the embedding matrix (words -> word vectors)
# for the vocabulary of LRW dataset
########################################

import numpy as np
import os
import tqdm

from gensim.models import KeyedVectors

from common_params import *
from GRIDcorpus.grid_params import *
from LRW.lrw_params import *

########################################
## Params
########################################

WORD_EMBEDDING_SAVE_DIR = '/media/voletiv/01D2BF774AC76280/Word-Embeddings/'

# Directory where the word2vec file
# 'GoogleNews-vectors-negative300.bin' is saved
WORD2VEC_BIN_SAVED_DIR = os.path.join(WORD_EMBEDDING_SAVE_DIR, 'Word2Vec')

# Directory where the fasttext file
# 'wiki.en.vec' is saved
FASTTEXT_BIN_SAVED_DIR = os.path.join(WORD_EMBEDDING_SAVE_DIR, 'FastText')

# Directory where the GloVe file
# 'glove.6B.300d.txt' is saved
GLOVE_BIN_SAVED_DIR = os.path.join(WORD_EMBEDDING_SAVE_DIR, 'GloVe')

# Directory where the EigenWordsWithPriorKnowledge file
# 'Eigenwords_Wiki5_alpha0.5_WordNetPriorKnowledge.txt' is saved
EWPK_BIN_SAVED_DIR = os.path.join(WORD_EMBEDDING_SAVE_DIR, 'EigenWordsPK')

# word2vec, fasttext, GloVe dim
EMBEDDING_DIM = 300

########################################
# Load vocabularies
########################################

# lrw_vocab = load_lrw_vocab_list(LRW_VOCAB_LIST_FILE)
# grid_vocab = load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE)

# Vocabularies loaded by GRIDcorpus.grid_params and LRW.lrw_params

########################################
# DEFINE WORD_EMBEDDING, DATASET
########################################

# word_embedding = 'word2vec'
# word_embedding = 'fasttext'
# word_embedding = 'glove'
word_embedding = 'ewpk'


dataset = 'grid'
# dataset = 'lrw'

########################################
# Load word2vec/fasttext/glove/EWPK binary files
########################################


def load_text_model(text_file, text_vocab_size):
    # Glove: 'glove.6B.300d.txt', 400000
    # EWPK: 'Eigenwords_Wiki5_alpha0.5_WordNetPriorKnowledge', 200000
    print("Loading model from", text_file)
    model = {}
    with open(text_file, 'r') as f:
        for line in tqdm.tqdm(f, total=text_vocab_size):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.", len(model)," words loaded!")
    return model

if word_embedding == 'word2vec':
    # word2vec
    word2vecBinFile = os.path.join(WORD2VEC_BIN_SAVED_DIR,
        'GoogleNews-vectors-negative300.bin')
    word2vec = KeyedVectors.load_word2vec_format(word2vecBinFile, binary=True)
elif word_embedding == 'fastText':
    # fasttext
    fasttextBinFile = os.path.join(FASTTEXT_BIN_SAVED_DIR,
        'wiki.en.vec')
    fasttext = KeyedVectors.load_word2vec_format(fasttextBinFile)
elif word_embedding == 'glove':
    # GloVe
    gloveBinFile = os.path.join(GLOVE_BIN_SAVED_DIR,
        'glove.6B.300d.txt')
    glove = load_text_model(gloveBinFile, 400000)
elif word_embedding == 'ewpk':
    # Eigenwords with Prior Knowledge
    ewpkBinFile = os.path.join(EWPK_BIN_SAVED_DIR,
        'Eigenwords_Wiki5_alpha0.5_WordNetPriorKnowledge.txt')
    ewpk = load_text_model(ewpkBinFile, 200000)


########################################
# Make embedding matrices
########################################

if word_embedding == 'word2vec':
    if dataset == 'lrw':
        # LRW
        lrw_embedding_matrix_word2vec = np.zeros((LRW_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(LRW_VOCAB):
            lrw_embedding_matrix_word2vec[i] = word2vec.word_vec(word)
    elif dataset == 'grid':
        # GRIDcorpus
        grid_embedding_matrix_word2vec = np.zeros((GRID_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(GRID_VOCAB):
            grid_embedding_matrix_word2vec[i] = word2vec.word_vec(word)
elif word_embedding == 'fasttext':
    if dataset == 'lrw':
        # LRW
        lrw_embedding_matrix_fasttext = np.zeros((LRW_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(LRW_VOCAB):
            lrw_embedding_matrix_fasttext[i] = fasttext.word_vec(word)
    elif dataset == 'grid':
        # GRIDcorpus
        grid_embedding_matrix_fasttext = np.zeros((GRID_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(GRID_VOCAB):
            grid_embedding_matrix_fasttext[i] = fasttext.word_vec(word)
elif word_embedding == 'glove':
    if dataset == 'lrw':
        # LRW
        lrw_embedding_matrix_glove = np.zeros((LRW_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(LRW_VOCAB):
            lrw_embedding_matrix_glove[i] = glove[word]
    elif dataset == 'grid':
        # GRIDcorpus
        grid_embedding_matrix_glove = np.zeros((GRID_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(GRID_VOCAB):
            grid_embedding_matrix_glove[i] = glove[word]
elif word_embedding == 'ewpk':
    if dataset == 'lrw':
        # LRW
        lrw_embedding_matrix_glove = np.zeros((LRW_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(LRW_VOCAB):
            ewpk_embedding_matrix_ewpk[i] = ewpk[word]
    elif dataset == 'grid':
        # GRIDcorpus
        grid_embedding_matrix_ewpk = np.zeros((GRID_VOCAB_SIZE, EMBEDDING_DIM))
        for i, word in enumerate(GRID_VOCAB):
            grid_embedding_matrix_ewpk[i] = ewpk[word]


########################################
# Save embedding matrices
########################################

if word_embedding == 'word2vec':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_word2vec", lrw_embedding_matrix_word2vec)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_word2vec", grid_embedding_matrix_word2vec)
elif word_embedding == 'fasttext':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_fasttext", lrw_embedding_matrix_fasttext)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_fasttext", grid_embedding_matrix_fasttext)
elif word_embedding == 'glove':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_glove", lrw_embedding_matrix_glove)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_glove", grid_embedding_matrix_glove)
elif word_embedding == 'ewpk':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_ewpk", lrw_embedding_matrix_ewpk)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_ewpk", grid_embedding_matrix_ewpk)

# ########################################
# # Check for words being in word2vec
# ########################################

# LRW

# # lrw_vocab in word2vec
# # All 500 words in lrw_vocab are present in word2vec vocab
# for word in LRW_VOCAB:
#     if word not in word2vec.vocab:
#         print(word)

# # lrw_vocab in fasttext
# # All 500 words in lrw_vocab are present in word2vec vocab
# for word in LRW_VOCAB:
#     if word not in fasttext.vocab:
#         print(word)

# # lrw_vocab in glove
# # All 500 words in lrw_vocab are present in word2vec vocab
# for word in LRW_VOCAB:
#     if word not in glove.vocab:
#         print(word)

# # lrw_vocab in ewpk
# # All 500 words in lrw_vocab are present in word2vec vocab
# for word in LRW_VOCAB:
#     if word not in ewpk.vocab:
#         print(word)

# GRID

# # grid_vocab in word2vec
# # 50 of 51 words in grid_vocab are present in word2vec vocab
# # *****'a' is missing*****
# for word in GRID_VOCAB:
#     if word not in word2vec.vocab:
#         print(word)
#         grid_vocab.remove(word)

# # grid_vocab in fasttext
# for word in GRID_VOCAB:
#     if word not in fasttext.vocab:
#         print(word)
#         grid_vocab.remove(word)

# # grid_vocab in glove
# for word in GRID_VOCAB:
#     if word not in glove.vocab:
#         print(word)
#         grid_vocab.remove(word)


# # grid_vocab in ewpk
# ewpk_vocab = ewpk.keys()
# for word in GRID_VOCAB:
#     if word not in ewpk_vocab:
#         print(word)
#         grid_vocab.remove(word)


