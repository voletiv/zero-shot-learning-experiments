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

from gensim.models import KeyedVectors

from common_params import *
from GRIDcorpus.grid_params import *
from LRW.lrw_params import *

########################################
## Params
########################################

# Directory where the word2vec file
# 'GoogleNews-vectors-negative300.bin' is saved
WORD2VEC_BIN_SAVED_DIR = '/media/voletiv/01D2BF774AC76280/Word-Embeddings/Word2Vec'

# Directory where the fasttext file
# 'wiki.en.vec' is saved
FASTTEXT_BIN_SAVED_DIR = '/media/voletiv/01D2BF774AC76280/Word-Embeddings/FastText'

# Directory where the GloVe file
# 'glove.6B.300d.txt' is saved
GLOVE_BIN_SAVED_DIR = '/media/voletiv/01D2BF774AC76280/Word-Embeddings/GloVe'

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
wordEmbedding = 'glove'

dataset = 'grid'
# dataset = 'lrw'

########################################
# Load word2vec, fasttext binary files
########################################

if wordEmbedding == 'word2vec':
    # word2vec
    word2vecBinFile = os.path.join(WORD2VEC_BIN_SAVED_DIR,
        'GoogleNews-vectors-negative300.bin')
    word2vec = KeyedVectors.load_word2vec_format(word2vecBinFile, binary=True)
elif wordEmbedding == 'fastText':
    # fasttext
    fasttextBinFile = os.path.join(FASTTEXT_BIN_SAVED_DIR,
        'wiki.en.vec')
    fasttext = KeyedVectors.load_word2vec_format(fasttextBinFile)
elif wordEmbedding == 'glove':
    # GloVe
    gloveBinFile = os.path.join(GLOVE_BIN_SAVED_DIR,
        'glove.6B.300d.txt')
    def load_glove_model(gloveFile):
        print("Loading Glove Model")
        model = {}
        nOfLines = i+1
        with open(gloveFile, 'r') as f:
            for line in tqdm.tqdm(f, total=400000):
                splitLine = line.split()
                word = splitLine[0]
                embedding = np.array([float(val) for val in splitLine[1:]])
                model[word] = embedding
        print("Done.", len(model)," words loaded!")
        return model
    glove = load_glove_model(gloveBinFile)

########################################
# Make embedding matrices
########################################

if wordEmbedding == 'word2vec':
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
elif wordEmbedding == 'fasttext':
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
elif wordEmbedding == 'glove':
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


########################################
# Save embedding matrices
########################################

if wordEmbedding == 'word2vec':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_word2vec", lrw_embedding_matrix_word2vec)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_word2vec", grid_embedding_matrix_word2vec)
elif wordEmbedding == 'fasttext':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_fasttext", lrw_embedding_matrix_fasttext)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_fasttext", grid_embedding_matrix_fasttext)
elif wordEmbedding == 'glove':
    if dataset == 'lrw':
        np.save("lrw_embedding_matrix_glove", lrw_embedding_matrix_glove)
    elif dataset == 'grid':
        np.save("grid_embedding_matrix_glove", grid_embedding_matrix_glove)

# ########################################
# # Check for words being in word2vec
# ########################################

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

# # grid_vocab in word2vec
# # 50 of 51 words in grid_vocab are present in word2vec vocab
# # 'a' is missing
# for word in GRID_VOCAB:
#     if word not in word2vec.vocab:
#         print(word)
#         grid_vocab.remove(word)

# # grid_vocab in fasttext
# # 50 of 51 words in grid_vocab are present in word2vec vocab
# # 'a' is missing
# for word in GRID_VOCAB:
#     if word not in fasttext.vocab:
#         print(word)
#         grid_vocab.remove(word)

