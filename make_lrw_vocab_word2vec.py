# http://mccormickml.com/2016/04/12/googles-pretrained-word2vec-model-in-python/
# https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings

from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess

########################################
## Params
########################################

WORD2VEC_BIN_SAVED_DIR = '/media/voletiv/01D2BF774AC76280/Word2Vec'

LRW_VOCAB_LIST_FILE = 'lrw_vocabulary.txt'

LRW_VOCAB_SIZE = 500

EMBEDDING_DIM = 300

########################################
## EMBEDDING MATRIX
# Make the embedding matrix (words -> word vectors)
# for the vocabulary of LRW dataset
########################################

########################################
# Load word2vec binary file
########################################

word2vecBinFile = os.path.join(WORD2VEC_BIN_SAVED_DIR,
    'GoogleNews-vectors-negative300.bin')

word2vec = KeyedVectors.load_word2vec_format(word2vecBinFile, binary=True)

########################################
# Load vocabulary
########################################

lrw_vocab = []

with open(LRW_VOCAB_LIST_FILE) as f:
    for line in f:
        word = line[:-1].lower()
        # LABOUR is not in word2vec, LABOR is
        if word == 'labour':
            word = 'labor'
        lrw_vocab.append(word)

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
