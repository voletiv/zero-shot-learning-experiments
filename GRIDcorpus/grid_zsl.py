from grid_functions import *
from grid_params import *

########################################
## ZSL equations
########################################

# minimise L(X^T * V * S, Y) + Ω(V)
# predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))

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

########################################
## Read GRIDcorpus directories
########################################

trainDirs, trainWordNumbers, trainWords, \
        valDirs, valWordNumbers, valWords, \
        siDirs, siWordNumbers, siWords \
    = load_speakerdirs_wordnums_words_lists(trainValSpeakersList \
                                            = [1, 2, 3, 4, 5, 6, 7, 10],
                                        siList = [13, 14])

########################################
## Vars
########################################

# Number of training classes
z = [10, 20, 30, 40]

# Number of test classes
t = 50 - z

# Number of atributes === word2vec dim
attr_dim = matrix_class_to_attr.shape[1]


########################################
## Assign Train, Val and Test Directories
########################################


