import glob
import numpy as np
import os
import tqdm

from grid_params import *
from LSTM_lipreader_function import *


#############################################################
# MAKE TRAINING AND TESTING DATA
#############################################################


def make_train_test_siInV_siOOV_data(train_num_of_words):

    ########################################
    # Read GRIDcorpus directories, etc.
    ########################################

    train_val_dirs, train_val_word_numbers, train_val_words, train_val_word_idx, \
        si_dirs, si_word_numbers, si_words, si_word_idx \
        = get_train_val_si_dirs_wordnumbers_words_wordidx()

    ########################################
    # Assign Training and Testing Directories
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
    training_data_idx = np.array([i for i in range(
        len(train_val_words)) if train_val_word_idx[i] in training_word_idx])

    # Choose the rest of the rows as testing data
    testing_data_idx = np.delete(
        np.arange(len(train_val_words)), training_data_idx)

    # Choose those rows in data that contain training words
    si_in_vocab_data_idx = np.array(
        [i for i in range(len(si_words)) if si_word_idx[i] in training_word_idx])

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

    return training_dirs, training_word_numbers, training_words, \
        training_word_idx, testing_dirs, testing_word_numbers, testing_words, \
        testing_word_idx, si_in_vocab_dirs, si_in_vocab_word_numbers, \
        si_in_vocab_words, si_in_vocab_word_idx, si_oov_dirs, \
        si_oov_word_numbers, si_oov_words, si_oov_word_idx


def get_train_val_si_dirs_wordnumbers_words_wordidx(
    trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
    siList=[13, 14]
    ):

    ########################################
    # Read GRIDcorpus directories
    ########################################

    train_val_dirs, train_val_word_numbers, train_val_words, \
        si_dirs, si_word_numbers, si_words \
        = load_speakerdirs_wordnums_words_lists(trainValSpeakersList, siList)

    train_val_dirs = np.array(train_val_dirs)
    train_val_word_numbers = np.array(train_val_word_numbers)
    train_val_words = np.array(train_val_words)

    si_dirs = np.array(si_dirs)
    si_word_numbers = np.array(si_word_numbers)
    si_words = np.array(si_words)

    ########################################
    # Make Word Idx
    # - map words to their index in vocab
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
    # Remove rows corresponding to
    ## words not in vocab ('a')
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

    return train_val_dirs, train_val_word_numbers, train_val_words, \
        train_val_word_idx, si_dirs, si_word_numbers, si_words, si_word_idx


#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_speakerdirs_wordnums_words_lists(trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
                                          siList=[13, 14]):
    # TRAIN AND VAL
    trainValDirs = []
    trainValWordNumbers = []
    trainValWords = []
    # For each speaker
    for speaker in tqdm.tqdm(sorted((trainValSpeakersList))):
        speakerDir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
        # print(speakerDir)
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        # Append training directories
        for vidDir in vidDirs:
            # print(vidDir)
            # Words
            alignFile = vidDir[:-1] + '.align'
            words = []
            with open(alignFile) as f:
                for line in f:
                    if 'sil' not in line and 'sp' not in line:
                        words.append(line.rstrip().split()[-1])
            # Append
            for wordNum in range(WORDS_PER_VIDEO):
                # print(wordNum)
                trainValDirs.append(vidDir)
                trainValWordNumbers.append(wordNum)
                trainValWords.append(words[wordNum])
    # SPEAKER INDEPENDENT
    siDirs = []
    siWordNumbers = []
    siWords = []
    # For each speaker
    for speaker in tqdm.tqdm(sorted((siList))):
        speakerDir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
        # print(speakerDir)
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        # Append training directories
        for vidDir in vidDirs:
            # print(vidDir)
            # Words
            alignFile = vidDir[:-1] + '.align'
            words = []
            with open(alignFile) as f:
                for line in f:
                    if 'sil' not in line and 'sp' not in line:
                        words.append(line.rstrip().split()[-1])
            # Append
            for wordNum in range(WORDS_PER_VIDEO):
                # print(wordNum)
                siDirs.append(vidDir)
                siWordNumbers.append(wordNum)
                siWords.append(words[wordNum])
    # Return
    return trainValDirs, trainValWordNumbers, trainValWords, \
        siDirs, siWordNumbers, siWords

#############################################################
# LOAD LSTM LIP READER MODEL
#############################################################


def load_LSTM_lipreader_encoder():
    LSTMLipReaderModel, LSTMLipreaderEncoder, fileNamePre = make_LSTM_lipreader_model()
    LSTMLipReaderModel.load_weights(os.path.join(
        LIPREADER_DIR, 'LSTMLipReader-revSeq-Mask-LSTMh256-LSTMactivtanh-depth2-enc64-encodedActivrelu-Adam-1e-03-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub-epoch078-tl0.2384-ta0.9224-vl0.5184-va0.8503-sil4.9063-sia0.2393.hdf5'))
    return LSTMLipreaderEncoder


def make_LSTM_lipreader_model():
    useMask = True
    hiddenDim = 256
    depth = 2
    LSTMactiv = 'tanh'
    encodedDim = 64
    encodedActiv = 'relu'
    optimizer = 'adam'
    lr = 1e-3
    # Make model
    LSTMLipReaderModel, LSTMEncoder, fileNamePre \
        = LSTM_lipreader(useMask=useMask, hiddenDim=hiddenDim, depth=depth,
                         LSTMactiv=LSTMactiv, encodedDim=encodedDim,
                         encodedActiv=encodedActiv, optimizer=optimizer, lr=lr)
    return LSTMLipReaderModel, LSTMEncoder, fileNamePre
