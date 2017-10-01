import cv2
import glob
import math
import numpy as np
import os
import tqdm

from grid_params import *
from lipreader_params import *
from LSTM_lipreader_function import *

#############################################################
# MAKE TRAINING AND TESTING DATA
#############################################################


def make_train_test_siI_siOOV_data(train_num_of_words,
                                   train_val_word_idx, si_word_idx,
                                   train_val_features, train_val_one_hot_words,
                                   si_features, si_one_hot_words):

    ########################################
    # Assign Training and Testing Directories
    ########################################

    # Train data - speakers speaking certain words
    # Test data - same speakers speaking other words
    # SI_Test data - different speakers speaking other words

    all_words_idx = np.arange(GRID_VOCAB_SIZE)

    # Choose words to keep in training data - training words
    np.random.seed(29)
    training_words_idx = np.random.choice(all_words_idx, train_num_of_words)

    # Choose those rows in data that contain training words
    train_data_idx = np.array([i for i in range(
        len(train_val_word_idx)) if train_val_word_idx[i] in training_words_idx])

    # Make the rest of the rows as testing data
    test_data_idx = np.delete(
        np.arange(len(train_val_word_idx)), train_data_idx)

    # Choose those rows in data that contain training words
    si_in_vocab_data_idx = np.array(
        [i for i in range(len(si_word_idx)) if si_word_idx[i] in training_words_idx])

    # Make the rest of the rows as testing data
    si_oov_data_idx = np.delete(
        np.arange(len(si_word_idx)), si_in_vocab_data_idx)

    # TRAIN DATA
    train_features = train_val_features[train_data_idx]
    train_one_hot_words = train_val_one_hot_words[train_data_idx]

    # (SPEAKER-DEPENDENT) TEST DATA
    test_features = train_val_features[test_data_idx]
    test_one_hot_words = train_val_one_hot_words[test_data_idx]

    # SPEAKER-INDEPENDENT IN-VOCAB DATA
    si_in_vocab_features = si_features[si_in_vocab_data_idx]
    si_in_vocab_one_hot_words = si_one_hot_words[si_in_vocab_data_idx]

    # SPEAKER-INDEPENDENT OOV DATA
    si_oov_features = si_features[si_oov_data_idx]
    si_oov_one_hot_words = si_one_hot_words[si_oov_data_idx]

    return train_features, train_one_hot_words, \
        test_features, test_one_hot_words, \
        si_in_vocab_features, si_in_vocab_one_hot_words, \
        si_oov_features, si_oov_one_hot_words

#############################################################
# MAKE FEATURES AND ATTRIBUTES
#############################################################


def make_features_and_one_hot_words(dirs,
                                    word_numbers,
                                    word_idx,
                                    LSTMLipreaderEncoder,
                                    word_to_attr_matrix):
    features = np.zeros((len(dirs), LIPREADER_ENCODED_DIM))
    one_hot_words = np.zeros((len(dirs), GRID_VOCAB_SIZE))
    # For each data point
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
        # GET SEQUENCE OF MOUTH IMAGES
        # align file
        alignFile = vidDir[:-1] + '.align'
        # Word-Time data
        wordTimeData = open(alignFile).readlines()
        # Get the max time of the video
        maxClipDuration = float(wordTimeData[-1].split(' ')[1])
        # Remove Silent and Short Pauses
        for line in wordTimeData:
            if 'sil' in line or 'sp' in line:
                wordTimeData.remove(line)
        # Find the start and end frame for this word
        wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                    0]) / maxClipDuration * FRAMES_PER_VIDEO)
        wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                  1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # All mouth file names of video
        mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[
            wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[0][-f - 1] = np.reshape(cv2.imread(wordMouthFrame,
                                                          0) / 255., (NUM_OF_MOUTH_PIXELS,))
        # MAKE FEATURES
        features[i] = LSTMLipreaderEncoder.predict(wordImages)
        # MAKE ONE HOT WORDS
        one_hot_words[i][wordIndex] = 1
    # Return
    return features, one_hot_words

#############################################################
# GET DIRS, WORDNUMBERS, WORDIDX
#############################################################


def get_train_val_si_dirs_wordnumbers_wordidx(
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

    train_val_word_idx = -np.ones((len(train_val_words)), dtype=int)
    for i in range(len(train_val_words)):
        if train_val_words[i] in GRID_VOCAB:
            train_val_word_idx[i] = GRID_VOCAB.index(train_val_words[i])

    si_word_idx = -np.ones((len(si_words)), dtype=int)
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

    return train_val_dirs, train_val_word_numbers, train_val_word_idx, \
        si_dirs, si_word_numbers, si_word_idx


#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_speakerdirs_wordnums_words_lists(
    trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
    siList=[13, 14]
):
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


def load_LSTM_lipreader_and_encoder():
    LSTMLipReaderModel, LSTMLipreaderEncoder, _ = make_LSTM_lipreader_model()
    LSTMLipReaderModel.load_weights(LIPREADER_MODEL_FILE)
    return LSTMLipReaderModel, LSTMLipreaderEncoder


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
