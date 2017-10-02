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
# LEARN V AND CALCULATE ACCURACIES
#############################################################


def learn_v_and_calc_accs(train_num_of_words_list, word_to_attr_matrix,
                          train_val_features, train_val_one_hot_words,
                          si_features=None, si_one_hot_words=None,
                          optG=1e-6, optL=1e-3):
    pred_Vs = []
    train_accs = []
    test_accs = []
    si_in_vocab_accs = []
    si_oov_accs = []
    si_accs = []

    # For each value of number of training classes
    for train_num_of_words in train_num_of_words_list:

        ########################################
        # Split into train and test (OOV) data
        ########################################

        # Choose words for training
        training_words_idx = choose_words_for_training(
            train_num_of_words, GRID_VOCAB_SIZE)

        # Split train_val into training and testing
        train_features, train_one_hot_words, test_features, test_one_hot_words \
            = split_data_into_in_vocab_and_oov(training_words_idx,
                                               train_val_features, train_val_one_hot_words)

        # Split si into training and testing
        if si_features is not None and si_one_hot_words is not None:
            si_in_vocab_features, si_in_vocab_one_hot_words, \
                si_oov_features, si_oov_one_hot_words \
                = split_data_into_in_vocab_and_oov(training_words_idx,
                                                   si_features, si_one_hot_words)

        ########################################
        # Split embedding matrix into train and test (OOV) words
        ########################################

        training_word_to_attr_matrix, oov_word_to_attr_matrix \
            = split_embedding_matrix_into_in_vocab_and_oov(
                training_words_idx, word_to_attr_matrix)

        ########################################
        # EMBARRASSINGLY SIMPLE LEARNING
        ########################################
        # predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
        # === dxa = (dxd) . dxm . mxz . zxa . (axa)
        pred_V = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
            train_features.T, train_features)
            + optG * np.eye(train_features.shape[1])),
            train_features.T), train_one_hot_words), training_word_to_attr_matrix),
            np.linalg.inv(np.dot(training_word_to_attr_matrix.T,
                                 training_word_to_attr_matrix)
                          + optL * np.eye(training_word_to_attr_matrix.shape[1])))

        pred_Vs.append(pred_V)

        ########################################
        # ACCURACY CALCULATION
        ########################################

        # Train Acc
        y_train_preds = np.argmax(
            np.dot(np.dot(train_features, pred_V), training_word_to_attr_matrix.T), axis=1)
        train_accs.append(np.sum(y_train_preds == np.argmax(
            train_one_hot_words, axis=1)) / len(train_one_hot_words))

        # Test Acc
        y_test_preds = np.argmax(
            np.dot(np.dot(test_features, pred_V), oov_word_to_attr_matrix.T), axis=1)
        test_accs.append(np.sum(y_test_preds == np.argmax(
            test_one_hot_words, axis=1)) / len(test_one_hot_words))

        if si_features is not None and si_one_hot_words is not None:
            # SI in vocab Acc
            y_si_in_vocab_preds = np.argmax(
                np.dot(np.dot(si_in_vocab_features, pred_V), training_word_to_attr_matrix.T), axis=1)
            si_in_vocab_accs.append(np.sum(y_si_in_vocab_preds == np.argmax(
                si_in_vocab_one_hot_words, axis=1)) / len(si_in_vocab_one_hot_words))

            # SI OOV Acc
            y_si_oov_preds = np.argmax(
                np.dot(np.dot(si_oov_features, pred_V), oov_word_to_attr_matrix.T), axis=1)
            si_oov_accs.append(np.sum(y_si_oov_preds == np.argmax(
                si_oov_one_hot_words, axis=1)) / len(si_oov_one_hot_words))

            # SI Acc
            y_si_preds = np.append(y_si_in_vocab_preds, y_si_oov_preds)
            si_accs.append(np.sum(y_si_preds == np.append(np.argmax(
                si_in_vocab_one_hot_words, axis=1), np.argmax(si_oov_one_hot_words, axis=1))) / len(y_si_preds))

    return pred_Vs, train_accs, test_accs, si_in_vocab_accs, si_oov_accs, si_accs

#############################################################
# CHOOSE WORDS FOR TRAINING, REST FOR OOV TESTING
#############################################################


def choose_words_for_training(train_num_of_words, vocab_size=GRID_VOCAB_SIZE):
    # Choose words to keep in training data - training words
    np.random.seed(29)
    training_words_idx = np.sort(np.random.choice(
        vocab_size, train_num_of_words, replace=False))
    return training_words_idx

#############################################################
# SPLIT DATA INTO IN_VOCAB AND OOV
#############################################################


def split_data_into_in_vocab_and_oov(training_words_idx, features, one_hot_words):

    oov_words_idx = np.delete(np.arange(one_hot_words.shape[1]), training_words_idx)

    # Choose those rows in data that contain training words
    in_vocab_data_idx = np.array([i for i in range(
        len(one_hot_words)) if np.argmax(one_hot_words[i]) in training_words_idx])

    # Make the rest of the rows as testing data
    oov_data_idx = np.delete(
        np.arange(len(one_hot_words)), in_vocab_data_idx)

    # IN_VOCAB
    in_vocab_features = features[in_vocab_data_idx]
    in_vocab_one_hot_words = one_hot_words[in_vocab_data_idx][:, training_words_idx]

    # (SPEAKER-DEPENDENT) TEST DATA
    oov_features = features[oov_data_idx]
    oov_one_hot_words = one_hot_words[oov_data_idx][:, oov_words_idx]

    return in_vocab_features, in_vocab_one_hot_words, oov_features, oov_one_hot_words


#############################################################
# SPLIT EMBEDDING MATRIX INTO IN_VOCAB AND OOV
#############################################################


def split_embedding_matrix_into_in_vocab_and_oov(training_words_idx,
                                                 word_to_attr_matrix):
    training_word_to_attr_matrix = word_to_attr_matrix[training_words_idx]
    oov_word_to_attr_matrix = word_to_attr_matrix[np.delete(
        np.arange(len(word_to_attr_matrix)), training_words_idx)]
    return training_word_to_attr_matrix, oov_word_to_attr_matrix


#############################################################
# MAKE FEATURES AND ATTRIBUTES
#############################################################


def make_features_and_one_hot_words(dirs,
                                    word_numbers,
                                    word_idx,
                                    LSTMLipreaderEncoder):
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
