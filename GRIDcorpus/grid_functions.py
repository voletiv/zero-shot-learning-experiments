import glob
import numpy as np
import os
import tqdm

from grid_params import *

#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_speakerdirs_wordnums_words_lists(trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
                                          siList=[13, 14]):
    # TRAIN AND VAL
    trainDirs = []
    trainWordNumbers = []
    trainWords = []
    valDirs = []
    valWordNumbers = []
    valWords = []
    # For each speaker
    for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
        speakerDir = os.path.join(GRID_DIR, 's' + '{0:02d}'.format(speaker))
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        # Append training directories
        for vidDir in vidDirs:
            # Words
            alignFile = vidDir[:-1] + '.align'
            words = []
            with open(alignFile) as f:
                for line in f:
                    if 'sil' not in line and 'sp' not in line:
                        words.append(line.rstrip().split()[-1])
            # Append
            for wordNum in range(WORDS_PER_VIDEO):
                trainDirs.append(vidDir)
                trainWordNumbers.append(wordNum)
                trainWords.append(words[wordNum])

#############################################################
# LSTM LIP READER MODEL
#############################################################


def load_LSTM_lipreader_and_encoder():
    LSTMLipReaderModel, LSTMEncoder, fileNamePre = make_LSTM_lipreader_model()
    LSTMLipReaderModel.load_weights(os.path.join(
        LSTMModelSavedDir,
        'LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-10PercentSelfTraining-LRthresh0.90-iter00-epoch079-tl1.1377-ta0.6460-vl1.5886-va0.5360-sil3.9002-sia0.2181.hdf5'))
    return LSTMLipReaderModel, LSTMEncoder


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

#############################################################
# LOAD IMAGE DIRS AND WORD NUMBERS
#############################################################


def load_image_dirs_and_word_numbers(trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
                                     valSplit=0.1,
                                     siList=[13, 14]):
    # TRAIN AND VAL
    trainDirs = []
    trainWordNumbers = []
    valDirs = []
    valWordNumbers = []
    np.random.seed(29)

    # For each speaker
    for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
        speakerDir = os.path.join(GRID_DIR, 's' + '{0:02d}'.format(speaker))
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        totalNumOfImages = len(vidDirs)
        # To shuffle directories before splitting into train and validate
        fullListIdx = list(range(totalNumOfImages))
        np.random.shuffle(fullListIdx)
        # Append training directories
        for i in fullListIdx[:int((1 - valSplit) * totalNumOfImages)]:
            for j in range(wordsPerVideo):
                trainDirs.append(vidDirs[i])
                trainWordNumbers.append(j)
        # Append val directories
        for i in fullListIdx[int((1 - valSplit) * totalNumOfImages):]:
            for j in range(wordsPerVideo):
                valDirs.append(vidDirs[i])
                valWordNumbers.append(j)

    # Numbers
    print("No. of training words: " + str(len(trainDirs)))
    print("No. of val words: " + str(len(valDirs)))

    # SPEAKER INDEPENDENT
    siDirs = []
    siWordNumbers = []
    for speaker in sorted(tqdm.tqdm(siList)):
        speakerDir = os.path.join(GRID_DIR, 's' + '{0:02d}'.format(speaker))
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        for i in fullListIdx:
            for j in range(wordsPerVideo):
                siDirs.append(vidDirs[i])
                siWordNumbers.append(j)

    # Numbers
    print("No. of speaker-independent words: " + str(len(siDirs)))

    # Return
    return trainDirs, trainWordNumbers, valDirs, valWordNumbers, \
        siDirs, siWordNumbers
