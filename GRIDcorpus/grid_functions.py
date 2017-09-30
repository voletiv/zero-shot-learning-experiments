import glob
import numpy as np
import os
import tqdm

from grid_params import *
from LSTM_lipreader_function import *

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
        speakerDir = os.path.join(GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
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
        speakerDir = os.path.join(GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
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
    LSTMLipReaderModel.load_weights(os.path.join(LIPREADER_DIR, 'LSTMLipReader-revSeq-Mask-LSTMh256-LSTMactivtanh-depth2-enc64-encodedActivrelu-Adam-1e-03-GRIDcorpus-s0107-09-tMouth-valMouth-NOmeanSub-epoch078-tl0.2384-ta0.9224-vl0.5184-va0.8503-sil4.9063-sia0.2393.hdf5'))
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
