import cv2
import glob
import math
import numpy as np
import os
import tqdm

from common_functions import *
from grid_params import *
from lipreader_params import *
from LSTM_lipreader_function import *

#############################################################
# MAKE FEATURES AND ONE_HOT_WORDS
#############################################################


def make_GRIDcorpus_features_and_one_hot_words(dirs,
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
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(
    speakers_list=TRAIN_VAL_SPEAKERS_LIST
):
    # TRAIN AND VAL
    all_dirs = []
    all_word_numbers = []
    all_words = []
    # For each speaker
    for speaker in tqdm.tqdm(sorted((speakers_list))):
        speaker_dir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
        # List of all videos for each speaker
        vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))
        # Append training directories
        for vid_dir in vid_dirs_list:
            # Words
            align_file = vid_dir[:-1] + '.align'
            words = []
            with open(align_file) as f:
                for line in f:
                    if 'sil' not in line and 'sp' not in line:
                        words.append(line.rstrip().split()[-1])
            # Append
            for word_num in range(WORDS_PER_VIDEO):
                # print(wordNum)
                all_dirs.append(vid_dir)
                all_word_numbers.append(word_num)
                all_wordidx.append(GRID_VOCAB.index(words[word_num]))
    # Return
    return np.array(all_dirs), np.array(all_word_numbers), np.array(all_wordidx)


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
