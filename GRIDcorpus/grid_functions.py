import cv2
import glob
import imageio
import math
import numpy as np
import os
import tqdm

from grid_params import *
from lipreader_params import *
# from LSTM_lipreader_function import *
from zsl_functions import *

from skimage.transform import resize

#############################################################
# MAKE FEATURES AND ONE_HOT_WORDS
#############################################################


def make_GRIDcorpus_features_and_one_hot_words(dirs,
                                               word_numbers,
                                               word_idx,
                                               lipreaderEncoder,
                                               lipreader_encoded_dim=LIPREADER_ENCODED_DIM):
    features = np.zeros((len(dirs), lipreader_encoded_dim))
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
        features[i] = lipreaderEncoder.predict(wordImages)
        # MAKE ONE HOT WORDS
        one_hot_words[i][wordIndex] = 1
    # Return
    return features, one_hot_words


#############################################################
# MAKE FEATURES AND ONE_HOT_WORDS FROM SYNCNET
#############################################################

SHAPE_PREDICTOR_PATH = '/home/voletiv/GitHubRepos/lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat'

FACIAL_LANDMARKS_IDXS = dict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

MOUTH_SHAPE_FROM = FACIAL_LANDMARKS_IDXS["mouth"][0]
MOUTH_SHAPE_TO = FACIAL_LANDMARKS_IDXS["mouth"][1]

MOUTH_TO_FACE_RATIO = .65


def make_GRIDcorpus_features_and_one_hot_words_using_syncnet(dirs,
                                                             word_numbers,
                                                             word_idx,
                                                             lipreaderEncoder):
    features = np.zeros((len(dirs), 128))
    one_hot_words = np.zeros((len(dirs), GRID_VOCAB_SIZE))
    detector, predictor = load_detector_and_predictor()
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
        wordEndFrame = math.ceil(int(wordTimeData[wordNum].split(' ')[
                                 1]) / maxClipDuration * FRAMES_PER_VIDEO)
        wordMiddleFrame = (wordStartFrame + wordEndFrame) // 2
        wordStartFrame = wordMiddleFrame - 2
        wordEndFrame = wordMiddleFrame + 2
        # All mouth file names of video
        frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Frame*.jpg')))
        # Note the file names of the word
        wordMouthFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, 112, 112, 5))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles):
            wordImages[0, :, :, f] = load_mouth_from_frame(wordMouthFrame)
        # MAKE FEATURES
        features[i] = lipreaderEncoder.predict(wordImages)
        # MAKE ONE HOT WORDS
        one_hot_words[i][wordIndex] = 1
    # Return
    return features, one_hot_words


def load_mouth_from_frame(wordMouthFrame):
    frameImage = imageio.imread(wordMouthFrame)
    face = detector(frame, 1)[0]
    shape = predictor(frame, face)
    mouthCoords = np.array([[shape.part(i).x, shape.part(i).y]
                            for i in range(MOUTH_SHAPE_FROM, MOUTH_SHAPE_TO)])
    mouthRect = (np.min(mouthCoords[:, 0]), np.min(mouthCoords[:, 1]),
         np.max(mouthCoords[:, 0]) - np.min(mouthCoords[:, 0]),
         np.max(mouthCoords[:, 1]) - np.min(mouthCoords[:, 1]))
    mouthRect = make_rect_shape_square(mouthRect)
    expandedMouthRect = expand_rect(mouthRect,
        scale=(MOUTH_TO_FACE_RATIO * face.width() / mouthRect[2]),
        frame_shape=(frame.shape[0], frame.shape[1]))
    resizedMouthImage = resize(frame[expandedMouthRect[1]:expandedMouthRect[1] + expandedMouthRect[3],
                                     expandedMouthRect[0]:expandedMouthRect[0] + expandedMouthRect[2]],
                               (112, 112))
    return resizedMouthImage


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def make_rect_shape_square(rect):
    # Rect: (x, y, w, h)
    # If width > height
    if rect[2] > rect[3]:
        rect = (rect[0], int(rect[1] + rect[3] / 2 - rect[2] / 2),
                rect[2], rect[2])
    # Else (height > width)
    else:
        rect = (int(rect[0] + rect[2] / 2 - rect[3] / 2), rect[1],
                rect[3], rect[3])
    # Return
    return rect


def expand_rect(rect, scale=1.5, frame_shape=(256, 256)):
    # Rect: (x, y, w, h)
    w = int(rect[2] * scale)
    h = int(rect[3] * scale)
    x = max(0, min(frame_shape[1] - w, rect[0] - int((w - rect[2]) / 2)))
    y = max(0, min(frame_shape[0] - h, rect[1] - int((h - rect[3]) / 2)))
    return (x, y, w, h)


#############################################################
# LOAD SPEAKER_DIRS, WORD_NUMBERS, WORDS
#############################################################


def load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(
    speakers_list=TRAIN_VAL_SPEAKERS_LIST,
    grid_data_dir=GRID_DATA_DIR
):
    # TRAIN AND VAL
    all_dirs = []
    all_word_numbers = []
    all_wordidx = []
    # For each speaker
    for speaker in tqdm.tqdm(sorted((speakers_list))):
        speaker_dir = os.path.join(
            grid_data_dir, 's' + '{0:02d}'.format(speaker))
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
                if words[word_num] != 'a':
                    all_wordidx.append(GRID_VOCAB.index(words[word_num]))
                else:
                    all_wordidx.append(-1)
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
