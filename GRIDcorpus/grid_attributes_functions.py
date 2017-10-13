# GET GRIDCORPUS ATTRIBUTES

import dlib
import glob
import math
import numpy as np
import tqdm

from grid_functions import *

def make_LSTMlipreader_predictions(lipreader_pred_word_idx,
                                   lipreader_preds_correct_or_wrong,
                                   word_durations,
                                   dirs,
                                   word_numbers,
                                   word_idx,
                                   lipreader,
                                   grid_vocab=GRID_VOCAB_FULL,
                                   startNum=0):
    # dirs = train_val_dirs
    # word_numbers = train_val_word_numbers
    # word_idx = train_val_word_idx
    detector, predictor = load_detector_and_predictor()
    # For each data point
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
        if i < startNum:
            continue
        # GET SEQUENCE OF FRAMES
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
        # Word duration
        word_durations[i] = wordEndFrame - wordStartFrame + 1
        # All mouth file names of video
        mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[0][-f - 1] = np.reshape(cv2.imread(wordMouthFrame,
                                                          0) / 255., (NUM_OF_MOUTH_PIXELS,))
        # MAKE PREDICTION
        lipreader_pred_word_idx[i] = np.argmax(lipreader.predict(wordImages))
        lipreader_preds_correct_or_wrong[i] = lipreader_pred_word_idx[i] == wordIndex


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


def read_txt_files_line_by_line(word='Estimating', stop_idx=None):
    while 1:
        idx = -1
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_00.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_01.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_02.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_03.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_04.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line
        with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val.txt', 'r') as f:
            for line in f:
                if word in line:
                    idx += 1
                    # train_val_head_pose_lines.append(line)
                    # if idx == stop_idx:
                    #     return line
                    yield line


def read_txt_files_line_range(word='Estimating', start_idx=0, stop_idx=3):
    lines = []
    idx = -1
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_00.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_01.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_02.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_03.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val_04.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)
    with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val.txt', 'r') as f:
        for line in f:
            if word in line:
                idx += 1
                # train_val_head_pose_lines.append(line)
                if idx >= stop_idx:
                    return lines
                if idx >= start_idx:
                    lines.append(line)




