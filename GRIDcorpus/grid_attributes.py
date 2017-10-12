# GET GRIDCORPUS ATTRIBUTES

from grid_attributes_functions import *

attributes = {}

########################################
# ATTRIBUTES
########################################

# Speaker identity

# Duration of word

# Presence of bilabials in word

# Mouth-to-face ratio

# Pose

# Range of mouth movement??


# Audio freq - L, M, H

########################################
# Get FULL Data
########################################

train_val_dirs, train_val_word_numbers, train_val_word_idx, \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(TRAIN_VAL_SPEAKERS_LIST, GRID_DATA_DIR, GRID_VOCAB_FULL)

si_dirs, si_word_numbers, si_word_idx \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(SI_SPEAKERS_LIST, GRID_DATA_DIR, GRID_VOCAB_FULL)

########################################
# ATTR - speaker identity
########################################

train_val_speaker_identity = []
for speaker in tqdm.tqdm(sorted((TRAIN_VAL_SPEAKERS_LIST))):
    speaker_dir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
    # List of all videos for each speaker
    vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))
    # Append training directories
    for vid_dir in vid_dirs_list:
        for word_num in range(WORDS_PER_VIDEO):
            train_val_speaker_identity.append(speaker)

train_val_speaker_identity = np.array(train_val_speaker_identity, dtype=int)

attributes['train_val_speaker_identity'] = train_val_speaker_identity
np.save("attributes", attributes)

si_speaker_identity = []
for speaker in tqdm.tqdm(sorted((SI_SPEAKERS_LIST))):
    speaker_dir = os.path.join(
            GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
    # List of all videos for each speaker
    vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))
    # Append training directories
    for vid_dir in vid_dirs_list:
        for word_num in range(WORDS_PER_VIDEO):
            si_speaker_identity.append(speaker)

si_speaker_identity = np.array(si_speaker_identity, dtype=int)

attributes['si_speaker_identity'] = si_speaker_identity
np.save("attributes", attributes)


########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

########################################
# LipReader predictions, word durations
########################################

lipreader_preds = {}

train_val_lipreader_pred_word_idx = np.zeros(len(train_val_dirs), dtype=int)
train_val_lipreader_preds_correct_or_wrong = np.zeros(len(train_val_dirs), dtype=bool)
make_LSTMlipreader_predictions(train_val_lipreader_pred_word_idx,
                               train_val_lipreader_preds_correct_or_wrong,
                               train_val_dirs,
                               train_val_word_numbers,
                               train_val_word_idx,
                               LSTMLipreaderModel,
                               GRID_VOCAB_FULL,
                               0)

lipreader_preds['train_val_lipreader_pred_word_idx'] = train_val_lipreader_preds
lipreader_preds['train_val_lipreader_preds_correct_or_wrong'] = train_val_lipreader_preds_correct_or_wrong
np.save('lipreader_preds', lipreader_preds)


si_lipreader_pred_word_idx = np.zeros(len(si_dirs), dtype=int)
si_lipreader_preds_correct_or_wrong = np.zeros(len(si_dirs), dtype=bool)
make_LSTMlipreader_predictions(si_lipreader_pred_word_idx,
                               si_lipreader_preds_correct_or_wrong,
                               si_dirs,
                               si_word_numbers,
                               si_word_idx,
                               LSTMLipreaderModel,
                               GRID_VOCAB_FULL,
                               0)

lipreader_preds['si_lipreader_pred_word_idx'] = si_lipreader_pred_word_idx
lipreader_preds['si_lipreader_preds_correct_or_wrong'] = si_lipreader_preds_correct_or_wrong
np.save('lipreader_preds', lipreader_preds)


########################################
# ATTR - Word durations
########################################

train_val_word_durations = np.zeros(len(train_val_dirs), dtype=int)
for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(train_val_dirs, train_val_word_numbers, train_val_word_idx)), total=len(train_val_dirs)):
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
    train_val_word_durations[i] = wordEndFrame - wordStartFrame + 1

attributes['train_val_word_durations'] = train_val_word_durations
np.save("attributes", attributes)

si_word_durations = np.zeros(len(si_dirs), dtype=int)
for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(si_dirs, si_word_numbers, si_word_idx)), total=len(si_dirs)):
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
    si_word_durations[i] = wordEndFrame - wordStartFrame + 1

attributes['si_word_durations'] = si_word_durations
np.save("attributes", attributes)


########################################
# ATTR - Head pose estimations
########################################

# Creating fileNamesList.txt

# TRAIN_VAL
train_val_frame_file_names = []
for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(train_val_dirs, train_val_word_numbers, train_val_word_idx)), total=len(train_val_dirs)):
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
    frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
    # Note the file names of the word
    wordFrameFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
    # Loop
    for f, wordFrameFile in enumerate(wordFrameFiles):
        # File names
        train_val_frame_file_names.append(wordFrameFile.replace('Mouth', 'Frame'))

with open("frame_names_list_train_val.txt", 'w') as f:
    f.write('\n'.join(train_val_frame_file_names))

# SI
si_frame_file_names = []
for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(si_dirs, si_word_numbers, si_word_idx)), total=len(si_dirs)):
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
    frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
    # Note the file names of the word
    wordFrameFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
    # Loop
    for f, wordFrameFile in enumerate(wordFrameFiles):
        # File names
        si_frame_file_names.append(wordFrameFile.replace('Mouth', 'Frame'))

with open("frame_names_list_si.txt", 'w') as f:
    f.write('\n'.join(si_frame_file_names))

'''
cd /home/voletiv/GitHubRepos/gazr/build
./benchmark_head_pose_estimation_multiple_frames ../../lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat ../../zero-shot-learning-experiments/GRIDcorpus/frame_names_list_train_val.txt > head_poses_train_val.txt
./benchmark_head_pose_estimation_multiple_frames ../../lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat ../../zero-shot-learning-experiments/GRIDcorpus/frame_names_list_si.txt > head_poses_si.txt
'''

# TRAIN_VAL
train_val_head_poses = []
with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_train_val.txt', 'r') as f:
    for line in f:
        if 'Head pose' in line:
            train_val_head_poses.append(line)

attributes['train_val_head_poses'] = train_val_head_poses
np.save("attributes", attributes)

# SI
si_head_poses = []
with open('/home/voletiv/GitHubRepos/gazr/build/head_poses_si.txt', 'r') as f:
    for line in f:
        if 'Head pose' in line:
            si_head_poses.append(line)

attributes['si_head_poses'] = si_head_poses
np.save("attributes", attributes)






########################################
# Get ATTRIBUTES
########################################

train_val_dirs, train_val_word_numbers, train_val_word_idx, \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(TRAIN_VAL_SPEAKERS_LIST, GRID_DATA_DIR, GRID_VOCAB_ZSL)


def make_GRIDcorpus_features_and_one_hot_words_using_syncnet(dirs,
                                                             word_numbers,
                                                             word_idx,
                                                             lipreaderEncoder):
    features = np.zeros((len(dirs), 128))
    one_hot_words = np.zeros((len(dirs), GRID_VOCAB_SIZE))
    detector, predictor = load_detector_and_predictor()
    # For each data point
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
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
        wordEndFrame = math.ceil(int(wordTimeData[wordNum].split(' ')[
                                 1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # All frame file names of video
        frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Frame*.jpg')))
        # Note the file names of the word
        wordFrameFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
        # 



########################################
# BILABIALS
########################################

bilabials = {}
bilabials['a'] = 0
bilabials['again'] = 0
bilabials['at'] = 0
bilabials['b'] = 1
bilabials['bin'] = 1
bilabials['blue'] = 1
bilabials['by'] = 1
bilabials['c'] = 0
bilabials['d'] = 0
bilabials['e'] = 0
bilabials['eight'] = 0
bilabials['f'] = 0
bilabials['five'] = 0
bilabials['four'] = 0
bilabials['g'] = 0
bilabials['green'] = 0
bilabials['h'] = 0
bilabials['i'] = 0
bilabials['in'] = 0
bilabials['j'] = 0
bilabials['k'] = 0
bilabials['l'] = 0
bilabials['lay'] = 0
bilabials['m'] = 1
bilabials['n'] = 0
bilabials['nine'] = 0
bilabials['now'] = 0
bilabials['o'] = 0
bilabials['one'] = 0
bilabials['p'] = 1
bilabials['place'] = 1
bilabials['please'] = 1
bilabials['q'] = 0
bilabials['r'] = 0
bilabials['red'] = 0
bilabials['s'] = 0
bilabials['set'] = 0
bilabials['seven'] = 0
bilabials['six'] = 0
bilabials['soon'] = 0
bilabials['t'] = 0
bilabials['three'] = 0
bilabials['two'] = 0
bilabials['u'] = 0
bilabials['v'] = 0
bilabials['white'] = 0
bilabials['with'] = 0
bilabials['x'] = 0
bilabials['y'] = 0
bilabials['z'] = 0
bilabials['zero'] = 0
