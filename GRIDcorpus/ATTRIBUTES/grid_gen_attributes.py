# GET GRIDCORPUS ATTRIBUTES

from grid_attributes_functions import *
from grid_functions import *

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
# BILABIALS
########################################

bilabials = {}
bilabial_words = {'b', 'bin', 'blue', 'by', 'place', 'please'}

train_val_bilabial_or_not = np.zeros(len(train_val_dirs), dtype=bool)
for i in range(len(train_val_word_idx)):
    train_val_bilabial_or_not[i] = GRID_VOCAB_FULL[train_val_word_idx[i]] in bilabial_words

attributes['train_val_bilabial_or_not'] = train_val_bilabial_or_not
np.save("grid_attributes_dict", attributes)

si_bilabial_or_not = np.zeros(len(si_dirs), dtype=bool)
for i in range(len(si_word_idx)):
    si_bilabial_or_not[i] = GRID_VOCAB_FULL[si_word_idx[i]] in bilabial_words

attributes['si_bilabial_or_not'] = si_bilabial_or_not
np.save("grid_attributes_dict", attributes)

########################################
# ATTR - speaker identity
########################################

# train_val_speaker_identity = []
# for speaker in tqdm.tqdm(sorted((TRAIN_VAL_SPEAKERS_LIST))):
#     speaker_dir = os.path.join(
#             GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
#     # List of all videos for each speaker
#     vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))
#     # Append training directories
#     for vid_dir in vid_dirs_list:
#         for word_num in range(WORDS_PER_VIDEO):
#             train_val_speaker_identity.append(speaker)

# train_val_speaker_identity = np.array(train_val_speaker_identity, dtype=int)

# attributes['train_val_speaker_identity'] = train_val_speaker_identity
# np.save("grid_attributes_dict", attributes)

# si_speaker_identity = []
# for speaker in tqdm.tqdm(sorted((SI_SPEAKERS_LIST))):
#     speaker_dir = os.path.join(
#             GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
#     # List of all videos for each speaker
#     vid_dirs_list = sorted(glob.glob(os.path.join(speaker_dir, '*/')))
#     # Append training directories
#     for vid_dir in vid_dirs_list:
#         for word_num in range(WORDS_PER_VIDEO):
#             si_speaker_identity.append(speaker)

# si_speaker_identity = np.array(si_speaker_identity, dtype=int)

# attributes['si_speaker_identity'] = si_speaker_identity
# np.save("grid_attributes_dict", attributes)

attributes = np.load('grid_attributes_dict.npy').item()

########################################
# SPEAKER MALE OR NOT
########################################

# # speaker_male_or_not
# male_speakers = {1, 2, 3, 5, 6, 10, 12, 13, 14}

# train_val_male_or_not = np.zeros(len(train_val_dirs), dtype=bool)
# for i in range(len(train_val_word_idx)):
#     train_val_male_or_not[i] = attributes['train_val_speaker_identity'][i] in male_speakers

# attributes['train_val_male_or_not'] = train_val_male_or_not
# np.save("grid_attributes_dict", attributes)

# si_male_or_not = np.zeros(len(si_dirs), dtype=bool)
# for i in range(len(si_word_idx)):
#     si_male_or_not[i] = attributes['si_speaker_identity'][i] in male_speakers

# attributes['si_male_or_not'] = si_male_or_not
# np.save("grid_attributes_dict", attributes)

########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

########################################
# LipReader predictions
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

lipreader_preds = np.load('lipreader_preds.npy').item()

########################################
# ATTR - Word durations
########################################

# train_val_word_metadata = {}

# train_val_word_metadata['startFrame'] = np.zeros(len(train_val_dirs), dtype=int)
# train_val_word_metadata['endFrame'] = np.zeros(len(train_val_dirs), dtype=int)
# train_val_word_metadata['wordDuration'] = np.zeros(len(train_val_dirs), dtype=int)
# for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(train_val_dirs, train_val_word_numbers, train_val_word_idx)), total=len(train_val_dirs)):
#     alignFile = vidDir[:-1] + '.align'
#     # Word-Time data
#     wordTimeData = open(alignFile).readlines()
#     # Get the max time of the video
#     maxClipDuration = float(wordTimeData[-1].split(' ')[1])
#     # Remove Silent and Short Pauses
#     for line in wordTimeData:
#         if 'sil' in line or 'sp' in line:
#             wordTimeData.remove(line)
#     # Find the start and end frame for this word
#     wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                                 0]) / maxClipDuration * FRAMES_PER_VIDEO)
#     wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                              1]) / maxClipDuration * FRAMES_PER_VIDEO)
#     train_val_word_metadata['startFrame'][i] = wordStartFrame
#     train_val_word_metadata['endFrame'][i] = wordEndFrame
#     train_val_word_metadata['wordDuration'][i] = wordEndFrame - wordStartFrame + 1

# attributes['train_val_word_metadata'] = train_val_word_metadata
# np.save("grid_attributes_dict", attributes)

# si_word_metadata = {}
# si_word_metadata['startFrame'] = np.zeros(len(si_dirs), dtype=int)
# si_word_metadata['endFrame'] = np.zeros(len(si_dirs), dtype=int)
# si_word_metadata['wordDuration'] = np.zeros(len(si_dirs), dtype=int)
# for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(si_dirs, si_word_numbers, si_word_idx)), total=len(si_dirs)):
#     alignFile = vidDir[:-1] + '.align'
#     # Word-Time data
#     wordTimeData = open(alignFile).readlines()
#     # Get the max time of the video
#     maxClipDuration = float(wordTimeData[-1].split(' ')[1])
#     # Remove Silent and Short Pauses
#     for line in wordTimeData:
#         if 'sil' in line or 'sp' in line:
#             wordTimeData.remove(line)
#     # Find the start and end frame for this word
#     wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                                 0]) / maxClipDuration * FRAMES_PER_VIDEO)
#     wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                              1]) / maxClipDuration * FRAMES_PER_VIDEO)
#     si_word_metadata['startFrame'][i] = wordStartFrame
#     si_word_metadata['endFrame'][i] = wordEndFrame
#     si_word_metadata['wordDuration'][i] = wordEndFrame - wordStartFrame + 1

# attributes['si_word_metadata'] = si_word_metadata
# np.save("grid_attributes_dict", attributes)

attributes = np.load('grid_attributes_dict.npy').item()

train_val_word_metadata = attributes['train_val_word_metadata']
train_val_word_durations = train_val_word_metadata['wordDuration']

si_word_metadata = attributes['si_word_metadata']
si_word_durations = si_word_metadata['wordDuration']

########################################
# ATTR - Head pose estimations
########################################

# # EXTRACT FRAMES FROM VIDEO USING FFMPEG

# for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(si_dirs, si_word_numbers, si_word_idx)), total=len(si_dirs)):
#     makeVideos = False
#     frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Frame*.jpg')))
#     if len(frameFiles) != 75:
#         makeVideos = True
#     elif frameFiles[-1].split('/')[-1].split('.')[0][-2:] == '75':
#         makeVideos = True
#     if makeVideos:
#         print("Extracting from", vidDir)
#         command = "ffmpeg -i " + vidDir[:-1] + ".mpg -y -an -qscale 0 -f image2 " + vidDir[:-1] + '/' +  vidDir.split('/')[-2] + "Frame%02d.jpg"
#         os.system(command)
#         print("Renaming")
#         for num in range(75):
#             command = "mv " + vidDir[:-1] + '/' +  vidDir.split('/')[-2] + "Frame{0:02d}.jpg".format(num+1) + " " +  vidDir[:-1] + '/' +  vidDir.split('/')[-2] + "Frame{0:02d}.jpg".format(num)
#             ret = os.system(command)

# # CREATING fileNamesList.txt

# # TRAIN_VAL
# train_val_frame_file_names = []
# for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(train_val_dirs, train_val_word_numbers, train_val_word_idx)), total=len(train_val_dirs)):
#     alignFile = vidDir[:-1] + '.align'
#     # Word-Time data
#     wordTimeData = open(alignFile).readlines()
#     # Get the max time of the video
#     maxClipDuration = float(wordTimeData[-1].split(' ')[1])
#     # Remove Silent and Short Pauses
#     for line in wordTimeData:
#         if 'sil' in line or 'sp' in line:
#             wordTimeData.remove(line)
#     # Find the start and end frame for this word
#     wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                                 0]) / maxClipDuration * FRAMES_PER_VIDEO)
#     wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                              1]) / maxClipDuration * FRAMES_PER_VIDEO)
#     # All mouth file names of video
#     frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
#     # Note the file names of the word
#     wordFrameFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
#     # Loop
#     for f, wordFrameFile in enumerate(wordFrameFiles):
#         # File names
#         train_val_frame_file_names.append(wordFrameFile.replace('Mouth', 'Frame'))

# with open("frame_names_list_train_val.txt", 'w') as f:
#     f.write('\n'.join(train_val_frame_file_names))

# # SI
# si_frame_file_names = []
# for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(si_dirs, si_word_numbers, si_word_idx)), total=len(si_dirs)):
#     alignFile = vidDir[:-1] + '.align'
#     # Word-Time data
#     wordTimeData = open(alignFile).readlines()
#     # Get the max time of the video
#     maxClipDuration = float(wordTimeData[-1].split(' ')[1])
#     # Remove Silent and Short Pauses
#     for line in wordTimeData:
#         if 'sil' in line or 'sp' in line:
#             wordTimeData.remove(line)
#     # Find the start and end frame for this word
#     wordStartFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                                 0]) / maxClipDuration * FRAMES_PER_VIDEO)
#     wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
#                              1]) / maxClipDuration * FRAMES_PER_VIDEO)
#     # All mouth file names of video
#     frameFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
#     # Note the file names of the word
#     wordFrameFiles = frameFiles[wordStartFrame:wordEndFrame + 1]
#     # Loop
#     for f, wordFrameFile in enumerate(wordFrameFiles):
#         # File names
#         si_frame_file_names.append(wordFrameFile.replace('Mouth', 'Frame'))

# with open("frame_names_list_si.txt", 'w') as f:
#     f.write('\n'.join(si_frame_file_names))

'''
cd /home/voletiv/GitHubRepos/gazr/build
./gazr_benchmark_head_pose_multiple_frames ../../lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat ../../zero-shot-learning-experiments/GRIDcorpus/frame_names_list_train_val.txt > head_poses_train_val.txt
./gazr_benchmark_head_pose_multiple_frames ../../lipreading-in-the-wild-experiments/shape-predictor/shape_predictor_68_face_landmarks.dat ../../zero-shot-learning-experiments/GRIDcorpus/frame_names_list_si.txt > head_poses_si.txt
'''

# READ ALL POSES
train_val_head_poses = read_head_poses(mode='train_val', num=np.cumsum(train_val_word_durations)[-1])

attributes['train_val_head_poses'] = train_val_head_poses
np.save("grid_attributes_dict", attributes)

# SI
si_head_poses = read_head_poses(mode='si', num=np.cumsum(si_word_durations)[-1])

attributes['si_head_poses'] = si_head_poses
np.save("grid_attributes_dict", attributes)


# # Find the index of the first unfilled row in poses
# for i, pose in enumerate(train_val_head_poses):
#     if np.all(pose == np.array([0, 0, 0])):
#             print(i)
#             break
# # 306216

# # Find the index in train_val_dirs, train_val_word_numbers
# np.where(np.cumsum(train_val_word_durations) == i)

# Check entries
idx = -1
r = gen_txt_files_line_by_line(mode='si', word='Estimating head pose')
for si_dirs_index in tqdm.tqdm(range(len(si_dirs))):
    duration = si_word_durations[si_dirs_index]
    # if 's06/lrak' in si_dirs[train_val_dirs_index]:
    #     print('\n')
    for frameNumber in range(duration):
        idx += 1
        line = next(r)
        # if 's06/lrak' in si_dirs[train_val_dirs_index]:
        #     print(si_dirs[si_dirs_index], si_word_numbers[si_dirs_index], si_word_durations[si_dirs_index], line)
        if si_dirs[si_dirs_index] not in line:
            raise KeyboardInterrupt
r.close()

# # train_val_dirs_index = 32135
# # idx = 272852

# lines = read_txt_files_line_range(mode='train_val', start_idx=272840, stop_idx=272853, word='Estimating head pose')
# for ll in lines:
#     ll

########################################
# ATTR - Lipreader features
########################################

train_val_lipreader_64_features = np.zeros((len(train_val_dirs), 64))
make_LSTMlipreader_predictions(train_val_lipreader_64_features,
                               train_val_lipreader_64_features,
                               train_val_dirs,
                               train_val_word_numbers,
                               train_val_word_idx,
                               LSTMLipreaderEncoder,
                               GRID_VOCAB_FULL,
                               0)

si1314_lipreader_preds = np.zeros((len(si1314_dirs), 51))
make_LSTMlipreader_predictions(si1314_lipreader_preds,
                               si1314_lipreader_preds,
                               si1314_dirs,
                               si1314_word_numbers,
                               si1314_word_idx,
                               LSTMLipreaderModel,
                               GRID_VOCAB_FULL,
                               0)

np.savez('lipreader_64_features', train_val_lipreader_64_features=train_val_lipreader_64_features, si1314_lipreader_64_features=si1314_lipreader_64_features)


########################################
# CRITIC
########################################

C3DCriticModel = load_C3DCritic()

lipreader_preds_wordidx_and_correctorwrong = np.load('lipreader_preds_wordidx_and_correctorwrong.npy').item()

train_val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_val_lipreader_pred_word_idx']
si1314_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si1314_lipreader_preds_word_idx']

train_val_critic_preds = np.zeros(len(train_val_dirs))
make_critic_predictions(train_val_critic_preds,
                        train_val_lipreader_preds_word_idx,
                        train_val_dirs,
                        train_val_word_numbers,
                        train_val_word_idx,
                        C3DCriticModel,
                        GRID_VOCAB_FULL,
                        0)

si1314_critic_preds = np.zeros(len(si1314_dirs))
make_critic_predictions(si1314_critic_preds,
                        si1314_lipreader_preds_word_idx,
                        si1314_dirs,
                        si1314_word_numbers,
                        si1314_word_idx,
                        C3DCriticModel,
                        GRID_VOCAB_FULL,
                        0)

np.savez('critic_preds', train_val_critic_preds=train_val_critic_preds, si1314_critic_preds=si1314_critic_preds)


#############################################################
# LIPREADER SELF-TRAIN 10%
#############################################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()
LSTMLipreaderModel.load_weights(os.path.join(LIPREADER_DIR, 'SELF-TRAINING', 'LSTMLipReader-revSeq-Mask-LSTMh256-tanh-depth2-enc64-relu-adam-1e-03-tMouth-valMouth-NOmeanSub-GRIDcorpus-s0107-10-si-s1314-10PercentSelfTraining-LRthresh0.90-iter00-epoch079-tl1.1377-ta0.6460-vl1.5886-va0.5360-sil3.9002-sia0.2181.hdf5'))

# Make 64-dim features
train_val_10pc_lipreader_64_features = np.zeros((len(train_val_dirs), 64))
make_LSTMlipreader_predictions(train_val_10pc_lipreader_64_features,
                               train_val_10pc_lipreader_64_features,
                               train_val_dirs,
                               train_val_word_numbers,
                               train_val_word_idx,
                               LSTMLipreaderEncoder,
                               GRID_VOCAB_FULL,
                               0)

si1314_lipreader_10pc_64_features = np.zeros((len(si1314_dirs), 64))
make_LSTMlipreader_predictions(si1314_lipreader_10pc_64_features,
                               si1314_lipreader_10pc_64_features,
                               si1314_dirs,
                               si1314_word_numbers,
                               si1314_word_idx,
                               LSTMLipreaderEncoder,
                               GRID_VOCAB_FULL,
                               0)

np.savez('lipreader_10pc_64_features', train_val_10pc_lipreader_64_features=train_val_10pc_lipreader_64_features, si1314_lipreader_10pc_64_features=si1314_lipreader_10pc_64_features)

train_val_lipreader_preds = np.zeros((len(train_val_dirs), len(GRID_VOCAB_FULL)))
train_val_lipreader_pred_word_idx = np.zeros(len(train_val_dirs), dtype=int)
train_val_lipreader_preds_correct_or_wrong = np.zeros(len(train_val_dirs), dtype=bool)
make_LSTMlipreader_predictions(train_val_lipreader_preds,
                               train_val_lipreader_pred_word_idx,
                               train_val_lipreader_preds_correct_or_wrong,
                               train_val_dirs,
                               train_val_word_numbers,
                               train_val_word_idx,
                               LSTMLipreaderModel,
                               GRID_VOCAB_FULL,
                               0)
