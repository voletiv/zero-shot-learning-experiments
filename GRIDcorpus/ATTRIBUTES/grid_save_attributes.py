import matplotlib.pyplot as plt
import numpy as np
import os

from grid_attributes_functions import *

########################################
# ATTRIBUTES
########################################

# Speaker identity

# Male or not

# Bilabial or not

# Duration of word

# Head Pose Mean (3)

# Head Pose Range (3)

# LipreaderEncoder Features (64)

#############################################################
# LOAD ATTRIBUTES
#############################################################

grid_attributes_dict = np.load(os.path.join(GRID_ATTR_DIR, 'grid_attributes_dict.npy')).item()

# grid_attributes_dict.keys()
# dict_keys(['train_val_speaker_identity', 'si_speaker_identity', 'train_val_male_or_not', 'si_male_or_not', 'train_val_word_durations', 'si_word_durations',
# 'train_val_bilabial_or_not', 'si_bilabial_or_not', 'train_val_head_poses_per_frame_in_word', 'si_head_poses_per_frame_in_word', 'train_val_word_metadata', 'si_word_metadata'])

# grid_attributes_dict['train_val_word_metadata'].keys()
# dict_keys(['startFrame', 'endFrame', 'wordDuration'])

train_val_dirs = grid_attributes_dict['train_val_dirs']
train_val_word_numbers = grid_attributes_dict['train_val_word_numbers']
si1314_dirs = grid_attributes_dict['si1314_dirs']
si1314_word_numbers = grid_attributes_dict['si1314_word_numbers']

#############################################################
# LOAD TRAINDIRS, VALDIRS, SI DIRS
#############################################################

# # TRAIN AND VAL
# trainValSpeakersList = [1, 2, 3, 4, 5, 6, 7, 9]
# valSplit = 0.1
# trainDirs = []
# valDirs = []
# np.random.seed(29)
# # For each speaker
# for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
#     speakerDir = os.path.join(GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
#     # List of all videos for each speaker
#     vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
#     totalNumOfImages = len(vidDirs)
#     # To shuffle directories before splitting into train and validate
#     fullListIdx = list(range(totalNumOfImages))
#     np.random.shuffle(fullListIdx)
#     # Append training directories
#     for i in fullListIdx[:int((1 - valSplit) * totalNumOfImages)]:
#         trainDirs.append(vidDirs[i])
#     # Append val directories
#     for i in fullListIdx[int((1 - valSplit) * totalNumOfImages):]:
#         valDirs.append(vidDirs[i])

# # TRAIN
# train_dirs_binary = np.array([d in trainDirs for d in train_val_dirs])
# grid_attributes_dict['train_dirs_binary'] = train_dirs_binary

# # VAL
# val_dirs_binary = np.array([d in valDirs for d in train_val_dirs])
# grid_attributes_dict['val_dirs_binary'] = val_dirs_binary

# # SI
# si_dirs_binary = np.array(['s10' in d for d in train_val_dirs])

# train_dirs = train_val_dirs[train_dirs_binary]
# grid_attributes_dict['train_dirs'] = train_dirs

# val_dirs = train_val_dirs[val_dirs_binary]
# grid_attributes_dict['val_dirs'] = val_dirs

# for i, d in enumerate(train_val_dirs):
#     if si_dirs_binary[i]:
#         si_dirs = np.append(si_dirs, d)

# # SAVE
# train_dirs = grid_attributes_dict['train_dirs']
# train_dirs_binary = grid_attributes_dict['train_dirs_binary']

# val_dirs = grid_attributes_dict['val_dirs']
# val_dirs_binary = grid_attributes_dict['val_dirs_binary']

# si_dirs = grid_attributes_dict['si_dirs']
# si_dirs_binary = grid_attributes_dict['si_dirs_binary']

# train_val_word_numbers = grid_attributes_dict['train_val_word_numbers']
# # train_word_numbers = train_val_word_numbers[train_dirs_binary]
# grid_attributes_dict['train_word_numbers'] = train_word_numbers
# val_word_numbers = train_val_word_numbers[val_dirs_binary]
# grid_attributes_dict['val_word_numbers'] = val_word_numbers


# train_val_word_idx = grid_attributes_dict['train_val_word_idx']
# train_word_idx = train_val_word_idx[train_dirs_binary]
# grid_attributes_dict['train_word_idx'] = train_word_idx
# val_word_idx = train_val_word_idx[val_dirs_binary]
# grid_attributes_dict['val_word_idx'] = val_word_idx

# si1314_word_numbers = grid_attributes_dict['si1314_word_numbers']
# si_word_numbers = np.append(si1314_word_numbers, train_val_word_numbers[si_dirs_binary])
# grid_attributes_dict['si_word_numbers'] = si_word_numbers

# si1314_word_idx = grid_attributes_dict['si1314_word_idx']
# si_word_idx = np.append(si1314_word_idx, train_val_word_idx[si_dirs_binary])
# grid_attributes_dict['si_word_idx'] = si_word_idx

train_dirs = grid_attributes_dict['train_dirs']
train_dirs_binary = grid_attributes_dict['train_dirs_binary']
val_dirs = grid_attributes_dict['val_dirs']
val_dirs_binary = grid_attributes_dict['val_dirs_binary']
si_dirs = grid_attributes_dict['si_dirs']
si_dirs_binary = grid_attributes_dict['si_dirs_binary']
train_val_word_numbers = grid_attributes_dict['train_val_word_numbers']
si1314_word_numbers = grid_attributes_dict['si1314_word_numbers']
train_val_word_idx = grid_attributes_dict['train_val_word_idx']
si1314_word_idx = grid_attributes_dict['si1314_word_idx']

#############################################################
# LOAD CORRECT_OR_NOT
#############################################################

# lipreader_preds = np.load(os.path.join(GRID_DIR, 'lipreader_preds.npy')).item()

# train_val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_val_lipreader_pred_word_idx']
# train_val_lipreader_preds_correct_or_wrong = lipreader_preds['train_val_lipreader_preds_correct_or_wrong']
# si_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si1314_lipreader_pred_word_idx']
# si_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['si1314_lipreader_preds_correct_or_wrong']

# train_lipreader_preds_word_idx = train_val_lipreader_preds_word_idx[train_dirs_binary]
# train_lipreader_preds_correct_or_wrong = train_val_lipreader_preds_correct_or_wrong[train_dirs_binary]

# val_lipreader_preds_word_idx = train_val_lipreader_preds_correct_or_wrong[val_dirs_binary]
# val_lipreader_preds_correct_or_wrong = train_val_lipreader_preds_correct_or_wrong[val_dirs_binary]

# for siwi, sicw in zip(train_val_lipreader_preds_word_idx[si_dirs_binary], train_val_lipreader_preds_correct_or_wrong[si_dirs_binary]):
#     si_lipreader_preds_word_idx = np.append(si_lipreader_preds_word_idx, siwi)
#     si_lipreader_preds_correct_or_wrong = np.append(si_lipreader_preds_correct_or_wrong, sicw)

# lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_word_idx'] = train_lipreader_preds_word_idx
# lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_word_idx'] = val_lipreader_preds_word_idx
# lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_word_idx'] = si_lipreader_preds_word_idx
# lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_correct_or_wrong'] = train_lipreader_preds_correct_or_wrong
# lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_correct_or_wrong'] = val_lipreader_preds_correct_or_wrong
# lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_correct_or_wrong'] = si_lipreader_preds_correct_or_wrong

# np.save('lipreader_preds_wordidx_and_correctorwrong', lipreader_preds_wordidx_and_correctorwrong)

lipreader_preds_wordidx_and_correctorwrong = np.load('lipreader_preds_wordidx_and_correctorwrong.npy').item()

train_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_word_idx']
val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_word_idx']
si_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_word_idx']

train_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_correct_or_wrong']
val_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_correct_or_wrong']
si_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_correct_or_wrong']

#############################################################
# MAKE ATTRIBUTES ARRAY: n x a
#############################################################

train_num_of_rows = len(train_dirs)
val_num_of_rows = len(val_dirs)
si_num_of_rows = len(si_dirs)

train_grid_attributes = np.empty((train_num_of_rows, 0))
val_grid_attributes = np.empty((val_num_of_rows, 0))
si_grid_attributes = np.empty((si_num_of_rows, 0))

# SPEAKER IDENTITY
train_val_speaker_identity = grid_attributes_dict['train_val_speaker_identity']
# Train
train_speaker_identity = train_val_speaker_identity[train_dirs_binary]
train_grid_attributes = np.hstack((train_grid_attributes, np.reshape(np.array(train_speaker_identity, dtype=float), (train_num_of_rows, 1))))
# Val
val_speaker_identity = train_val_speaker_identity[val_dirs_binary]
val_grid_attributes = np.hstack((val_grid_attributes, np.reshape(np.array(val_speaker_identity, dtype=float), (val_num_of_rows, 1))))
# Si
si_speaker_identity = np.append(grid_attributes_dict['si1314_speaker_identity'], train_val_speaker_identity[si_dirs_binary])
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_speaker_identity, dtype=float), (si_num_of_rows, 1))))

# MALE OR NOT
train_val_male_or_not = grid_attributes_dict['train_val_male_or_not']
# Train
train_male_or_not = train_val_male_or_not[train_dirs_binary]
train_grid_attributes = np.hstack((train_grid_attributes, np.reshape(np.array(train_male_or_not, dtype=float), (train_num_of_rows, 1))))
# Val
val_male_or_not = train_val_male_or_not[val_dirs_binary]
val_grid_attributes = np.hstack((val_grid_attributes, np.reshape(np.array(val_male_or_not, dtype=float), (val_num_of_rows, 1))))
# Si
si_male_or_not = np.append(grid_attributes_dict['si1314_male_or_not'], train_val_male_or_not[si_dirs_binary])
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_male_or_not, dtype=float), (si_num_of_rows, 1))))

# BILABIAL OR NOT
train_val_bilabial_or_not = grid_attributes_dict['train_val_bilabial_or_not']
# Train
train_bilabial_or_not = train_val_bilabial_or_not[train_dirs_binary]
train_grid_attributes = np.hstack((train_grid_attributes, np.reshape(np.array(train_bilabial_or_not, dtype=float), (train_num_of_rows, 1))))
# Val
val_bilabial_or_not = train_val_bilabial_or_not[val_dirs_binary]
val_grid_attributes = np.hstack((val_grid_attributes, np.reshape(np.array(val_bilabial_or_not, dtype=float), (val_num_of_rows, 1))))
# Si
si_bilabial_or_not = np.append(grid_attributes_dict['si1314_bilabial_or_not'], train_val_word_durations[si_dirs_binary])
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_bilabial_or_not, dtype=float), (si_num_of_rows, 1))))

# WORD DURATION
train_val_word_durations = grid_attributes_dict['train_val_word_durations']
# Train
train_word_durations = train_val_word_durations[train_dirs_binary]
train_grid_attributes = np.hstack((train_grid_attributes, np.reshape(np.array(train_word_durations, dtype=float), (train_num_of_rows, 1))))
# Val
val_word_durations = train_val_word_durations[val_dirs_binary]
val_grid_attributes = np.hstack((val_grid_attributes, np.reshape(np.array(val_word_durations, dtype=float), (val_num_of_rows, 1))))
# Si
si1314_word_durations = grid_attributes_dict['si1314_word_durations']
si_word_durations = np.append(si1314_word_durations, train_val_word_durations[si_dirs_binary])
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_word_durations, dtype=float), (si_num_of_rows, 1))))

# SAVE
np.save('train_grid_attributes_matrix', train_grid_attributes)
np.save('val_grid_attributes_matrix', val_grid_attributes)
np.save('si_grid_attributes_matrix', si_grid_attributes)

#############################################################
# PLOTS OF ACCURACY WITH ATTRIBUTES
#############################################################

# SPEAKER IDENTITY
for speaker in TRAIN_VAL_SPEAKERS_LIST:
    speaker_dir = "s{0:02d}".format(speaker)
    # Train
    train_speaker_binary = [speaker_dir in train_dir for train_dir in train_dirs]
    train_speaker_correctornot = train_lipreader_preds_correct_or_wrong[train_speaker_binary]
    train_speaker_acc = np.sum(train_speaker_correctornot)/len(train_speaker_correctornot)
    # Val
    val_speaker_binary = [speaker_dir in val_dir for val_dir in val_dirs]
    val_speaker_correctornot = val_lipreader_preds_correct_or_wrong[val_speaker_binary]
    val_speaker_acc = np.sum(val_speaker_correctornot)/len(val_speaker_correctornot)
    print(speaker, train_speaker_acc, val_speaker_acc)

# 1 0.902190051967 0.893333333333
# 2 0.885916016351 0.876666666667
# 3 0.947741694662 0.943333333333
# 4 0.905821282907 0.895
# 5 0.932332220986 0.943333333333
# 6 0.940703329592 0.945
# 7 0.940449438202 0.946127946128
# 10 nan nan


# BILABIAL OR NOT

# Train
train_bilabial_acc = np.sum(train_lipreader_preds_correct_or_wrong[train_bilabial_or_not])/np.sum(train_bilabial_or_not)
# 0.97038413878562579
train_not_bilabial_acc = np.sum(train_lipreader_preds_correct_or_wrong[np.logical_not(train_bilabial_or_not)])/np.sum(np.logical_not(train_bilabial_or_not))
# 0.90891236624678318

# Val
val_bilabial_acc = np.sum(val_lipreader_preds_correct_or_wrong[val_bilabial_or_not])/np.sum(val_bilabial_or_not)
# 0.96752519596864506
val_not_bilabial_acc = np.sum(val_lipreader_preds_correct_or_wrong[np.logical_not(val_bilabial_or_not)])/np.sum(np.logical_not(val_bilabial_or_not))
# 0.90760375643744318

# Si
si_bilabial_acc = np.sum(si_lipreader_preds_correct_or_wrong[si_bilabial_or_not])/np.sum(si_bilabial_or_not)
# 0.37346693921080698
si_not_bilabial_acc = np.sum(si_lipreader_preds_correct_or_wrong[np.logical_not(si_bilabial_or_not)])/np.sum(np.logical_not(si_bilabial_or_not))
# 0.35498248965297674


#############################################################
# POSE
#############################################################

# train_val_head_poses_per_frame_in_word = grid_attributes_dict['train_val_head_poses_per_frame_in_word']
# si1314_head_poses_per_frame_in_word = grid_attributes_dict['si1314_head_poses_per_frame_in_word']

# # Poses
# np.mean(train_val_head_poses_per_frame_in_word, axis=0)
# np.max(train_val_head_poses_per_frame_in_word, axis=0)
# np.argmax(train_val_head_poses_per_frame_in_word, axis=0)

# plt.subplot(131)
# plt.hist(train_val_head_poses_per_frame_in_word[:, 0], bins=20)
# plt.xlabel('X')
# plt.subplot(132)
# plt.hist(train_val_head_poses_per_frame_in_word[:, 1], bins=20)
# plt.xlabel('Y')
# plt.subplot(133)
# plt.hist(train_val_head_poses_per_frame_in_word[:, 2], bins=20)
# plt.xlabel('Z')
# plt.suptitle("Histograms of head pose values")

# MEAN AND RANGE OF HEAD POSES
# Train_Val
train_val_head_poses_per_frame_in_word = grid_attributes_dict['train_val_head_poses_per_frame_in_word']
train_val_word_durations = grid_attributes_dict['train_val_word_durations']
train_val_word_durations_cum_sum = np.cumsum(train_val_word_durations)
train_val_word_durations_cum_sum = np.append(0, train_val_word_durations_cum_sum)
train_val_head_poses_means = np.empty((0, 3))
train_val_head_poses_ranges = np.empty((0, 3))
for i in range(len(train_val_dirs)):
    head_poses = train_val_head_poses_per_frame_in_word[train_val_word_durations_cum_sum[i]:train_val_word_durations_cum_sum[i+1]]
    train_val_head_poses_means = np.vstack((train_val_head_poses_means, np.mean(head_poses, axis=0)))
    train_val_head_poses_ranges = np.vstack((train_val_head_poses_ranges, np.max(head_poses, axis=0) - np.min(head_poses, axis=0)))
# Si_notfromtrainval
si1314_head_poses_per_frame_in_word = grid_attributes_dict['si1314_head_poses_per_frame_in_word']
si1314_word_durations = grid_attributes_dict['si1314_word_durations']
si1314_word_durations_cum_sum = np.cumsum(si1314_word_durations)
si1314_word_durations_cum_sum = np.append(0, si1314_word_durations_cum_sum)
si1314_head_poses_means = np.empty((0, 3))
si1314_head_poses_ranges = np.empty((0, 3))
for i in range(len(si1314_dirs)):
    head_poses = si1314_head_poses_per_frame_in_word[si1314_word_durations_cum_sum[i]:si1314_word_durations_cum_sum[i+1]]
    si1314_head_poses_means = np.vstack((si1314_head_poses_means, np.mean(head_poses, axis=0)))
    si1314_head_poses_ranges = np.vstack((si1314_head_poses_ranges, np.max(head_poses, axis=0) - np.min(head_poses, axis=0)))
# Train
train_head_poses_means = train_val_head_poses_means[train_dirs_binary]
train_head_poses_ranges = train_val_head_poses_ranges[train_dirs_binary]
# Val
val_head_poses_means = train_val_head_poses_means[val_dirs_binary]
val_head_poses_ranges = train_val_head_poses_ranges[val_dirs_binary]
# Si
si_head_poses_means = si1314_head_poses_means
si_head_poses_means_from_trainval = train_val_head_poses_means[si_dirs_binary]
si_head_poses_means = np.vstack((si_head_poses_means, si_head_poses_means_from_trainval))
si_head_poses_ranges = si1314_head_poses_ranges
si_head_poses_ranges_from_trainval = train_val_head_poses_ranges[si_dirs_binary]
si_head_poses_ranges = np.vstack((si_head_poses_ranges, si_head_poses_ranges_from_trainval))

train_grid_attributes = np.hstack((train_grid_attributes, train_head_poses_means))
train_grid_attributes = np.hstack((train_grid_attributes, train_head_poses_ranges))

val_grid_attributes = np.hstack((val_grid_attributes, val_head_poses_means))
val_grid_attributes = np.hstack((val_grid_attributes, val_head_poses_ranges))

si_grid_attributes = np.hstack((si_grid_attributes, si_head_poses_means))
si_grid_attributes = np.hstack((si_grid_attributes, si_head_poses_ranges))

np.save('train_grid_attributes_matrix', train_grid_attributes)
np.save('val_grid_attributes_matrix', val_grid_attributes)
np.save('si_grid_attributes_matrix', si_grid_attributes)

#############################################################
# LIPREADER FEATURES
#############################################################

lipreader_64_features = np.load('lipreader_64_features.npz')

train_val_lipreader_64_features = lipreader_64_features['train_val_lipreader_64_features']
si1314_lipreader_64_features = lipreader_64_features['si1314_lipreader_64_features']

train_lipreader_64_features = train_val_lipreader_64_features[train_dirs_binary]
val_lipreader_64_features = train_val_lipreader_64_features[val_dirs_binary]
si_lipreader_64_features = si1314_lipreader_64_features
si_lipreader_64_features = np.vstack((si_lipreader_64_features, train_val_lipreader_64_features[si_dirs_binary]))

train_grid_attributes = np.hstack((train_grid_attributes, train_lipreader_64_features))
val_grid_attributes = np.hstack((val_grid_attributes, val_lipreader_64_features))
si_grid_attributes = np.hstack((si_grid_attributes, si_lipreader_64_features))

np.save('train_grid_attributes_matrix', train_grid_attributes)
np.save('val_grid_attributes_matrix', val_grid_attributes)
np.save('si_grid_attributes_matrix', si_grid_attributes)


