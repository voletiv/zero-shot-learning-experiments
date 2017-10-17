import matplotlib.pyplot as plt
import numpy as np
import os

from grid_attributes_functions import *

#############################################################
# LOAD ATTRIBUTES
#############################################################

grid_attributes_dict = np.load(os.path.join(GRID_ATTR_DIR, 'grid_attributes_dict.npy')).item()

# grid_attributes_dict.keys()
# dict_keys(['train_val_speaker_identity', 'si_speaker_identity', 'train_val_male_or_not', 'si_male_or_not', 'train_val_word_durations', 'si_word_durations',
# 'train_val_bilabial_or_not', 'si_bilabial_or_not', 'train_val_head_poses_per_frame_in_word', 'si_head_poses_per_frame_in_word', 'train_val_word_metadata', 'si_word_metadata'])

# grid_attributes_dict['train_val_word_metadata'].keys()
# dict_keys(['startFrame', 'endFrame', 'wordDuration'])

#############################################################
# LOAD CORRECT_OR_NOT
#############################################################

lipreader_preds_wordidx_and_correctornot = np.load(os.path.join(GRID_ATTR_DIR, 'lipreader_preds_wordidx_and_correctornot.npy')).item()
train_val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctornot['train_val_lipreader_pred_word_idx']
train_val_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctornot['train_val_lipreader_preds_correct_or_wrong']
si_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctornot['si_lipreader_pred_word_idx']
si_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctornot['si_lipreader_preds_correct_or_wrong']

########################################
# ATTRIBUTES
########################################

# Speaker identity

# Male or not

# Duration of word

# Bilabial or not

# Pose

#############################################################
# MAKE ATTRIBUTES ARRAY: n x a
#############################################################

train_val_num_of_rows = len(grid_attributes_dict['train_val_male_or_not'])
si_num_of_rows = len(grid_attributes_dict['si_male_or_not'])

train_val_word_metadata = grid_attributes_dict['train_val_word_metadata']

train_val_grid_attributes = np.empty((train_val_num_of_rows, 0))
si_grid_attributes = np.empty((si_num_of_rows, 0))

# Speaker Idenity
train_val_speaker_identity = grid_attributes_dict['train_val_speaker_identity']
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(train_val_speaker_identity, dtype=float), (train_val_num_of_rows, 1))))
si_speaker_identity = grid_attributes_dict['si_speaker_identity']
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_speaker_identity, dtype=float), (si_num_of_rows, 1))))

# Male or not
train_val_male_or_not = grid_attributes_dict['train_val_male_or_not']
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(train_val_male_or_not, dtype=float), (train_val_num_of_rows, 1))))
si_male_or_not = grid_attributes_dict['si_male_or_not']
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_male_or_not, dtype=float), (si_num_of_rows, 1))))

# Duration of word
train_val_word_durations = grid_attributes_dict['train_val_word_durations']
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(train_val_word_durations, dtype=float)/np.max(train_val_word_durations), (train_val_num_of_rows, 1))))
si_word_durations = grid_attributes_dict['si_word_durations']
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(si_word_durations, dtype=float)/np.max(si_word_durations), (si_num_of_rows, 1))))

# Bilabial or not
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(grid_attributes_dict['train_val_bilabial_or_not'], dtype=float), (train_val_num_of_rows, 1))))
si_grid_attributes = np.hstack((si_grid_attributes, np.reshape(np.array(grid_attributes_dict['si_bilabial_or_not'], dtype=float), (si_num_of_rows, 1))))

#############################################################
# PLOTS OF ACCURACY WITH ATTRIBUTES
#############################################################

train_val_dirs = grid_attributes_dict['train_val_dirs']
si_dirs = grid_attributes_dict['si_dirs']

# Speaker Idenity
for speaker in TRAIN_VAL_SPEAKERS_LIST:
    speaker_dir = "s{0:02d}".format(speaker)
    speaker_idx = [speaker_dir in dirs_list_element for dirs_list_element in train_val_dirs]
    speaker_correctornot = train_val_lipreader_preds_correct_or_wrong[speaker_idx]
    speaker_acc = np.sum(speaker_correctornot)/len(speaker_correctornot)
    print(speaker, speaker_acc)

#############################################################
# POSE
#############################################################

train_val_head_poses_per_frame_in_word = grid_attributes_dict['train_val_head_poses_per_frame_in_word']
si_head_poses_per_frame_in_word = grid_attributes_dict['si_head_poses_per_frame_in_word']

# Poses
np.mean(train_val_head_poses_per_frame_in_word, axis=0)
np.max(train_val_head_poses_per_frame_in_word, axis=0)
np.argmax(train_val_head_poses_per_frame_in_word, axis=0)

plt.subplot(131)
plt.hist(train_val_head_poses_per_frame_in_word[:, 0], bins=20)
plt.xlabel('X')
plt.subplot(132)
plt.hist(train_val_head_poses_per_frame_in_word[:, 1], bins=20)
plt.xlabel('Y')
plt.subplot(133)
plt.hist(train_val_head_poses_per_frame_in_word[:, 2], bins=20)
plt.xlabel('Z')
plt.suptitle("Histograms of head pose values")

# Take mean and range among frames as attributes

