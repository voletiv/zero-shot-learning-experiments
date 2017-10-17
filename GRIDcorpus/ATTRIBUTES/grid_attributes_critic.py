import numpy as np

from grid_attributes_params import *

#############################################################
# LOAD ATTRIBUTES
#############################################################

grid_attributes_dict = np.load(os.path.join(GRID_DIR, 'grid_attributes.npy')).item()

# grid_attributes_dict.keys()
# dict_keys(['train_val_speaker_identity', 'si_speaker_identity', 'train_val_male_or_not', 'si_male_or_not', 'train_val_word_durations', 'si_word_durations',
# 'train_val_bilabial_or_not', 'si_bilabial_or_not', 'train_val_head_poses', 'si_head_poses', 'train_val_word_metadata', 'si_word_metadata'])

# grid_attributes_dict['train_val_word_metadata'].keys()
# dict_keys(['startFrame', 'wordDuration', 'endFrame'])

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

nOfRows = len(grid_attributes_dict['train_val_male_or_not'])

train_val_grid_attributes = np.empty((nOfRows, 0))

# Speaker Idenity
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(grid_attributes_dict['train_val_speaker_identity'], dtype=float), (nOfRows, 1))))

# Male or not
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(grid_attributes_dict['train_val_male_or_not'], dtype=float), (nOfRows, 1))))

# Duration of word
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(grid_attributes_dict['train_val_word_durations'], dtype=float)/np.max(grid_attributes_dict['train_val_word_durations']), (nOfRows, 1))))

# Bilabial or not
train_val_grid_attributes = np.hstack((train_val_grid_attributes, np.reshape(np.array(grid_attributes_dict['train_val_bilabial_or_not'], dtype=float), (nOfRows, 1))))

# Pose


