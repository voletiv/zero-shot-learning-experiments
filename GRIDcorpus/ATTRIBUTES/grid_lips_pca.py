# PCA

from sklearn import decomposition

from grid_attributes_functions import *


#############################################################
# LOAD BASICS
#############################################################

grid_basics = np.load('grid_basics.npz')

train_dirs = grid_basics['train_dirs']
train_word_numbers = grid_basics['train_word_numbers']
train_word_idx = grid_basics['train_word_idx']

val_dirs = grid_basics['val_dirs']
val_word_numbers = grid_basics['val_word_numbers']
val_word_idx = grid_basics['val_word_idx']

si131410_dirs = grid_basics['si131410_dirs']
si131410_word_numbers = grid_basics['si131410_word_numbers']
si131410_word_idx = grid_basics['si131410_word_idx']

#############################################################
# TRAIN
#############################################################

num_of_frames = 5

grid_mouth_images = np.zeros((len(train_dirs)*num_of_frames, NUM_OF_MOUTH_PIXELS))
get_grid_mouth_images(grid_mouth_images,
                      train_dirs,
                      train_word_numbers,
                      num_of_frames=num_of_frames,
                      grid_vocab=GRID_VOCAB_FULL,
                      startNum=0)

np.savez('grid_lips_pca', grid_mouth_images=grid_mouth_images)

