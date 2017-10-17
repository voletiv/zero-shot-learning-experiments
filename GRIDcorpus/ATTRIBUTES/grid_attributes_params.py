# PARAMS

import os
import sys

#############################################################
# DIRECTORIES
#############################################################

# GRID directory
if 'GRID_ATTR_DIR' not in dir():
    GRID_ATTR_DIR = os.path.dirname(os.path.realpath(__file__))

if GRID_ATTR_DIR not in sys.path:
    sys.path.append(GRID_ATTR_DIR)

# GRID directory
if 'GRID_DIR' not in dir():
    GRID_DIR = os.path.normpath(os.path.join(GRID_ATTR_DIR, '..'))

if GRID_DIR not in sys.path:
    sys.path.append(GRID_DIR)

#############################################################
# PARAMS
#############################################################

from grid_params import *

TRAIN_VAL_HEAD_POSE_TXT_FILES = [os.path.join(GRID_DIR, 'head_poses_train_val_00.txt'),
                                 os.path.join(GRID_DIR, 'head_poses_train_val_01.txt'),
                                 os.path.join(GRID_DIR, 'head_poses_train_val_02.txt')]

SI_HEAD_POSE_TXT_FILES = [os.path.join(GRID_DIR, 'head_poses_si.txt')]
