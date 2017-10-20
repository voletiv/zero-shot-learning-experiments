
#############################################################
# IMPORT
#############################################################

from common_params import *

#############################################################
# PARAMETERS FOR LIPREADER
#############################################################

LIPREADER_ENCODED_DIM = 64

LIPREADER_MODEL_FILE = ""
lipreader_filelist = os.listdir(LIPREADER_DIR)
for file in lipreader_filelist:
    if ('lipreader' in file or 'Lipreader' in file or 'LipReader' in file or 'LIPREADER' in file) and '.hdf5' in file:
        LIPREADER_MODEL_FILE = os.path.join(LIPREADER_DIR, file)

CRITIC_MODEL_FILE = ""
lipreader_filelist = os.listdir(LIPREADER_DIR)
for file in lipreader_filelist:
    if ('critic' in file or 'Critic' in file or 'CRITIC' in file) and '.hdf5' in file:
        CRITIC_MODEL_FILE = os.path.join(LIPREADER_DIR, file)


