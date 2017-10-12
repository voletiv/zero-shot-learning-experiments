# GET GRIDCORPUS ATTRIBUTES

from grid_params import *
from grid_functions import *

########################################
# Get FULL Data
########################################

train_val_dirs, train_val_word_numbers, train_val_word_idx, \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(TRAIN_VAL_SPEAKERS_LIST)

si_dirs, si_word_numbers, si_word_idx \
    = load_GRIDcorpus_speakers_dirs_wordnums_wordidx_lists(SI_SPEAKERS_LIST)

########################################
# Load LipReader
########################################

LSTMLipreaderModel, LSTMLipreaderEncoder = load_LSTM_lipreader_and_encoder()

########################################
# ATTRIBUTES
########################################

# Mouth-to-face ratio

# Pose

# Duration of word

# Presence of bilabials

# Range of mouth movement??


########################################
# ATTRIBUTES
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


########################################
# Get ATTRIBUTES
########################################

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
        wordMouthFiles = frameFiles[wordStartFrame:wordEndFrame + 1]





# a
# again
# at
# b
# bin
# blue
# by
# c
# d
# e
# eight
# f
# five
# four
# g
# green
# h
# i
# in
# j
# k
# l
# lay
# m
# n
# nine
# now
# o
# one
# p
# place
# please
# q
# r
# red
# s
# set
# seven
# six
# soon
# t
# three
# two
# u
# v
# white
# with
# x
# y
# z
# zero

