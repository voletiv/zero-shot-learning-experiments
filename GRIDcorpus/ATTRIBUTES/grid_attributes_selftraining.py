import numpy as np
import optunity
import optunity.metrics
import random as rn
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

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

# train_val_dirs = np.append(np.append(train_dirs, val_dirs), si_dirs[12000:])
# train_val_word_numbers = np.append(np.append(train_word_numbers, val_word_numbers), si131410_word_numbers[12000:])
# train_val_word_idx = np.append(np.append(train_word_idx, val_word_idx), si131410_word_idx[12000:])
# si1314_dirs = si131410_dirs[:12000]
# si1314_word_numbers = si131410_word_numbers[:12000]
# si1314_word_idx = si131410_word_idx[:12000]

train_val_dirs = grid_basics['train_val_dirs']
train_val_word_numbers = grid_basics['train_val_word_numbers']
train_val_word_idx = grid_basics['train_val_word_idx']

si1314_dirs = grid_basics['si1314_dirs']
si1314_word_numbers = grid_basics['si1314_word_numbers']
si1314_word_idx = grid_basics['si1314_word_idx']

# #############################################################
# # FIND IN BASICS
# #############################################################

# train_10pc_idx = np.array([np.where(train_val_dirs == i)[0][0] + np.where(train_val_word_numbers[np.where(train_val_dirs == i)] == j)[0][0] for i,j in zip(train_10pc_dirs, train_10pc_word_numbers)])

# val_10pc_idx = np.array([np.where(train_val_dirs == i)[0][0] + np.where(train_val_word_numbers[np.where(train_val_dirs == i)] == j)[0][0] for i,j in zip(val_10pc_dirs, val_10pc_word_numbers)])

# si1314_10pc_idx = np.array([np.where(si1314_dirs == i)[0][0] + np.where(si1314word_numbers[np.where(si1314_dirs == i)] == j)[0][0] for i,j in zip(si1314_10pc_dirs, si1314_10pc_word_numbers)])

# np.savez('grid_basics', train_dirs=train_dirs, train_word_numbers=train_word_numbers, train_word_idx=train_word_idx,
#     val_dirs=val_dirs, val_word_numbers=val_word_numbers, val_word_idx=val_word_idx,
#     si131410_dirs=si131410_dirs, si131410_word_numbers=si131410_word_numbers, si131410_word_idx=si131410_word_idx,
#     train_val_dirs=train_val_dirs, train_val_word_numbers=train_val_word_numbers, train_val_word_idx=train_val_word_idx,
#     si1314_dirs=si1314_dirs, si1314_word_numbers=si1314_word_numbers, si1314_word_idx=si1314_word_idx,
#     train_10pc_idx=train_10pc_idx, val_10pc_idx=val_10pc_idx, si1314_10pc_idx=si1314_10pc_idx)

train_10pc_idx = grid_basics['train_10pc_idx']
val_10pc_idx = grid_basics['val_10pc_idx']
si1314_10pc_idx = grid_basics['si1314_10pc_idx']

# #############################################################
# # LOAD 10pc INFO
# #############################################################

# train_10pc_dirs, train_10pc_word_numbers, val_10pc_dirs, val_10pc_word_numbers, si1314_10pc_dirs, si1314_10pc_word_numbers = load_image_dirs_and_word_numbers(
#     trainValSpeakersList=[1, 2, 3, 4, 5, 6, 7, 10],
#     siList = [13, 14])

# # fileNamePre += "-GRIDcorpus-s0107-10-si-s1314"

train_10pc_dirs = train_val_dirs[train_10pc_idx]
train_10pc_word_numbers = train_val_word_numbers[train_10pc_idx]

val_10pc_dirs = train_val_dirs[val_10pc_idx]
val_10pc_word_numbers = train_val_word_numbers[val_10pc_idx]

si1314_10pc_dirs = train_val_dirs[si1314_10pc_idx]
val_10pc_word_numbers = train_val_word_numbers[si1314_10pc_idx]

#################################################################
# 10%
#################################################################

labelledPercent = 10

# Starting training with only 10% of training data,
# and assuming the rest 90% to be unlabelled

# Split into labelled and unlabelled training data
os.environ['PYTHONHASHSEED'] = '0'  # Necessary for python3
np.random.seed(29)
rn.seed(29)
tf.set_random_seed(29)
train_10pc_full_list = np.arange(len(train_10pc_dirs))
np.random.shuffle(train_10pc_full_list)

train_10pc_labelled_idx = train_10pc_full_list[:int(labelledPercent/100*len(train_10pc_dirs))]
train_10pc_labelled_dirs = np.array(train_10pc_dirs)[train_10pc_labelled_idx]
train_10pc_labelled_word_numbers = np.array(train_10pc_word_numbers)[train_10pc_labelled_idx]

train_10pc_unlabelled_idx = train_10pc_full_list[int(labelledPercent/100*len(train_10pc_dirs)):]
train_10pc_unlabelled_dirs = np.array(train_10pc_dirs)[train_10pc_unlabelled_idx]
train_10pc_unlabelled_word_numbers = np.array(train_10pc_word_numbers)[train_10pc_unlabelled_idx]

#############################################################
# LOAD CORRECT_OR_NOT, PREDS_WORDIDX
#############################################################

lipreader_preds_wordidx_and_correctorwrong = np.load('lipreader_preds_wordidx_and_correctorwrong.npy').item()

train_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_correct_or_wrong']
val_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_correct_or_wrong']
si131410_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_correct_or_wrong']

train_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_word_idx']
val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_word_idx']
si131410_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_word_idx']

# 10pc preds_corect_or_wrong
train_val_lipreader_preds_correct_or_wrong = np.append(np.append(train_lipreader_preds_correct_or_wrong, val_lipreader_preds_correct_or_wrong), si131410_lipreader_preds_correct_or_wrong[12000:])
si1314_lipreader_preds_correct_or_wrong = si131410_lipreader_preds_correct_or_wrong[:12000]

train_10pc_lipreader_preds_correct_or_wrong = train_val_lipreader_preds_correct_or_wrong[train_10pc_idx]
val_10pc_lipreader_preds_correct_or_wrong = train_val_lipreader_preds_correct_or_wrong[val_10pc_idx]
si1314_10pc_lipreader_preds_correct_or_wrong = si1314_lipreader_preds_correct_or_wrong[si1314_10pc_idx]

# 10pc preds_word_idx
train_val_lipreader_preds_word_idx = np.append(np.append(train_lipreader_preds_word_idx, val_lipreader_preds_word_idx), si131410_lipreader_preds_word_idx[12000:])
si1314_lipreader_preds_word_idx = si131410_lipreader_preds_word_idx[:12000]

train_10pc_lipreader_preds_word_idx = train_val_lipreader_preds_word_idx[train_10pc_idx]
val_10pc_lipreader_preds_word_idx = train_val_lipreader_preds_word_idx[val_10pc_idx]
si1314_10pc_lipreader_preds_word_idx = si1314_lipreader_preds_word_idx[si1314_10pc_idx]

#############################################################
# LOAD LIPREADER SELF-TRAIN 10% FEATURES
#############################################################

# lipreader_10pc_64_features = np.load('lipreader_10pc_64_features.npz')

# train_10pc_lipreader_64_features = lipreader_10pc_64_features['train_10pc_lipreader_64_features']
# val_10pc_lipreader_64_features = lipreader_10pc_64_features['val_10pc_lipreader_64_features']
# si131410_10pc_lipreader_64_features = lipreader_10pc_64_features['si_10pc_lipreader_64_features']

# train_val_10pc_lipreader_64_features = np.vstack((train_10pc_lipreader_64_features, val_10pc_lipreader_64_features, si131410_10pc_lipreader_64_features[12000:]))
# si1314_10pc_lipreader_64_features = si131410_10pc_lipreader_64_features[:12000]

# train_10pc_lipreader_64_features = train_val_10pc_lipreader_64_features[train_10pc_idx]
# val_10pc_lipreader_64_features = train_val_10pc_lipreader_64_features[val_10pc_idx]
# si1314_10pc_lipreader_64_features = si1314_10pc_lipreader_64_features[si1314_10pc_idx]

# np.savez('lipreader_10pc_64_features', train_10pc_lipreader_64_features=train_10pc_lipreader_64_features,
#     val_10pc_lipreader_64_features=val_10pc_lipreader_64_features, si1314_10pc_lipreader_64_features=si1314_10pc_lipreader_64_features)

lipreader_10pc_64_features = np.load('lipreader_10pc_64_features.npz')

train_10pc_lipreader_64_features = lipreader_10pc_64_features['train_10pc_lipreader_64_features']
val_10pc_lipreader_64_features = lipreader_10pc_64_features['val_10pc_lipreader_64_features']
si1314_10pc_lipreader_64_features = lipreader_10pc_64_features['si1314_10pc_lipreader_64_features']

#############################################################
# LOAD ATTRIBUTES
#############################################################

train_grid_attributes = np.load('train_grid_attributes_matrix.npy')
val_grid_attributes = np.load('val_grid_attributes_matrix.npy')
si131410_grid_attributes = np.load('si131410_grid_attributes_matrix.npy')

train_val_grid_attributes = np.vstack((train_grid_attributes, val_grid_attributes, si131410_grid_attributes[12000:]))
si1314_grid_attributes = si131410_grid_attributes[:12000]

train_10pc_grid_attributes = train_val_grid_attributes[train_10pc_idx]
val_10pc_grid_attributes = train_val_grid_attributes[val_10pc_idx]
si1314_10pc_grid_attributes = si1314_grid_attributes[si1314_10pc_idx]

# Replace with Lipreader_10pc predcicitions
train_10pc_grid_attributes[:, -64:] = train_10pc_lipreader_64_features
val_10pc_grid_attributes[:, -64:] = val_10pc_lipreader_64_features
si1314_10pc_grid_attributes[:, -64:] = si1314_10pc_lipreader_64_features

# Normalization
train_10pc_grid_attributes_peak_to_peak = train_10pc_grid_attributes.ptp(0)
train_10pc_grid_attributes_peak_to_peak[np.argwhere(train_10pc_grid_attributes_peak_to_peak == 0)] = 1
train_10pc_grid_attributes_norm = (train_10pc_grid_attributes - train_grid_attributes.min(0)) / train_10pc_grid_attributes_peak_to_peak
val_10pc_grid_attributes_peak_to_peak = val_10pc_grid_attributes.ptp(0)
val_10pc_grid_attributes_peak_to_peak[np.argwhere(val_10pc_grid_attributes_peak_to_peak == 0)] = 1
val_10pc_grid_attributes_norm = (val_10pc_grid_attributes - val_10pc_grid_attributes.min(0)) / val_10pc_grid_attributes_peak_to_peak
si1314_10pc_grid_attributes_peak_to_peak = si1314_10pc_grid_attributes.ptp(0)
si1314_10pc_grid_attributes_peak_to_peak[np.argwhere(si1314_10pc_grid_attributes_peak_to_peak == 0)] = 1
si1314_10pc_grid_attributes_norm = (si1314_10pc_grid_attributes - si1314_10pc_grid_attributes.min(0)) / si1314_10pc_grid_attributes_peak_to_peak

# Leave out the first three attributes
train_10pc_matrix = train_10pc_grid_attributes_norm[:, 3:]
val_10pc_matrix = val_10pc_grid_attributes_norm[:, 3:]
si1314_10pc_matrix = si1314_10pc_grid_attributes_norm[:, 3:]

# plt.imshow(np.abs(np.corrcoef(train_10pc_grid_attributes.T)), cmap='gray', clim=[0, 1]); plt.title("10pc Self-Train Attributes correlation"); plt.show()

# Pick only the labelled ones
train_10pc_labelled_matrix = train_10pc_matrix[train_10pc_labelled_idx]
train_10pc_unlabelled_matrix = train_10pc_matrix[train_10pc_unlabelled_idx]

########################################
# LIPREADER ROC
########################################




########################################
# LOGISTIC REGRESSOR ROC
########################################

logReg_10pc_unopt = LogisticRegression()
logReg_10pc_unopt.fit(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])

# Save
joblib.dump(logReg_10pc_unopt, 'logReg_10pc_unopt.pkl', compress=3)

# >>> # Acc
# ... logReg_10pc_unopt.score(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])
# 0.8651162790697674
# >>> logReg_10pc_unopt.score(train_10pc_unlabelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx])
# 0.86941243346597075
# >>> logReg_10pc_unopt.score(val_10pc_matrix, val_10pc_lipreader_preds_correct_or_wrong)
# 0.86795994993742176
# >>> logReg_10pc_unopt.score(si1314_10pc_matrix, si1314_10pc_lipreader_preds_correct_or_wrong)
# 0.38466666666666666

# Acc
logReg_10pc_unopt.score(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])
logReg_10pc_unopt.score(train_10pc_unlabelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx])
logReg_10pc_unopt.score(val_10pc_matrix, val_10pc_lipreader_preds_correct_or_wrong)
logReg_10pc_unopt.score(si1314_10pc_matrix, si1314_10pc_lipreader_preds_correct_or_wrong)

# CONFUSION MATRIX, OPERATING POINT
train_10pc_l_logReg_unopt_OP_fpr, train_10pc_l_logReg_unopt_OP_tpr, \
        train_10pc_unl_logReg_unopt_OP_fpr, train_10pc_unl_logReg_opt_OP_tpr, \
        val_10pc_logReg_unopt_OP_fpr, val_10pc_logReg_unopt_OP_tpr, \
        si1314_10pc_logReg_unopt_OP_fpr, si1314_10pc_logReg_unopt_OP_tpr = \
    calc_pc_grid_operating_points(logReg_unopt,
        train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx], train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx],
        val_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_lipreader_preds_correct_or_wrong,
        train_10pc_labelled_matrix, train_10pc_unlabelled_matrix, val_10pc_matrix, si1314_10pc_matrix)

# Scores
train_10pc_l_logReg_unopt_score = logReg_10pc_unopt.decision_function(train_10pc_labelled_matrix)
train_10pc_unl_logReg_unopt_score = logReg_10pc_unopt.decision_function(train_10pc_unlabelled_matrix)
val_10pc_logReg_unopt_score = logReg_10pc_unopt.decision_function(val_10pc_matrix)
si1314_10pc_logReg_unopt_score = logReg_10pc_unopt.decision_function(si1314_10pc_matrix)

# Compute ROC
train_10pc_l_logReg_unopt_fpr, train_10pc_l_logReg_unopt_tpr, train_10pc_l_logReg_unopt_thresholds, train_10pc_l_logReg_unopt_roc_auc, \
        train_10pc_unl_logReg_unopt_fpr, train_10pc_unl_logReg_unopt_tpr, train_10pc_unl_logReg_unopt_thresholds, train_10pc_unl_logReg_unopt_roc_auc, \
        val_10pc_logReg_unopt_fpr, val_10pc_logReg_unopt_tpr, val_10pc_logReg_unopt_thresholds, val_10pc_logReg_unopt_roc_auc, \
        si1314_10pc_logReg_unopt_fpr, si1314_10pc_logReg_unopt_tpr, si1314_10pc_logReg_unopt_thresholds, si1314_10pc_logReg_unopt_roc_auc = \
    compute_pc_ROC_grid_singleclass(train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx], train_10pc_l_logReg_unopt_score,
        train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx], train_10pc_unl_logReg_unopt_score,
        val_10pc_lipreader_preds_correct_or_wrong, val_10pc_logReg_unopt_score,
        si1314_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_logReg_unopt_score,
        train_10pc_l_logReg_unopt_OP_fpr, train_10pc_l_logReg_unopt_OP_tpr,
        train_10pc_unl_logReg_unopt_OP_fpr, train_10pc_unl_logReg_unopt_OP_tpr,
        val_10pc_logReg_unopt_OP_fpr, val_10pc_logReg_unopt_OP_tpr,
        si1314_10pc_logReg_unopt_OP_fpr, si1314_10pc_logReg_unopt_OP_tpr,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of 10pc Logistic Regressor (unoptimized)')


# FINDING OPTIMAL ROC OPERATING POINT

# Old fpr, tpr, acc
train_10pc_l_logReg_unopt_OP_fpr, train_10pc_l_logReg_unopt_OP_tpr
# (0.63129251700680267, 0.96746143057503509, 0.8651162790697674)
train_10pc_unl_logReg_unopt_OP_fpr, train_10pc_unl_logReg_unopt_OP_tpr
# (0.65632943568886626, 0.96399500015243433, 0.86941243346597075)
val_10pc_logReg_unopt_OP_fpr, val_10pc_logReg_unopt_OP_tpr
# (0.72423398328690802, 0.97227674190382729, 0.86795994993742176)
si1314_10pc_logReg_unopt_OP_fpr, si1314_10pc_logReg_unopt_OP_tpr
# (0.99932194195823165, 0.99675745784695202, 0.38466666666666666)

# Finding optimal point, accs
logReg_unopt_optimalOP_threshold, train_10pc_l_logReg_unopt_optimalOP_fpr, train_10pc_l_logReg_unopt_optimalOP_tpr, train_10pc_l_logReg_unopt_optimalOP_acc = \
    find_ROC_optimalOP(train_10pc_l_logReg_unopt_fpr, train_10pc_l_logReg_unopt_tpr, train_10pc_l_logReg_unopt_thresholds, train_10pc_l_logReg_unopt_score, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])

train_10pc_unl_logReg_unopt_optimalOP_fpr, train_10pc_unl_logReg_unopt_optimalOP_tpr, train_10pc_unl_logReg_unopt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx], train_10pc_unl_logReg_unopt_score, logReg_unopt_optimalOP_threshold)
val_10pc_logReg_unopt_optimalOP_fpr, val_10pc_logReg_unopt_optimalOP_tpr, val_10pc_logReg_unopt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(val_10pc_lipreader_preds_correct_or_wrong, val_10pc_logReg_unopt_score, logReg_unopt_optimalOP_threshold)
si1314_10pc_logReg_unopt_optimalOP_fpr, si1314_10pc_logReg_unopt_optimalOP_tpr, si1314_10pc_logReg_unopt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(si1314_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_logReg_unopt_score, logReg_unopt_optimalOP_threshold)

# New fpr, tpr, acc
train_10pc_l_logReg_unopt_optimalOP_fpr, train_10pc_l_logReg_unopt_optimalOP_tpr, train_10pc_l_logReg_unopt_optimalOP_acc
# (0.18095238095238095, 0.77054698457223003, 0.77883720930232558)
train_10pc_unl_logReg_unopt_optimalOP_fpr, train_10pc_unl_logReg_unopt_optimalOP_tpr, train_10pc_unl_logReg_unopt_optimalOP_acc
# (0.22301304863582444, 0.75833053870308831, 0.76117513306805851)
val_10pc_logReg_unopt_optimalOP_fpr, val_10pc_logReg_unopt_optimalOP_tpr, val_10pc_logReg_unopt_optimalOP_acc
# (0.2618384401114206, 0.79317958783120701, 0.78493950771798082)
si1314_10pc_logReg_unopt_optimalOP_fpr, si1314_10pc_logReg_unopt_optimalOP_tpr, si1314_10pc_logReg_unopt_optimalOP_acc
# (0.90439381611065905, 0.90834414180717682, 0.40891666666666665)

plot_pc_grid_ROC(train_10pc_l_logReg_unopt_fpr, train_10pc_l_logReg_unopt_tpr, train_10pc_l_logReg_unopt_roc_auc,
        train_10pc_unl_logReg_unopt_fpr, train_10pc_unl_logReg_unopt_tpr, train_10pc_unl_logReg_unopt_roc_auc,
        val_10pc_logReg_unopt_fpr, val_10pc_logReg_unopt_tpr, val_10pc_logReg_unopt_roc_auc,
        si1314_10pc_logReg_unopt_fpr, si1314_10pc_logReg_unopt_tpr, si1314_10pc_logReg_unopt_roc_auc,
        train_l_OP_fpr=train_10pc_l_logReg_unopt_OP_fpr, train_l_OP_tpr=train_10pc_l_logReg_unopt_OP_tpr,
        train_unl_OP_fpr=train_10pc_unl_logReg_unopt_OP_fpr, train_unl_OP_tpr=train_10pc_unl_logReg_unopt_OP_tpr,
        val_OP_fpr=val_10pc_logReg_unopt_OP_fpr, val_OP_tpr=val_10pc_logReg_unopt_OP_tpr,
        si_OP_fpr=si1314_10pc_logReg_unopt_OP_fpr, si_OP_tpr=si1314_10pc_logReg_unopt_OP_tpr,
        train_l_optimalOP_fpr=train_10pc_l_logReg_unopt_optimalOP_fpr, train_l_optimalOP_tpr=train_10pc_l_logReg_unopt_optimalOP_tpr,
        train_unl_optimalOP_fpr=train_10pc_unl_logReg_unopt_optimalOP_fpr, train_unl_optimalOP_tpr=train_10pc_unl_logReg_unopt_optimalOP_tpr,
        val_optimalOP_fpr=val_10pc_logReg_unopt_optimalOP_fpr, val_optimalOP_tpr=val_10pc_logReg_unopt_optimalOP_tpr,
        si_optimalOP_fpr=si1314_10pc_logReg_unopt_optimalOP_fpr, si_optimalOP_tpr=si1314_10pc_logReg_unopt_optimalOP_tpr,
        plot_title='ROC curve of 10pc Logistic Regressor (unoptimized)')

# Save
np.savez('ROC_10pc_logReg_unopt',
    train_10pc_l_logReg_unopt_score=train_10pc_l_logReg_unopt_score, train_10pc_unl_logReg_unopt_score=train_10pc_unl_logReg_unopt_score, val_10pc_logReg_unopt_score=val_10pc_logReg_unopt_score, si1314_10pc_logReg_unopt_score=si1314_10pc_logReg_unopt_score,
    train_10pc_l_logReg_unopt_fpr=train_10pc_l_logReg_unopt_fpr, train_10pc_l_logReg_unopt_tpr=train_10pc_l_logReg_unopt_tpr, train_10pc_l_logReg_unopt_thresholds=train_10pc_l_logReg_unopt_thresholds, train_10pc_l_logReg_unopt_roc_auc=train_10pc_l_logReg_unopt_roc_auc,
    train_10pc_unl_logReg_unopt_fpr=train_10pc_unl_logReg_unopt_fpr, train_10pc_unl_logReg_unopt_tpr=train_10pc_unl_logReg_unopt_tpr, train_10pc_unl_logReg_unopt_thresholds=train_10pc_unl_logReg_unopt_thresholds, train_10pc_unl_logReg_unopt_roc_auc=train_10pc_unl_logReg_unopt_roc_auc,
    val_10pc_logReg_unopt_fpr=val_10pc_logReg_unopt_fpr, val_10pc_logReg_unopt_tpr=val_10pc_logReg_unopt_tpr, val_10pc_logReg_unopt_thresholds=val_10pc_logReg_unopt_thresholds, val_10pc_logReg_unopt_roc_auc=val_10pc_logReg_unopt_roc_auc,
    si1314_10pc_logReg_unopt_fpr=si1314_10pc_logReg_unopt_fpr, si1314_10pc_logReg_unopt_tpr=si1314_10pc_logReg_unopt_tpr, si1314_10pc_logReg_unopt_thresholds=si1314_10pc_logReg_unopt_thresholds, si1314_10pc_logReg_unopt_roc_auc=si1314_10pc_logReg_unopt_roc_auc,
    train_10pc_l_logReg_unopt_OP_fpr=train_10pc_l_logReg_unopt_OP_fpr, train_10pc_l_logReg_unopt_OP_tpr=train_10pc_l_logReg_unopt_OP_tpr,
    train_10pc_unl_logReg_unopt_OP_fpr=train_10pc_unl_logReg_unopt_OP_fpr, train_10pc_unl_logReg_unopt_OP_tpr=train_10pc_unl_logReg_unopt_OP_tpr,
    val_10pc_logReg_unopt_OP_fpr=val_10pc_logReg_unopt_OP_fpr, val_10pc_logReg_unopt_OP_tpr=val_10pc_logReg_unopt_OP_tpr,
    si1314_10pc_logReg_unopt_OP_fpr=si1314_10pc_logReg_unopt_OP_fpr, si1314_10pc_logReg_unopt_OP_tpr=si1314_10pc_logReg_unopt_OP_tpr,
    logReg_unopt_optimalOP_threshold=logReg_unopt_optimalOP_threshold,
    train_10pc_l_logReg_unopt_optimalOP_fpr=train_10pc_l_logReg_unopt_optimalOP_fpr, train_10pc_l_logReg_unopt_optimalOP_tpr=train_10pc_l_logReg_unopt_optimalOP_tpr,
    train_10pc_unl_logReg_unopt_optimalOP_fpr=train_10pc_unl_logReg_unopt_optimalOP_fpr, train_10pc_unl_logReg_unopt_optimalOP_tpr=train_10pc_unl_logReg_unopt_optimalOP_tpr,
    val_10pc_logReg_unopt_optimalOP_fpr=val_10pc_logReg_unopt_optimalOP_fpr, val_10pc_logReg_unopt_optimalOP_tpr=val_10pc_logReg_unopt_optimalOP_tpr,
    si1314_10pc_logReg_unopt_optimalOP_fpr=si1314_10pc_logReg_unopt_optimalOP_fpr, si1314_10pc_logReg_unopt_optimalOP_tpr=si1314_10pc_logReg_unopt_optimalOP_tpr)


###########################
# OPT
###########################


# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=train_10pc_labelled_matrix, y=train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx], num_folds=2, num_iter=1)
def logReg_auc(x_train, y_train, x_test, y_test, logC):
    model = LogisticRegression(C=10 ** logC, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

hps_logReg, _, _ = optunity.maximize(logReg_auc, num_evals=10, logC=[-5, 2])

logReg_10pc_opt = LogisticRegression(C=10 ** hps_logReg['logC'], class_weight='balanced').fit(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])

# Save
joblib.dump(logReg_10pc_opt, 'logReg_10pc_opt.pkl', compress=3)

# Acc
logReg_10pc_opt.score(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])
logReg_10pc_opt.score(train_10pc_unlabelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx])
logReg_10pc_opt.score(val_10pc_matrix, val_10pc_lipreader_preds_correct_or_wrong)
logReg_10pc_opt.score(si1314_10pc_matrix, si1314_10pc_lipreader_preds_correct_or_wrong)
# >>> # Acc
# ... logReg_10pc_opt.score(train_10pc_labelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])
# 0.78186046511627905
# >>> logReg_10pc_opt.score(train_10pc_unlabelled_matrix, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx])
# 0.76151103302154932
# >>> logReg_10pc_opt.score(val_10pc_matrix, val_10pc_lipreader_preds_correct_or_wrong)
# 0.7882770129328327
# >>> logReg_10pc_opt.score(si1314_10pc_matrix, si1314_10pc_lipreader_preds_correct_or_wrong)
# 0.41849999999999998

# CONFUSION MATRIX, OPERATING POINT
train_10pc_l_logReg_opt_OP_fpr, train_10pc_l_logReg_opt_OP_tpr, \
        train_10pc_unl_logReg_opt_OP_fpr, train_10pc_unl_logReg_opt_OP_tpr, \
        val_10pc_logReg_opt_OP_fpr, val_10pc_logReg_opt_OP_tpr, \
        si1314_10pc_logReg_opt_OP_fpr, si1314_10pc_logReg_opt_OP_tpr = \
    calc_pc_grid_operating_points(logReg_10pc_opt,
        train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx], train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx],
        val_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_lipreader_preds_correct_or_wrong,
        train_10pc_labelled_matrix, train_10pc_unlabelled_matrix, val_10pc_matrix, si1314_10pc_matrix)

# Scores
train_10pc_l_logReg_opt_score = logReg_10pc_opt.decision_function(train_10pc_labelled_matrix)
train_10pc_unl_logReg_opt_score = logReg_10pc_opt.decision_function(train_10pc_unlabelled_matrix)
val_10pc_logReg_opt_score = logReg_10pc_opt.decision_function(val_10pc_matrix)
si1314_10pc_logReg_opt_score = logReg_10pc_opt.decision_function(si1314_10pc_matrix)

# Compute ROC
train_10pc_l_logReg_opt_fpr, train_10pc_l_logReg_opt_tpr, train_10pc_l_logReg_opt_thresholds, train_10pc_l_logReg_opt_roc_auc, \
        train_10pc_unl_logReg_opt_fpr, train_10pc_unl_logReg_opt_tpr, train_10pc_unl_logReg_opt_thresholds, train_10pc_unl_logReg_opt_roc_auc, \
        val_10pc_logReg_opt_fpr, val_10pc_logReg_opt_tpr, val_10pc_logReg_opt_thresholds, val_10pc_logReg_opt_roc_auc, \
        si1314_10pc_logReg_opt_fpr, si1314_10pc_logReg_opt_tpr, si1314_10pc_logReg_opt_thresholds, si1314_10pc_logReg_opt_roc_auc = \
    compute_pc_ROC_grid_singleclass(train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx], train_10pc_l_logReg_opt_score,
        train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx], train_10pc_unl_logReg_opt_score,
        val_10pc_lipreader_preds_correct_or_wrong, val_10pc_logReg_opt_score,
        si1314_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_logReg_opt_score,
        train_10pc_l_logReg_opt_OP_fpr, train_10pc_l_logReg_opt_OP_tpr,
        train_10pc_unl_logReg_opt_OP_fpr, train_10pc_unl_logReg_opt_OP_tpr,
        val_10pc_logReg_opt_OP_fpr, val_10pc_logReg_opt_OP_tpr,
        si1314_10pc_logReg_opt_OP_fpr, si1314_10pc_logReg_opt_OP_tpr,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of 10pc Logistic Regressor (optimized)')


# FINDING OPTIMAL ROC OPERATING POINT

# Old fpr, tpr, acc
train_10pc_l_logReg_opt_OP_fpr, train_10pc_l_logReg_opt_OP_tpr
# (0.16870748299319727, 0.77166900420757367, 0.78186046511627905)
train_10pc_unl_logReg_opt_OP_fpr, train_10pc_unl_logReg_opt_OP_tpr
# (0.2172513133367226, 0.75769031431968537, 0.76151103302154932)
val_10pc_logReg_opt_OP_fpr, val_10pc_logReg_opt_OP_tpr
# (0.25348189415041783, 0.79563297350343476, 0.7882770129328327)
si1314_10pc_logReg_opt_OP_fpr, si1314_10pc_logReg_opt_OP_tpr
# (0.88744236506644969, 0.90618244703847817, 0.41849999999999998)

# Finding optimal point, accs
logReg_opt_optimalOP_threshold, train_10pc_l_logReg_opt_optimalOP_fpr, train_10pc_l_logReg_opt_optimalOP_tpr, train_10pc_l_logReg_opt_optimalOP_acc = \
    find_ROC_optimalOP(train_10pc_l_logReg_opt_fpr, train_10pc_l_logReg_opt_tpr, train_10pc_l_logReg_opt_thresholds, train_10pc_l_logReg_opt_score, train_10pc_lipreader_preds_correct_or_wrong[train_10pc_labelled_idx])

train_10pc_unl_logReg_opt_optimalOP_fpr, train_10pc_unl_logReg_opt_optimalOP_tpr, train_10pc_unl_logReg_opt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(train_10pc_lipreader_preds_correct_or_wrong[train_10pc_unlabelled_idx], train_10pc_unl_logReg_opt_score, logReg_opt_optimalOP_threshold)
val_10pc_logReg_opt_optimalOP_fpr, val_10pc_logReg_opt_optimalOP_tpr, val_10pc_logReg_opt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(val_10pc_lipreader_preds_correct_or_wrong, val_10pc_logReg_opt_score, logReg_opt_optimalOP_threshold)
si1314_10pc_logReg_opt_optimalOP_fpr, si1314_10pc_logReg_opt_optimalOP_tpr, si1314_10pc_logReg_opt_optimalOP_acc = find_fpr_tpr_acc_from_thresh(si1314_10pc_lipreader_preds_correct_or_wrong, si1314_10pc_logReg_opt_score, logReg_opt_optimalOP_threshold)

# New fpr, tpr, acc
train_10pc_l_logReg_opt_optimalOP_fpr, train_10pc_l_logReg_opt_optimalOP_tpr, train_10pc_l_logReg_opt_optimalOP_acc
# (0.19727891156462585, 0.80392706872370268, 0.80372093023255819)
train_10pc_unl_logReg_opt_optimalOP_fpr, train_10pc_unl_logReg_opt_optimalOP_tpr, train_10pc_unl_logReg_opt_optimalOP_acc
# (0.2448737502118285, 0.78482363342581019, 0.78029559195907183)
val_10pc_logReg_opt_optimalOP_fpr, val_10pc_logReg_opt_optimalOP_tpr, val_10pc_logReg_opt_optimalOP_acc
# (0.28272980501392758, 0.81746810598626107, 0.80246141009595329)
si1314_10pc_logReg_opt_optimalOP_fpr, si1314_10pc_logReg_opt_optimalOP_tpr, si1314_10pc_logReg_opt_optimalOP_acc
# (0.91049633848657441, 0.92931258106355386, 0.41325000000000001)

plot_pc_grid_ROC(train_10pc_l_logReg_opt_fpr, train_10pc_l_logReg_opt_tpr, train_10pc_l_logReg_opt_roc_auc,
        train_10pc_unl_logReg_opt_fpr, train_10pc_unl_logReg_opt_tpr, train_10pc_unl_logReg_opt_roc_auc,
        val_10pc_logReg_opt_fpr, val_10pc_logReg_opt_tpr, val_10pc_logReg_opt_roc_auc,
        si1314_10pc_logReg_opt_fpr, si1314_10pc_logReg_opt_tpr, si1314_10pc_logReg_opt_roc_auc,
        train_l_OP_fpr=train_10pc_l_logReg_opt_OP_fpr, train_l_OP_tpr=train_10pc_l_logReg_opt_OP_tpr,
        train_unl_OP_fpr=train_10pc_unl_logReg_opt_OP_fpr, train_unl_OP_tpr=train_10pc_unl_logReg_opt_OP_tpr,
        val_OP_fpr=val_10pc_logReg_opt_OP_fpr, val_OP_tpr=val_10pc_logReg_opt_OP_tpr,
        si_OP_fpr=si1314_10pc_logReg_opt_OP_fpr, si_OP_tpr=si1314_10pc_logReg_opt_OP_tpr,
        train_l_optimalOP_fpr=train_10pc_l_logReg_opt_optimalOP_fpr, train_l_optimalOP_tpr=train_10pc_l_logReg_opt_optimalOP_tpr,
        train_unl_optimalOP_fpr=train_10pc_unl_logReg_opt_optimalOP_fpr, train_unl_optimalOP_tpr=train_10pc_unl_logReg_opt_optimalOP_tpr,
        val_optimalOP_fpr=val_10pc_logReg_opt_optimalOP_fpr, val_optimalOP_tpr=val_10pc_logReg_opt_optimalOP_tpr,
        si_optimalOP_fpr=si1314_10pc_logReg_opt_optimalOP_fpr, si_optimalOP_tpr=si1314_10pc_logReg_opt_optimalOP_tpr,
        plot_title='ROC curve of 10pc Logistic Regressor (optimized)')

# Save
np.savez('ROC_10pc_logReg_opt',
    train_10pc_l_logReg_opt_score=train_10pc_l_logReg_opt_score, train_10pc_unl_logReg_opt_score=train_10pc_unl_logReg_opt_score, val_10pc_logReg_opt_score=val_10pc_logReg_opt_score, si1314_10pc_logReg_opt_score=si1314_10pc_logReg_opt_score,
    train_10pc_l_logReg_opt_fpr=train_10pc_l_logReg_opt_fpr, train_10pc_l_logReg_opt_tpr=train_10pc_l_logReg_opt_tpr, train_10pc_l_logReg_opt_thresholds=train_10pc_l_logReg_opt_thresholds, train_10pc_l_logReg_opt_roc_auc=train_10pc_l_logReg_opt_roc_auc,
    train_10pc_unl_logReg_opt_fpr=train_10pc_unl_logReg_opt_fpr, train_10pc_unl_logReg_opt_tpr=train_10pc_unl_logReg_opt_tpr, train_10pc_unl_logReg_opt_thresholds=train_10pc_unl_logReg_opt_thresholds, train_10pc_unl_logReg_opt_roc_auc=train_10pc_unl_logReg_opt_roc_auc,
    val_10pc_logReg_opt_fpr=val_10pc_logReg_opt_fpr, val_10pc_logReg_opt_tpr=val_10pc_logReg_opt_tpr, val_10pc_logReg_opt_thresholds=val_10pc_logReg_opt_thresholds, val_10pc_logReg_opt_roc_auc=val_10pc_logReg_opt_roc_auc,
    si1314_10pc_logReg_opt_fpr=si1314_10pc_logReg_opt_fpr, si1314_10pc_logReg_opt_tpr=si1314_10pc_logReg_opt_tpr, si1314_10pc_logReg_opt_thresholds=si1314_10pc_logReg_opt_thresholds, si1314_10pc_logReg_opt_roc_auc=si1314_10pc_logReg_opt_roc_auc,
    train_10pc_l_logReg_opt_OP_fpr=train_10pc_l_logReg_opt_OP_fpr, train_10pc_l_logReg_opt_OP_tpr=train_10pc_l_logReg_opt_OP_tpr,
    train_10pc_unl_logReg_opt_OP_fpr=train_10pc_unl_logReg_opt_OP_fpr, train_10pc_unl_logReg_opt_OP_tpr=train_10pc_unl_logReg_opt_OP_tpr,
    val_10pc_logReg_opt_OP_fpr=val_10pc_logReg_opt_OP_fpr, val_10pc_logReg_opt_OP_tpr=val_10pc_logReg_opt_OP_tpr,
    si1314_10pc_logReg_opt_OP_fpr=si1314_10pc_logReg_opt_OP_fpr, si1314_10pc_logReg_opt_OP_tpr=si1314_10pc_logReg_opt_OP_tpr,
    logReg_opt_optimalOP_threshold=logReg_opt_optimalOP_threshold,
    train_10pc_l_logReg_opt_optimalOP_fpr=train_10pc_l_logReg_opt_optimalOP_fpr, train_10pc_l_logReg_opt_optimalOP_tpr=train_10pc_l_logReg_opt_optimalOP_tpr,
    train_10pc_unl_logReg_opt_optimalOP_fpr=train_10pc_unl_logReg_opt_optimalOP_fpr, train_10pc_unl_logReg_opt_optimalOP_tpr=train_10pc_unl_logReg_opt_optimalOP_tpr,
    val_10pc_logReg_opt_optimalOP_fpr=val_10pc_logReg_opt_optimalOP_fpr, val_10pc_logReg_opt_optimalOP_tpr=val_10pc_logReg_opt_optimalOP_tpr,
    si1314_10pc_logReg_opt_optimalOP_fpr=si1314_10pc_logReg_opt_optimalOP_fpr, si1314_10pc_logReg_opt_optimalOP_tpr=si1314_10pc_logReg_opt_optimalOP_tpr)

# Load
logReg_opt = joblib.load('logReg_opt.pkl')
ROC_10pc_logReg_opt = np.load('ROC_10pc_logReg_opt.npz')
train_10pc_logReg_opt_score, val_10pc_logReg_opt_score, si_10pc_logReg_opt_score, \
        train_10pc_logReg_opt_fpr, train_10pc_logReg_opt_tpr, train_10pc_logReg_opt_thresholds, train_10pc_logReg_opt_roc_auc, \
        val_10pc_logReg_opt_fpr, val_10pc_logReg_opt_tpr, val_10pc_logReg_opt_thresholds, val_10pc_logReg_opt_roc_auc, \
        si_10pc_logReg_opt_fpr, si_10pc_logReg_opt_tpr, si_10pc_logReg_opt_thresholds, si_10pc_logReg_opt_roc_auc , \
        train_10pc_logReg_opt_OP_fpr, train_10pc_logReg_opt_OP_tpr, \
        val_10pc_logReg_opt_OP_fpr, val_10pc_logReg_opt_OP_tpr, \
        si_10pc_logReg_opt_OP_fpr, si_10pc_logReg_opt_OP_tpr, \
        train_10pc_logReg_opt_optimalOP_fpr, train_10pc_logReg_opt_optimalOP_tpr, \
        val_10pc_logReg_opt_optimalOP_fpr, val_10pc_logReg_opt_optimalOP_tpr, \
        si_10pc_logReg_opt_optimalOP_fpr, si_10pc_logReg_opt_optimalOP_tpr = \
    ROC_10pc_logReg_opt['train_10pc_logReg_opt_score'], ROC_10pc_logReg_opt['val_10pc_logReg_opt_score'], ROC_10pc_logReg_opt['si_10pc_logReg_opt_score'], \
        ROC_10pc_logReg_opt['train_10pc_logReg_opt_fpr'], ROC_10pc_logReg_opt['train_10pc_logReg_opt_tpr'], ROC_10pc_logReg_opt['train_10pc_logReg_opt_thresholds'], ROC_10pc_logReg_opt['train_10pc_logReg_opt_roc_auc'].item(), \
        ROC_10pc_logReg_opt['val_10pc_logReg_opt_fpr'], ROC_10pc_logReg_opt['val_10pc_logReg_opt_tpr'], ROC_10pc_logReg_opt['val_10pc_logReg_opt_thresholds'], ROC_10pc_logReg_opt['val_10pc_logReg_opt_roc_auc'].item(), \
        ROC_10pc_logReg_opt['si_10pc_logReg_opt_fpr'], ROC_10pc_logReg_opt['si_10pc_logReg_opt_tpr'], ROC_10pc_logReg_opt['si_10pc_logReg_opt_thresholds'], ROC_10pc_logReg_opt['si_10pc_logReg_opt_roc_auc'].item(), \
        ROC_10pc_logReg_opt['train_10pc_logReg_opt_OP_fpr'].item(), ROC_10pc_logReg_opt['train_10pc_logReg_opt_OP_tpr'].item(), \
        ROC_10pc_logReg_opt['val_10pc_logReg_opt_OP_fpr'].item(), ROC_10pc_logReg_opt['val_10pc_logReg_opt_OP_tpr'].item(), \
        ROC_10pc_logReg_opt['si_10pc_logReg_opt_OP_fpr'].item(), ROC_10pc_logReg_opt['si_10pc_logReg_opt_OP_tpr'].item(), \
        ROC_10pc_logReg_opt['train_10pc_logReg_opt_optimalOP_fpr'].item(), ROC_10pc_logReg_opt['train_10pc_logReg_opt_optimalOP_tpr'].item(), \
        ROC_10pc_logReg_opt['val_10pc_logReg_opt_optimalOP_fpr'].item(), ROC_10pc_logReg_opt['val_10pc_logReg_opt_optimalOP_tpr'].item(), \
        ROC_10pc_logReg_opt['si_10pc_logReg_opt_optimalOP_fpr'].item(), ROC_10pc_logReg_opt['si_10pc_logReg_opt_optimalOP_tpr'].item()



