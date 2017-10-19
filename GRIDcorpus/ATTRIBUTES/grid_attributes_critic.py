import numpy as np
import optunity
import optunity.metrics

from sklearn.svm import SVC

########################################
# ATTRIBUTES
########################################

# Speaker identity

# Male or not

# Bilabial or not

# Duration of word

# Head Pose Mean (3)

# Head Pose Range (3)

#############################################################
# LOAD CORRECT_OR_NOT
#############################################################

lipreader_preds_wordidx_and_correctorwrong = np.load('lipreader_preds_wordidx_and_correctorwrong.npy').item()

train_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_word_idx']
val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_word_idx']
si_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_word_idx']

train_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_correct_or_wrong']
val_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_correct_or_wrong']
si_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_correct_or_wrong']

np.sum(train_lipreader_preds_correct_or_wrong)/len(train_lipreader_preds_correct_or_wrong)
np.sum(val_lipreader_preds_correct_or_wrong)/len(val_lipreader_preds_correct_or_wrong)
np.sum(si_lipreader_preds_correct_or_wrong)/len(si_lipreader_preds_correct_or_wrong)

#############################################################
# LOAD ATTRIBUTES
#############################################################

train_grid_attributes = np.load('train_grid_attributes_matrix.npy')
val_grid_attributes = np.load('val_grid_attributes_matrix.npy')
si_grid_attributes = np.load('si_grid_attributes_matrix.npy')

# Normalization
train_grid_attributes_matrix = (train_grid_attributes - train_grid_attributes.min(0)) / train_grid_attributes.ptp(0)
val_grid_attributes_matrix = (val_grid_attributes - val_grid_attributes.min(0)) / val_grid_attributes.ptp(0)
si_grid_attributes_matrix = (si_grid_attributes - si_grid_attributes.min(0)) / si_grid_attributes.ptp(0)

#############################################################
# LOAD LIPREADER FEATURES
#############################################################



#############################################################
# TRAIN SVM
#############################################################

# LINEAR
clf = SVC(kernel='linear', class_weight='balanced')

clf.fit(train_matrix[:, 3:], train_lipreader_preds_correct_or_wrong)
clf.score(train_matrix[:, 3:], train_lipreader_preds_correct_or_wrong)
clf.score(val_matrix[:, 3:], val_lipreader_preds_correct_or_wrong)
clf.score(si_matrix[:, 3:], si_lipreader_preds_correct_or_wrong)


# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=train_matrix[:, 3:], y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
def svm_linear_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)


@optunity.cross_validated(x=train_matrix[:, 3:], y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
def svm_rbf_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='rbf', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps_linear, _, _ = optunity.maximize(svm_linear_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
hps_linear

hps_rbf, _, _ = optunity.maximize(svm_rbf_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_SVM_linear = SVC(kernel='linear', C=10 ** hps_linear['logC'], gamma=10 ** hps_linear['logGamma'], class_weight='balanced').fit(train_matrix[:, 3:], train_lipreader_preds_correct_or_wrong)

optimal_SVM_linear_w = SVC(kernel='linear', C=10 ** hps_rbf['logC'], gamma=10 ** hps_rbf['logGamma'], class_weight='balanced').fit(train_matrix[:, 3:], train_lipreader_preds_correct_or_wrong)


optimal_SVM_linear.score(train_matrix[:, 3:], train_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(val_matrix[:, 3:], val_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(si_matrix[:, 3:], si_lipreader_preds_correct_or_wrong)


optimal_SVM_linear_w.score(train_matrix, train_lipreader_preds_correct_or_wrong)
optimal_SVM_linear_w.score(val_matrix, val_lipreader_preds_correct_or_wrong)
optimal_SVM_linear_w.score(si_matrix, si_lipreader_preds_correct_or_wrong)
