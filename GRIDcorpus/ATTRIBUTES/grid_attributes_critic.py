import numpy as np
import optunity
import optunity.metrics

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from scipy import interp

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

grid_attributes_dict = np.load(os.path.join(GRID_ATTR_DIR, 'grid_attributes_dict.npy')).item()

train_dirs = grid_attributes_dict['train_dirs']
train_word_numbers = grid_attributes_dict['train_word_numbers']
train_word_idx = grid_attributes_dict['train_word_idx']

val_dirs = grid_attributes_dict['val_dirs']
val_word_numbers = grid_attributes_dict['val_word_numbers']
val_word_idx = grid_attributes_dict['val_word_idx']

si_dirs = grid_attributes_dict['si_dirs']
si_word_numbers = grid_attributes_dict['si_word_numbers']
si_word_idx = grid_attributes_dict['si_word_idx']

#############################################################
# LOAD ATTRIBUTES
#############################################################

train_grid_attributes = np.load('train_grid_attributes_matrix.npy')
val_grid_attributes = np.load('val_grid_attributes_matrix.npy')
si_grid_attributes = np.load('si_grid_attributes_matrix.npy')

# Normalization
train_grid_attributes_norm = (train_grid_attributes - train_grid_attributes.min(0)) / train_grid_attributes.ptp(0)
val_grid_attributes_norm = (val_grid_attributes - val_grid_attributes.min(0)) / val_grid_attributes.ptp(0)
si_grid_attributes_norm = (si_grid_attributes - si_grid_attributes.min(0)) / si_grid_attributes.ptp(0); si_grid_attributes_matrix[:, 1] = 1

#############################################################
# LOAD PREDS
#############################################################

lipreader_preds = np.load('lipreader_preds.npz')

train_lipreader_preds = lipreader_preds['train_lipreader_preds']
val_lipreader_preds = lipreader_preds['val_lipreader_preds']
si_lipreader_preds = lipreader_preds['si_lipreader_preds']

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
# LIPREADER ROC
#############################################################

train_fpr = {}
train_tpr = {}
train_roc_auc = {}
val_fpr = {}
val_tpr = {}
val_roc_auc = {}
si_fpr = {}
si_tpr = {}
si_roc_auc = {}

# Train
train_lipreader_preds_confidence = np.max(train_lipreader_preds, axis=1)
# Val
val_lipreader_preds_confidence = np.max(val_lipreader_preds, axis=1)
# Si
si_lipreader_preds_confidence = np.max(si_lipreader_preds, axis=1)

# MICRO ROC
# Train
train_fpr["micro"], train_tpr["micro"], _ = roc_curve(train_lipreader_preds_correct_or_wrong, train_lipreader_preds_confidence)
train_roc_auc["micro"] = auc(train_fpr["micro"], train_tpr["micro"])
# Val
val_fpr["micro"], val_tpr["micro"], _ = roc_curve(val_lipreader_preds_correct_or_wrong, val_lipreader_preds_confidence)
val_roc_auc["micro"] = auc(val_fpr["micro"], val_tpr["micro"])
# Si
si_fpr["micro"], si_tpr["micro"], _ = roc_curve(si_lipreader_preds_correct_or_wrong, si_lipreader_preds_confidence)
si_roc_auc["micro"] = auc(si_fpr["micro"], si_tpr["micro"])

# MACRO ROC
for i in range(len(GRID_VOCAB_FULL)):
    # TRAIN
    train_word_binary = np.where(train_word_idx == i)
    train_fpr[i], train_tpr[i], _ = roc_curve(train_lipreader_preds_correct_or_wrong[train_word_binary], train_lipreader_preds_confidence[train_word_binary])
    train_roc_auc[i] = auc(train_fpr[i], train_tpr[i])
    # VAL
    val_word_binary = np.where(val_word_idx == i)
    val_fpr[i], val_tpr[i], _ = roc_curve(val_lipreader_preds_correct_or_wrong[val_word_binary], val_lipreader_preds_confidence[val_word_binary])
    # To save from all_True, i.e. fpr = nan
    val_fpr[i][np.argwhere(np.isnan(val_fpr[i]))] = 1.
    val_roc_auc[i] = auc(val_fpr[i], val_tpr[i])
    # SI
    si_word_binary = np.where(si_word_idx == i)
    si_fpr[i], si_tpr[i], _ = roc_curve(si_lipreader_preds_correct_or_wrong[si_word_binary], si_lipreader_preds_confidence[si_word_binary])
    si_roc_auc[i] = auc(si_fpr[i], si_tpr[i])


# First aggregate all false positive rates
train_all_fpr = np.unique(np.concatenate([train_fpr[i] for i in range(len(GRID_VOCAB_FULL))]))
val_all_fpr = np.unique(np.concatenate([val_fpr[i] for i in range(len(GRID_VOCAB_FULL))]))
si_all_fpr = np.unique(np.concatenate([si_fpr[i] for i in range(len(GRID_VOCAB_FULL))]))

# Then interpolate all ROC curves at this points
train_mean_tpr = np.zeros_like(train_all_fpr)
val_mean_tpr = np.zeros_like(val_all_fpr)
si_mean_tpr = np.zeros_like(si_all_fpr)
for i in range(len(GRID_VOCAB_FULL)):
    train_mean_tpr += interp(train_all_fpr, train_fpr[i], train_tpr[i])
    val_mean_tpr += interp(val_all_fpr, val_fpr[i], val_tpr[i])
    si_mean_tpr += interp(si_all_fpr, si_fpr[i], si_tpr[i])

# Finally average it
train_mean_tpr /= len(GRID_VOCAB_FULL)
val_mean_tpr /= len(GRID_VOCAB_FULL)
si_mean_tpr /= len(GRID_VOCAB_FULL)

# Compute AUC
train_fpr["macro"] = train_all_fpr
val_fpr["macro"] = val_all_fpr
si_fpr["macro"] = si_all_fpr
train_tpr["macro"] = train_mean_tpr
val_tpr["macro"] = val_mean_tpr
si_tpr["macro"] = si_mean_tpr
train_roc_auc["macro"] = auc(train_fpr["macro"], train_tpr["macro"])
val_roc_auc["macro"] = auc(val_fpr["macro"], val_tpr["macro"])
si_roc_auc["macro"] = auc(si_fpr["macro"], si_tpr["macro"])

plt.plot(train_fpr['micro'], train_tpr['micro'], color='C0', linestyle='-', label='micro_train')
plt.plot(train_fpr['macro'], train_tpr['macro'], color='C0', linestyle='--', label='macro_train')
plt.plot(val_fpr['micro'], val_tpr['micro'], color='C1', linestyle='-', label='micro_val')
plt.plot(val_fpr['macro'], val_tpr['macro'], color='C1', linestyle='--', label='macro_val')
plt.plot(si_fpr['micro'], si_tpr['micro'], color='C2', linestyle='-', label='micro_si')
plt.plot(si_fpr['macro'], si_tpr['macro'], color='C2', linestyle='--', label='macro_si')
plt.legend(loc='best')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Baseline ROC curve of Lipreader')
plt.show()

np.savez('ROC_baseline_lipreader', train_fpr, train_tpr, val_fpr, val_tpr, si_fpr, si_tpr)


#############################################################
# TRAIN SVM
#############################################################

train_matrix = train_grid_attributes_norm[:, 3:]
val_matrix = val_grid_attributes_norm[:, 3:]
si_matrix = si_grid_attributes_norm[:, 3:]

# LINEAR
clf = SVC(kernel='linear', class_weight='balanced')

clf.fit(train_matrix, train_lipreader_preds_correct_or_wrong)
clf.score(train_matrix, train_lipreader_preds_correct_or_wrong)
clf.score(val_matrix, val_lipreader_preds_correct_or_wrong)
clf.score(si_matrix, si_lipreader_preds_correct_or_wrong)


# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=train_matrix, y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
def svm_linear_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)


@optunity.cross_validated(x=train_matrix, y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
def svm_rbf_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='rbf', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning on linear
# hps_linear, _, _ = optunity.maximize(svm_linear_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
hps_linear = {'logC': 1.14892578125, 'logGamma': -4.9794921875}

# perform tuning on rbf
hps_rbf, _, _ = optunity.maximize(svm_rbf_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_SVM_linear = SVC(kernel='linear', C=10 ** hps_linear['logC'], gamma=10 ** hps_linear['logGamma'], class_weight='balanced').fit(train_matrix, train_lipreader_preds_correct_or_wrong)

optimal_SVM_linear_w = SVC(kernel='linear', C=10 ** hps_rbf['logC'], gamma=10 ** hps_rbf['logGamma'], class_weight='balanced').fit(train_matrix, train_lipreader_preds_correct_or_wrong)


optimal_SVM_linear.score(train_matrix, train_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(val_matrix, val_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(si_matrix, si_lipreader_preds_correct_or_wrong)


optimal_SVM_linear_w.score(train_matrix, train_lipreader_preds_correct_or_wrong)
optimal_SVM_linear_w.score(val_matrix, val_lipreader_preds_correct_or_wrong)
optimal_SVM_linear_w.score(si_matrix, si_lipreader_preds_correct_or_wrong)







