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

# LipreaderEncoder Features (64)

#############################################################
# LOAD BASICS
#############################################################

grid_basics = np.load('grid_basics.npz').item()

train_dirs = grid_basics['train_dirs']
train_word_numbers = grid_basics['train_word_numbers']
train_word_idx = grid_basics['train_word_idx']

val_dirs = grid_basics['val_dirs']
val_word_numbers = grid_basics['val_word_numbers']
val_word_idx = grid_basics['val_word_idx']

si_dirs = grid_basics['si_dirs']; si_dirs = si_dirs[:12000]
si_word_numbers = grid_basics['si_word_numbers']; si_word_numbers = si_word_numbers[:12000]
si_word_idx = grid_basics['si_word_idx']; si_word_idx = si_word_idx[:12000]

#############################################################
# LOAD ATTRIBUTES
#############################################################

train_grid_attributes = np.load('train_grid_attributes_matrix.npy')
val_grid_attributes = np.load('val_grid_attributes_matrix.npy')
si_grid_attributes = np.load('si_grid_attributes_matrix.npy'); si_grid_attributes = si_grid_attributes[:12000]

# Normalization
train_grid_attributes_norm = (train_grid_attributes - train_grid_attributes.min(0)) / train_grid_attributes.ptp(0)
val_grid_attributes_norm = (val_grid_attributes - val_grid_attributes.min(0)) / val_grid_attributes.ptp(0)
si_grid_attributes_norm = (si_grid_attributes - si_grid_attributes.min(0)) / si_grid_attributes.ptp(0); si_grid_attributes_norm[:, 1] = 1.

#############################################################
# LOAD CORRECT_OR_NOT
#############################################################

lipreader_preds_wordidx_and_correctorwrong = np.load('lipreader_preds_wordidx_and_correctorwrong.npy').item()

train_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_word_idx']
val_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_word_idx']
si_lipreader_preds_word_idx = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_word_idx']; si_lipreader_preds_word_idx = si_lipreader_preds_word_idx[:12000]

train_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['train_lipreader_preds_correct_or_wrong']
val_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['val_lipreader_preds_correct_or_wrong']
si_lipreader_preds_correct_or_wrong = lipreader_preds_wordidx_and_correctorwrong['si_lipreader_preds_correct_or_wrong']; si_lipreader_preds_correct_or_wrong = si_lipreader_preds_correct_or_wrong[:12000]

np.sum(train_lipreader_preds_correct_or_wrong)/len(train_lipreader_preds_correct_or_wrong)
np.sum(val_lipreader_preds_correct_or_wrong)/len(val_lipreader_preds_correct_or_wrong)
np.sum(si_lipreader_preds_correct_or_wrong)/len(si_lipreader_preds_correct_or_wrong)

#############################################################
# LOAD LIPREADER PREDS
#############################################################

lipreader_preds = np.load('lipreader_preds.npz')

train_lipreader_preds = lipreader_preds['train_lipreader_preds']
val_lipreader_preds = lipreader_preds['val_lipreader_preds']
si_lipreader_preds = lipreader_preds['si_lipreader_preds']; si_lipreader_preds = si_lipreader_preds[:12000]

#############################################################
# LIPREADER ROC
#############################################################

# Compute ROC
lipreader_train_fpr, lipreader_train_tpr, lipreader_train_roc_auc, \
        lipreader_val_fpr, lipreader_val_tpr, lipreader_val_roc_auc, \
        lipreader_si_fpr, lipreader_si_tpr, lipreader_si_roc_auc = \
    compute_ROC_grid_multiclass(train_word_idx, train_lipreader_preds,
        val_word_idx, val_lipreader_preds,
        si_word_idx, si_lipreader_preds,
        savePlot=True, showPlot=True,
        plot_title='Baseline ROC curve of lipreader')

np.savez('ROC_baseline_lipreader', lipreader_train_fpr=lipreader_train_fpr, lipreader_train_tpr=lipreader_train_tpr, lipreader_train_roc_auc=lipreader_train_roc_auc,
    lipreader_val_fpr=lipreader_val_fpr, lipreader_val_tpr=lipreader_val_tpr, lipreader_val_roc_auc=lipreader_val_roc_auc,
    lipreader_si_fpr=lipreader_si_fpr, lipreader_si_tpr=lipreader_si_tpr, lipreader_si_roc_auc=lipreader_si_roc_auc)

#############################################################
# LOAD CRITIC PREDS
#############################################################

critic_preds = np.load('critic_preds.npz')

train_critic_preds = critic_preds['train_critic_preds']
val_critic_preds = critic_preds['val_critic_preds']
si_critic_preds = critic_preds['si_critic_preds']; si_critic_preds = si_critic_preds[:12000]

#############################################################
# CRITIC ROC
#############################################################

# Compute ROC
train_critic_fpr, train_critic_tpr, train_critic_roc_auc, \
        val_critic_fpr, val_critic_tpr, val_critic_roc_auc, \
        si_critic_fpr, si_critic_tpr, si_critic_roc_auc = \
    compute_ROC_grid_singleclass(train_lipreader_preds_correct_or_wrong, train_critic_preds,
        val_lipreader_preds_correct_or_wrong, val_critic_preds,
        si_lipreader_preds_correct_or_wrong, si_critic_preds,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of C3DCritic')

np.savez('ROC_critic', train_critic_fpr=train_critic_fpr, train_critic_tpr=train_critic_tpr, train_critic_roc_auc=train_critic_roc_auc, \
    val_critic_fpr=val_critic_fpr, val_critic_tpr=val_critic_tpr, val_critic_roc_auc=val_critic_roc_auc, \
    si_critic_fpr=si_critic_fpr, si_critic_tpr=si_critic_tpr, si_critic_roc_auc=si_critic_roc_auc)

#############################################################
# TRAIN SVM
#############################################################

train_matrix = train_grid_attributes_norm[:, 3:]
val_matrix = val_grid_attributes_norm[:, 3:]
si_matrix = si_grid_attributes_norm[:, 3:]

# LINEAR UNOPT

clf = SVC(kernel='linear', class_weight='balanced', probability=True)
clf.fit(train_matrix, train_lipreader_preds_correct_or_wrong)

# Acc
clf.score(train_matrix, train_lipreader_preds_correct_or_wrong)
clf.score(val_matrix, val_lipreader_preds_correct_or_wrong)
clf.score(si_matrix, si_lipreader_preds_correct_or_wrong)

# Scores
train_linear_unopt_svm_score = clf.decision_function(train_matrix)
val_linear_unopt_svm_score = clf.decision_function(val_matrix)
si_linear_unopt_svm_score = clf.decision_function(si_matrix)

# Compute ROC
linearSVM_unopt_train_fpr, linearSVM_unopt_train_tpr, linearSVM_unopt_train_roc_auc, \
        linearSVM_unopt_val_fpr, linearSVM_unopt_val_tpr, linearSVM_unopt_val_roc_auc, \
        linearSVM_unopt_si_fpr, linearSVM_unopt_si_tpr, linearSVM_unopt_si_roc_auc = \
    compute_ROC_grid_singleclass(train_lipreader_preds_correct_or_wrong, train_linear_unopt_svm_score,
        val_lipreader_preds_correct_or_wrong, val_linear_unopt_svm_score,
        si_lipreader_preds_correct_or_wrong, si_linear_unopt_svm_score,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of linear SVM (unoptimized)')

np.savez('ROC_linearSVM_unopt', linearSVM_unopt_train_fpr=linearSVM_unopt_train_fpr, linearSVM_unopt_train_tpr=linearSVM_unopt_train_tpr, linearSVM_unopt_train_roc_auc=linearSVM_unopt_train_roc_auc, \
    linearSVM_unopt_val_fpr=linearSVM_unopt_val_fpr, linearSVM_unopt_val_tpr=linearSVM_unopt_val_tpr, linearSVM_unopt_val_roc_auc=linearSVM_unopt_val_roc_auc, \
    linearSVM_unopt_si_fpr=linearSVM_unopt_si_fpr, linearSVM_unopt_si_tpr=linearSVM_unopt_si_tpr, linearSVM_unopt_si_roc_auc=linearSVM_unopt_si_roc_auc)

# LINEAR OPT

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=train_matrix, y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
def svm_linear_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# # perform tuning on linear
# hps_linear, _, _ = optunity.maximize(svm_linear_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
# hps_linear = {'logC': 1.14892578125, 'logGamma': -4.9794921875}
hps_linear = {'logC': -1.07275390625, 'logGamma': -0.8486328125}

# train model on the full training set with tuned hyperparameters
optimal_SVM_linear = SVC(kernel='linear', C=10 ** hps_linear['logC'], gamma=10 ** hps_linear['logGamma'], class_weight='balanced', probability=True).fit(train_matrix, train_lipreader_preds_correct_or_wrong)

# Acc
optimal_SVM_linear.score(train_matrix, train_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(val_matrix, val_lipreader_preds_correct_or_wrong)
optimal_SVM_linear.score(si_matrix, si_lipreader_preds_correct_or_wrong)

# Scores
train_linearSVM_opt_score = optimal_SVM_linear.decision_function(train_matrix)
val_linearSVM_opt_score = optimal_SVM_linear.decision_function(val_matrix)
si_linearSVM_opt_score = optimal_SVM_linear.decision_function(si_matrix)

# Compute ROC
train_linearSVM_opt_fpr, train_linearSVM_opt_tpr, train_linearSVM_opt_roc_auc, \
        val_linearSVM_opt_fpr, val_linearSVM_opt_tpr, val_linearSVM_opt_roc_auc, \
        si_linearSVM_opt_fpr, si_linearSVM_opt_tpr, si_linearSVM_opt_roc_auc = \
    compute_ROC_grid_singleclass(train_lipreader_preds_correct_or_wrong, train_linearSVM_opt_score,
        val_lipreader_preds_correct_or_wrong, val_linearSVM_opt_score,
        si_lipreader_preds_correct_or_wrong, si_linearSVM_opt_score,
        savePlot=True, showPlot=True,
        plot_title='ROC curve of linear SVM (optimized)')

np.savez('ROC_linearSVM_opt', train_linearSVM_opt_fpr=train_linearSVM_opt_fpr, train_linearSVM_opt_tpr=train_linearSVM_opt_tpr, train_linearSVM_opt_roc_auc=train_linearSVM_opt_roc_auc, \
    val_linearSVM_opt_fpr=val_linearSVM_opt_fpr, val_linearSVM_opt_tpr=val_linearSVM_opt_tpr, val_linearSVM_opt_roc_auc=val_linearSVM_opt_roc_auc, \
    si_linearSVM_opt_fpr=si_linearSVM_opt_fpr, si_linearSVM_opt_tpr=si_linearSVM_opt_tpr, si_linearSVM_opt_roc_auc=si_linearSVM_opt_roc_auc)


# RBF OPT

# @optunity.cross_validated(x=train_matrix, y=train_lipreader_preds_correct_or_wrong, num_folds=2, num_iter=1)
# def svm_rbf_auc(x_train, y_train, x_test, y_test, logC, logGamma):
#     model = SVC(kernel='rbf', C=10 ** logC, gamma=10 ** logGamma, class_weight='balanced').fit(x_train, y_train)
#     decision_values = model.decision_function(x_test)
#     return optunity.metrics.roc_auc(y_test, decision_values)

# # perform tuning on rbf
# hps_rbf, _, _ = optunity.maximize(svm_rbf_auc, num_evals=10, logC=[-5, 2], logGamma=[-5, 1])
hps_rbf = {'logC': 0.62255859375, 'logGamma': 0.1357421875}

# train model on the full training set with tuned hyperparameters
optimal_SVM_rbf = SVC(kernel='rbf', C=10 ** hps_rbf['logC'], gamma=10 ** hps_rbf['logGamma'], class_weight='balanced', probability=True).fit(train_matrix, train_lipreader_preds_correct_or_wrong)

# Acc
optimal_SVM_rbf.score(train_matrix, train_lipreader_preds_correct_or_wrong)
optimal_SVM_rbf.score(val_matrix, val_lipreader_preds_correct_or_wrong)
optimal_SVM_rbf.score(si_matrix, si_lipreader_preds_correct_or_wrong)
# >>> # Acc
# ... optimal_SVM_rbf.score(train_matrix, train_lipreader_preds_correct_or_wrong)
# 0.5853943939152173
# >>> optimal_SVM_rbf.score(val_matrix, val_lipreader_preds_correct_or_wrong)
# 0.71793037672866
# >>> optimal_SVM_rbf.score(si_matrix, si_lipreader_preds_correct_or_wrong)
# 0.37216666666666665

# Scores
train_rbfSVM_opt_score = optimal_SVM_rbf.decision_function(train_matrix)
val_rbfSVM_opt_score = optimal_SVM_rbf.decision_function(val_matrix)
si_rbfSVM_opt_score = optimal_SVM_rbf.decision_function(si_matrix)

# Compute ROC
train_rbfSVM_opt_fpr, train_rbfSVM_opt_tpr, train_rbfSVM_opt_roc_auc, \
        val_rbfSVM_opt_fpr, val_rbfSVM_opt_tpr, val_rbfSVM_opt_roc_auc, \
        si_rbfSVM_opt_fpr, si_rbfSVM_opt_tpr, si_rbfSVM_opt_roc_auc = \
    compute_ROC_grid_singleclass(train_lipreader_preds_correct_or_wrong, train_rbfSVM_opt_score,
        val_lipreader_preds_correct_or_wrong, val_rbfSVM_opt_score,
        si_lipreader_preds_correct_or_wrong, si_rbfSVM_opt_score,
        savePlot=False, showPlot=False,
        plot_title='ROC curve of RBF SVM optimized')

np.savez('ROC_rbfSVM_opt', train_rbfSVM_opt_fpr=train_rbfSVM_opt_fpr, train_rbfSVM_opt_tpr=train_rbfSVM_opt_tpr, train_rbfSVM_opt_roc_auc=train_rbfSVM_opt_roc_auc, \
    val_rbfSVM_opt_fpr=val_rbfSVM_opt_fpr, val_rbfSVM_opt_tpr=val_rbfSVM_opt_tpr, val_rbfSVM_opt_roc_auc=val_rbfSVM_opt_roc_auc, \
    si_rbfSVM_opt_fpr=si_rbfSVM_opt_fpr, si_rbfSVM_opt_tpr=si_rbfSVM_opt_tpr, si_rbfSVM_opt_roc_auc=si_rbfSVM_opt_roc_auc)

ROC_rbfSVM_opt = np.load('ROC_rbfSVM_opt.npz')

train_rbfSVM_opt_fpr, train_rbfSVM_opt_tpr, train_rbfSVM_opt_roc_auc, \
        val_rbfSVM_opt_fpr, val_rbfSVM_opt_tpr, val_rbfSVM_opt_roc_auc, \
        si_rbfSVM_opt_fpr, si_rbfSVM_opt_tpr, si_rbfSVM_opt_roc_auc = \
    ROC_rbfSVM_opt['train_rbfSVM_opt_fpr'], ROC_rbfSVM_opt['train_rbfSVM_opt_tpr'], ROC_rbfSVM_opt['train_rbfSVM_opt_roc_auc'].item(), \
        ROC_rbfSVM_opt['val_rbfSVM_opt_fpr'], ROC_rbfSVM_opt['val_rbfSVM_opt_tpr'], ROC_rbfSVM_opt['val_rbfSVM_opt_roc_auc'].item(), \
        ROC_rbfSVM_opt['si_rbfSVM_opt_fpr'], ROC_rbfSVM_opt['si_rbfSVM_opt_tpr'], ROC_rbfSVM_opt['si_rbfSVM_opt_roc_auc'].item()

plt.plot(train_rbfSVM_opt_fpr, train_rbfSVM_opt_tpr, label='train; AUC={0:0.4f}'.format(train_rbfSVM_opt_roc_auc))
plt.plot(val_rbfSVM_opt_fpr, val_rbfSVM_opt_tpr, label='val; AUC={0:0.4f}'.format(val_rbfSVM_opt_roc_auc))
plt.plot(si_rbfSVM_opt_fpr, si_rbfSVM_opt_tpr, label='si; AUC={0:0.4f}'.format(si_rbfSVM_opt_roc_auc))
plt.legend(loc='lower right')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve of RBF SVM optimized')

#############################################################
# LIPREADER SELF-TRAIN 10%
#############################################################






