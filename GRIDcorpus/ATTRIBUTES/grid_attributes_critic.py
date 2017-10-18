import optunity
import optunity.metrics

from sklearn.svm import SVC

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

train_grid_attributes = np.load('train_grid_attributes_matrix.npy').item()
val_grid_attributes = np.load('val_grid_attributes_matrix.npy').item()
si_grid_attributes = np.load('si_grid_attributes_matrix.npy').item()

#############################################################
# TRAIN SVM
#############################################################

clf = SVC(kernel='linear')

train_matrix = (train_grid_attributes - train_grid_attributes.min(0)) / train_grid_attributes.ptp(0)

# clf.fit(train_matrix, train_lipreader_preds_correct_or_wrong)

# score function: twice iterated 10-fold cross-validated accuracy
@optunity.cross_validated(x=train_matrix, y=train_lipreader_preds_correct_or_wrong, num_folds=10, num_iter=2)
def svm_auc(x_train, y_train, x_test, y_test, logC, logGamma):
    model = SVC(kernel='linear', C=10 ** logC, gamma=10 ** logGamma).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-5, 2], logGamma=[-5, 1])

# train model on the full training set with tuned hyperparameters
optimal_model = SVC(C=10 ** hps['logC'], gamma=10 ** hps['logGamma']).fit(data, labels)



clf.score(train_grid_attributes, train_lipreader_preds_correct_or_wrong)
clf.score(val_grid_attributes, val_lipreader_preds_correct_or_wrong)
clf.score(si_grid_attributes, si_lipreader_preds_correct_or_wrong)
