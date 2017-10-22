# GET GRIDCORPUS ATTRIBUTES

import cv2
import dlib
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tqdm

from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

from grid_attributes_params import *


def plot_grid_ROC(train_fpr, train_tpr, train_roc_auc,
        val_fpr, val_tpr, val_roc_auc,
        si_fpr, si_tpr, si_roc_auc,
        train_OP_fpr=None, train_OP_tpr=None,
        val_OP_fpr=None, val_OP_tpr=None,
        si_OP_fpr=None, si_OP_tpr=None,
        train_optimalOP_fpr=None, train_optimalOP_tpr=None,
        val_optimalOP_fpr=None, val_optimalOP_tpr=None,
        si_optimalOP_fpr=None, si_optimalOP_tpr=None,
        plot_title='ROC curve of RBF SVM optimized'):
    plt.plot(train_fpr, train_tpr, color='C0', label='train; AUC={0:0.4f}'.format(train_roc_auc))
    plt.plot(val_fpr, val_tpr, color='C1', label='val; AUC={0:0.4f}'.format(val_roc_auc))
    plt.plot(si_fpr, si_tpr, color='C2', label='si; AUC={0:0.4f}'.format(si_roc_auc))
    if train_OP_fpr is not None and train_OP_tpr is not None:
        plt.plot(train_OP_fpr, train_OP_tpr, color='C0', marker='x')
    if val_OP_fpr is not None and val_OP_tpr is not None:
        plt.plot(val_OP_fpr, val_OP_tpr, color='C1', marker='x')
    if si_OP_fpr is not None and si_OP_tpr is not None:
        plt.plot(si_OP_fpr, si_OP_tpr, color='C2', marker='x')
    if train_optimalOP_fpr is not None and train_optimalOP_tpr is not None:
        plt.plot(train_optimalOP_fpr, train_optimalOP_tpr, color='C0', marker='o')
    if val_optimalOP_fpr is not None and val_optimalOP_tpr is not None:
        plt.plot(val_optimalOP_fpr, val_optimalOP_tpr, color='C1', marker='o')
    if si_optimalOP_fpr is not None and si_optimalOP_tpr is not None:
        plt.plot(si_optimalOP_fpr, si_optimalOP_tpr, color='C2', marker='o')
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(plot_title)
    plt.show()

def plot_pc_grid_ROC(train_l_fpr, train_l_tpr, train_l_roc_auc,
        train_unl_fpr, train_unl_tpr, train_unl_roc_auc,
        val_fpr, val_tpr, val_roc_auc,
        si_fpr, si_tpr, si_roc_auc,
        train_l_OP_fpr=None, train_l_OP_tpr=None,
        train_unl_OP_fpr=None, train_unl_OP_tpr=None,
        val_OP_fpr=None, val_OP_tpr=None,
        si_OP_fpr=None, si_OP_tpr=None,
        train_l_optimalOP_fpr=None, train_l_optimalOP_tpr=None,
        train_unl_optimalOP_fpr=None, train_unl_optimalOP_tpr=None,
        val_optimalOP_fpr=None, val_optimalOP_tpr=None,
        si_optimalOP_fpr=None, si_optimalOP_tpr=None,
        plot_title='ROC curve of RBF SVM optimized'):
    plt.plot(train_l_fpr, train_l_tpr, color='C0', label='train; AUC={0:0.4f}'.format(train_l_roc_auc))
    plt.plot(train_unl_fpr, train_unl_tpr, color='C1', label='train; AUC={0:0.4f}'.format(train_unl_roc_auc))
    plt.plot(val_fpr, val_tpr, color='C2', label='val; AUC={0:0.4f}'.format(val_roc_auc))
    plt.plot(si_fpr, si_tpr, color='C3', label='si; AUC={0:0.4f}'.format(si_roc_auc))
    if train_l_OP_fpr is not None and train_l_OP_tpr is not None:
        plt.plot(train_l_OP_fpr, train_l_OP_tpr, color='C0', marker='x')
    if train_unl_OP_fpr is not None and train_unl_OP_tpr is not None:
        plt.plot(train_unl_OP_fpr, train_unl_OP_tpr, color='C1', marker='x')
    if val_OP_fpr is not None and val_OP_tpr is not None:
        plt.plot(val_OP_fpr, val_OP_tpr, color='C2', marker='x')
    if si_OP_fpr is not None and si_OP_tpr is not None:
        plt.plot(si_OP_fpr, si_OP_tpr, color='C3', marker='x')
    if train_l_optimalOP_fpr is not None and train_l_optimalOP_tpr is not None:
        plt.plot(train_l_optimalOP_fpr, train_l_optimalOP_tpr, color='C0', marker='o')
    if train_unl_optimalOP_fpr is not None and train_unl_optimalOP_tpr is not None:
        plt.plot(train_unl_optimalOP_fpr, train_unl_optimalOP_tpr, color='C1', marker='o')
    if val_optimalOP_fpr is not None and val_optimalOP_tpr is not None:
        plt.plot(val_optimalOP_fpr, val_optimalOP_tpr, color='C2', marker='o')
    if si_optimalOP_fpr is not None and si_optimalOP_tpr is not None:
        plt.plot(si_optimalOP_fpr, si_optimalOP_tpr, color='C3', marker='o')
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(plot_title)
    plt.show()


def compute_ROC_grid_singleclass(train_correct_or_not, train_probabilities,
                                 val_correct_or_not, val_probabilities,
                                 si_correct_or_not, si_probabilities,
                                 train_fpr_op=None, train_tpr_op=None,
                                 val_fpr_op=None, val_tpr_op=None,
                                 si_fpr_op=None, si_tpr_op=None,
                                 savePlot=False, showPlot=False,
                                 plot_title='ROC curve of linear SVM unoptimized'):
    train_fpr, train_tpr, train_thresholds, train_roc_auc = compute_ROC_singleclass(train_correct_or_not, train_probabilities)
    val_fpr, val_tpr, val_thresholds, val_roc_auc = compute_ROC_singleclass(val_correct_or_not, val_probabilities)
    si_fpr, si_tpr, si_thresholds, si_roc_auc = compute_ROC_singleclass(si_correct_or_not, si_probabilities)
    if showPlot or savePlot:
        plt.plot(train_fpr, train_tpr, color='C0', label='train; AUC={0:0.4f}'.format(train_roc_auc))
        plt.plot(val_fpr, val_tpr, color='C1', label='val; AUC={0:0.4f}'.format(val_roc_auc))
        plt.plot(si_fpr, si_tpr, color='C2', label='si; AUC={0:0.4f}'.format(si_roc_auc))
        if train_fpr_op is not None and train_tpr_op is not None:
            plt.plot(train_fpr_op, train_tpr_op, color='C0', marker='x')
        if val_fpr_op is not None and val_tpr_op is not None:
            plt.plot(val_fpr_op, val_tpr_op, color='C1', marker='x')
        if si_fpr_op is not None and si_tpr_op is not None:
            plt.plot(si_fpr_op, si_tpr_op, color='C2', marker='x')
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(plot_title)
    if savePlot:
        plt.savefig('a.png')
    if showPlot:
        plt.show()
    if showPlot or savePlot:
        plt.close()
    return train_fpr, train_tpr, train_thresholds, train_roc_auc, val_fpr, val_tpr, val_thresholds, val_roc_auc, si_fpr, si_tpr, si_thresholds, si_roc_auc


def compute_pc_ROC_grid_singleclass(train_l_correct_or_not, train_l_probabilities,
                                    train_unl_correct_or_not, train_unl_probabilities,
                                    val_correct_or_not, val_probabilities,
                                    si_correct_or_not, si_probabilities,
                                    train_l_fpr_op=None, train_l_tpr_op=None,
                                    train_unl_fpr_op=None, train_unl_tpr_op=None,
                                    val_fpr_op=None, val_tpr_op=None,
                                    si_fpr_op=None, si_tpr_op=None,
                                    savePlot=False, showPlot=False,
                                    plot_title='ROC curve of linear SVM unoptimized'):
    train_l_fpr, train_l_tpr, train_l_thresholds, train_l_roc_auc = compute_ROC_singleclass(train_l_correct_or_not, train_l_probabilities)
    train_unl_fpr, train_unl_tpr, train_unl_thresholds, train_unl_roc_auc = compute_ROC_singleclass(train_unl_correct_or_not, train_unl_probabilities)
    val_fpr, val_tpr, val_thresholds, val_roc_auc = compute_ROC_singleclass(val_correct_or_not, val_probabilities)
    si_fpr, si_tpr, si_thresholds, si_roc_auc = compute_ROC_singleclass(si_correct_or_not, si_probabilities)
    if showPlot or savePlot:
        plt.plot(train_l_fpr, train_l_tpr, color='C0', label='train - labelled; AUC={0:0.4f}'.format(train_l_roc_auc))
        plt.plot(train_unl_fpr, train_unl_tpr, color='C1', label='train - unlabelled; AUC={0:0.4f}'.format(train_unl_roc_auc))
        plt.plot(val_fpr, val_tpr, color='C2', label='val; AUC={0:0.4f}'.format(val_roc_auc))
        plt.plot(si_fpr, si_tpr, color='C3', label='si; AUC={0:0.4f}'.format(si_roc_auc))
        if train_l_fpr_op is not None and train_l_tpr_op is not None:
            plt.plot(train_l_fpr_op, train_l_tpr_op, color='C0', marker='x')
        if train_unl_fpr_op is not None and train_unl_tpr_op is not None:
            plt.plot(train_unl_fpr_op, train_unl_tpr_op, color='C1', marker='x')
        if val_fpr_op is not None and val_tpr_op is not None:
            plt.plot(val_fpr_op, val_tpr_op, color='C2', marker='x')
        if si_fpr_op is not None and si_tpr_op is not None:
            plt.plot(si_fpr_op, si_tpr_op, color='C3', marker='x')
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(plot_title)
    if savePlot:
        plt.savefig('a.png')
    if showPlot:
        plt.show()
    if showPlot or savePlot:
        plt.close()
    return train_l_fpr, train_l_tpr, train_l_thresholds, train_l_roc_auc, train_unl_fpr, train_unl_tpr, train_unl_thresholds, train_unl_roc_auc, val_fpr, val_tpr, val_thresholds, val_roc_auc, si_fpr, si_tpr, si_thresholds, si_roc_auc


def compute_ROC_singleclass(correct_or_not, probability):
    # probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    # mean_fpr = np.linspace(0, 1, 100)
    fpr, tpr, thresholds = roc_curve(correct_or_not, probability)
    # tpr = interp(mean_fpr, fpr, tpr)
    tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def compute_pc_ROC_grid_multiclass(train_l_word_idx, train_l_confidences,
                train_unl_word_idx, train_unl_confidences,
                val_word_idx, val_confidences,
                si_word_idx, si_confidences,
                savePlot=False, showPlot=False,
                plot_title='ROC curve of linear SVM unoptimized'):
    train_l_fpr, train_l_tpr, train_l_roc_auc = compute_ROC_multiclass(label_binarize(train_l_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), train_l_confidences, len(GRID_VOCAB_FULL))
    train_unl_fpr, train_unl_tpr, train_unl_roc_auc = compute_ROC_multiclass(label_binarize(train_unl_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), train_unl_confidences, len(GRID_VOCAB_FULL))
    val_fpr, val_tpr, val_roc_auc = compute_ROC_multiclass(label_binarize(val_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), val_confidences, len(GRID_VOCAB_FULL))
    si_fpr, si_tpr, si_roc_auc = compute_ROC_multiclass(label_binarize(si_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), si_confidences, len(GRID_VOCAB_FULL))
    if showPlot or savePlot:
        plt.plot(train_l_fpr['micro'], train_l_tpr['micro'], color='C0', linestyle='-', label='train_labelled_micro; AUC={0:0.4f}'.format(train_l_roc_auc['micro']))
        plt.plot(train_l_fpr['macro'], train_l_tpr['macro'], color='C0', linestyle='--', label='train_labelled_macro; AUC={0:0.4f}'.format(train_l_roc_auc['macro']))
        plt.plot(train_unl_fpr['micro'], train_unl_tpr['micro'], color='C1', linestyle='-', label='train_unlabelled_micro; AUC={0:0.4f}'.format(train_unl_roc_auc['micro']))
        plt.plot(train_unl_fpr['macro'], train_unl_tpr['macro'], color='C1', linestyle='--', label='train_unlabelled_macro; AUC={0:0.4f}'.format(train_unl_roc_auc['macro']))
        plt.plot(val_fpr['micro'], val_tpr['micro'], color='C2', linestyle='-', label='val_micro; AUC={0:0.4f}'.format(val_roc_auc['micro']))
        plt.plot(val_fpr['macro'], val_tpr['macro'], color='C2', linestyle='--', label='val_macro; AUC={0:0.4f}'.format(val_roc_auc['macro']))
        plt.plot(si_fpr['micro'], si_tpr['micro'], color='C3', linestyle='-', label='si_micro; AUC={0:0.4f}'.format(si_roc_auc['micro']))
        plt.plot(si_fpr['macro'], si_tpr['macro'], color='C3', linestyle='--', label='si_macro; AUC={0:0.4f}'.format(si_roc_auc['macro']))
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(plot_title)
    if savePlot:
        plt.savefig('a.png')
    if showPlot:
        plt.show()
    plt.close()
    return train_l_fpr, train_l_tpr, train_l_roc_auc, train_unl_fpr, train_unl_tpr, train_unl_roc_auc, val_fpr, val_tpr, val_roc_auc, si_fpr, si_tpr, si_roc_auc


def compute_ROC_grid_multiclass(train_word_idx, train_confidences,
                val_word_idx, val_confidences,
                si_word_idx, si_confidences,
                savePlot=False, showPlot=False,
                plot_title='ROC curve of linear SVM unoptimized'):
    train_fpr, train_tpr, train_roc_auc = compute_ROC_multiclass(label_binarize(train_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), train_confidences, len(GRID_VOCAB_FULL))
    val_fpr, val_tpr, val_roc_auc = compute_ROC_multiclass(label_binarize(val_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), val_confidences, len(GRID_VOCAB_FULL))
    si_fpr, si_tpr, si_roc_auc = compute_ROC_multiclass(label_binarize(si_word_idx, classes=np.arange(len(GRID_VOCAB_FULL))), si_confidences, len(GRID_VOCAB_FULL))
    if showPlot or savePlot:
        plt.plot(train_fpr['micro'], train_tpr['micro'], color='C0', linestyle='-', label='train_micro; AUC={0:0.4f}'.format(train_roc_auc['micro']))
        plt.plot(train_fpr['macro'], train_tpr['macro'], color='C0', linestyle='--', label='train_macro; AUC={0:0.4f}'.format(train_roc_auc['macro']))
        plt.plot(val_fpr['micro'], val_tpr['micro'], color='C1', linestyle='-', label='val_micro; AUC={0:0.4f}'.format(val_roc_auc['micro']))
        plt.plot(val_fpr['macro'], val_tpr['macro'], color='C1', linestyle='--', label='val_macro; AUC={0:0.4f}'.format(val_roc_auc['macro']))
        plt.plot(si_fpr['micro'], si_tpr['micro'], color='C2', linestyle='-', label='si_micro; AUC={0:0.4f}'.format(si_roc_auc['micro']))
        plt.plot(si_fpr['macro'], si_tpr['macro'], color='C2', linestyle='--', label='si_macro; AUC={0:0.4f}'.format(si_roc_auc['macro']))
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(plot_title)
    if savePlot:
        plt.savefig('a.png')
    if showPlot:
        plt.show()
    plt.close()
    return train_fpr, train_tpr, train_roc_auc, val_fpr, val_tpr, val_roc_auc, si_fpr, si_tpr, si_roc_auc


def compute_ROC_multiclass(y_test, y_score, n_classes):
    # y_test == nxv one-hot
    # y_score == nxv full softmax scores
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # MICRO
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # MACRO
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it
    mean_tpr /= n_classes
    # compute AUC
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    return fpr, tpr, roc_auc


def calc_grid_operating_points(clf, train_y, val_y, si_y, train_matrix, val_matrix, si_matrix):
    # Train
    train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_y, clf.predict(train_matrix)).ravel()
    train_fpr_op = train_fp/(train_fp + train_tn)
    train_tpr_op = train_tp/(train_tp + train_fn)
    # Val
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_y, clf.predict(val_matrix)).ravel()
    val_fpr_op = val_fp/(val_fp + val_tn)
    val_tpr_op = val_tp/(val_tp + val_fn)
    # Si
    si_tn, si_fp, si_fn, si_tp = confusion_matrix(si_y, clf.predict(si_matrix)).ravel()
    si_fpr_op = si_fp/(si_fp + si_tn)
    si_tpr_op = si_tp/(si_tp + si_fn)
    # Return
    return train_fpr_op, train_tpr_op, val_fpr_op, val_tpr_op, si_fpr_op, si_tpr_op


def calc_pc_grid_operating_points(clf, train_labelled_y, train_unlabelled_y, val_y, si_y, train_l_matrix, train_unl_matrix, val_matrix, si_matrix):
    # Train_labelled
    train_l_tn, train_l_fp, train_l_fn, train_l_tp = confusion_matrix(train_labelled_y, clf.predict(train_l_matrix)).ravel()
    train_l_fpr_op = train_l_fp/(train_l_fp + train_l_tn)
    train_l_tpr_op = train_l_tp/(train_l_tp + train_l_fn)
    # Train_unlabelled
    train_unl_tn, train_unl_fp, train_unl_fn, train_unl_tp = confusion_matrix(train_unlabelled_y, clf.predict(train_unl_matrix)).ravel()
    train_unl_fpr_op = train_unl_fp/(train_unl_fp + train_unl_tn)
    train_unl_tpr_op = train_unl_tp/(train_unl_tp + train_unl_fn)
    # Val
    val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_y, clf.predict(val_matrix)).ravel()
    val_fpr_op = val_fp/(val_fp + val_tn)
    val_tpr_op = val_tp/(val_tp + val_fn)
    # Si
    si_tn, si_fp, si_fn, si_tp = confusion_matrix(si_y, clf.predict(si_matrix)).ravel()
    si_fpr_op = si_fp/(si_fp + si_tn)
    si_tpr_op = si_tp/(si_tp + si_fn)
    # Return
    return train_l_fpr_op, train_l_tpr_op, train_unl_fpr_op, train_unl_tpr_op, val_fpr_op, val_tpr_op, si_fpr_op, si_tpr_op


def find_ROC_optimalOP(fpr, tpr, thresholds, score, y):
    optimalOP_threshold = thresholds[np.argmin((1 - tpr)**2 + fpr**2)]
    optimalOP_fpr, optimalOP_tpr, optimalOP_acc = find_fpr_tpr_acc_from_thresh(y, score, optimalOP_threshold)
    return optimalOP_threshold, optimalOP_fpr, optimalOP_tpr, optimalOP_acc


def find_fpr_tpr_acc_from_thresh(y, score, optimalOP_threshold):
    tn, fp, fn, tp = confusion_matrix(y, score > optimalOP_threshold).ravel()
    optimalOP_fpr = fp/(fp + tn)
    optimalOP_tpr = tp/(tp + fn)
    optimalOP_acc = (tp + tn)/len(y)
    return optimalOP_fpr, optimalOP_tpr, optimalOP_acc


def make_LSTMlipreader_predictions(lipreader_preds,
                                   lipreader_pred_word_idx,
                                   lipreader_preds_correct_or_wrong,
                                   # word_durations,
                                   dirs,
                                   word_numbers,
                                   word_idx,
                                   lipreader,
                                   grid_vocab=GRID_VOCAB_FULL,
                                   startNum=0):
    # dirs = train_val_dirs
    # word_numbers = train_val_word_numbers
    # word_idx = train_val_word_idx
    # detector, predictor = load_detector_and_predictor()
    # For each data point
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
        if i < startNum:
            continue
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
        wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                 1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # # Word duration
        # word_durations[i] = wordEndFrame - wordStartFrame + 1
        # All mouth file names of video
        mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[0][-f - 1] = robust_imread(wordMouthFrame, 0)
        # SAVE ENCODER FEATURE
        # break
        # MAKE PREDICTION
        lipreader_preds[i] = lipreader.predict(wordImages)
        lipreader_pred_word_idx[i] = np.argmax(lipreader_preds[i])
        lipreader_preds_correct_or_wrong[i] = lipreader_pred_word_idx[i] == wordIndex


def make_critic_predictions(critic_preds,
                            lipreader_preds_word_idx,
                            # word_durations,
                            dirs,
                            word_numbers,
                            word_idx,
                            critic,
                            grid_vocab=GRID_VOCAB_FULL,
                            startNum=0):
    # dirs = train_val_dirs
    # word_numbers = train_val_word_numbers
    # word_idx = train_val_word_idx
    # detector, predictor = load_detector_and_predictor()
    # For each data point
    full_preds_word_idx = label_binarize(lipreader_preds_word_idx, classes=np.arange(len(grid_vocab)))
    for i, (vidDir, wordNum, wordIndex) in tqdm.tqdm(enumerate(zip(dirs, word_numbers, word_idx)), total=len(dirs)):
        if i < startNum:
            continue
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
        wordEndFrame = math.floor(int(wordTimeData[wordNum].split(' ')[
                                 1]) / maxClipDuration * FRAMES_PER_VIDEO)
        # # Word duration
        # word_durations[i] = wordEndFrame - wordStartFrame + 1
        # All mouth file names of video
        mouthFiles = sorted(glob.glob(os.path.join(vidDir, '*Mouth*.jpg')))
        # Note the file names of the word
        wordMouthFiles = mouthFiles[wordStartFrame:wordEndFrame + 1]
        # Initialize the array of images for this word
        wordImages = np.zeros((1, FRAMES_PER_WORD, NUM_OF_MOUTH_PIXELS))
        # For each frame of this word
        for f, wordMouthFrame in enumerate(wordMouthFiles[:FRAMES_PER_WORD]):
            # in reverse order of frames. eg. If there are 7 frames:
            # 0 0 0 0 0 0 0 7 6 5 4 3 2 1
            wordImages[0][-f - 1] = robust_imread(wordMouthFrame, 0)
        # SAVE ENCODER FEATURE
        critic_preds[i] = critic.predict([wordImages, np.reshape(full_preds_word_idx[i], (1, len(grid_vocab)))])


def robust_imread(wordMouthFrame, cv_option=0):
    try:
        image = np.reshape(cv2.imread(wordMouthFrame, cv_option) / 255., (NUM_OF_MOUTH_PIXELS,))
        return image
    except TypeError:
        return robust_imread(wordMouthFrame, cv_option)


def load_detector_and_predictor(verbose=False):
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        if verbose:
            print("Detector and Predictor loaded. (load_detector_and_predictor)")
        return detector, predictor
    # If error in SHAPE_PREDICTOR_PATH
    except RuntimeError:
        raise ValueError("\n\nERROR: Wrong Shape Predictor .dat file path - " + \
            SHAPE_PREDICTOR_PATH, "(load_detector_and_predictor)\n\n")


def read_head_poses(mode='train_val', num=10):
    # HEAD POSES
    head_poses = np.zeros((num, 3))
    lines_gen = gen_txt_files_line_by_line(mode=mode, word="Head pose")
    for idx in range(num):
        line = next(lines_gen)
        head_poses[idx, 0] = float(line.rstrip().split()[-3][1:-1])
        head_poses[idx, 1] = float(line.rstrip().split()[-2][:-1])
        head_poses[idx, 2] = float(line.rstrip().split()[-1][:-1])
    # Return
    lines_gen.close()
    return head_poses


def read_txt_files_line_range(mode='train_val', start_idx=0, stop_idx=3, word='Estimating head pose'):
    lines = []
    lines_gen = gen_txt_files_line_by_line(mode=mode, word=word)
    for idx in range(stop_idx):
        line = next(lines_gen)
        if idx >= start_idx:
            lines.append(line)
    lines_gen.close()
    return lines


def gen_txt_files_line_by_line(mode='train_val', word='Estimating head pose'):
    if mode == 'train_val':
        txt_files = TRAIN_VAL_HEAD_POSE_TXT_FILES
    elif mode == 'si':
        txt_files = SI_HEAD_POSE_TXT_FILES
    while 1:
        for file in txt_files:
            with open(file, 'r') as f:
                for line in f:
                    if word in line:
                        yield line

#############################################################
# LOAD IMAGE DIRS AND WORD NUMBERS
#############################################################


def load_image_dirs_and_word_numbers(trainValSpeakersList = [1, 2, 3, 4, 5, 6, 7, 9],
                                        valSplit = 0.1,
                                        siList = [10, 11]):
    # TRAIN AND VAL
    trainDirs = []
    trainWordNumbers = []
    valDirs = []
    valWordNumbers = []
    np.random.seed(29)
    # For each speaker
    for speaker in sorted(tqdm.tqdm(trainValSpeakersList)):
        speakerDir = os.path.join(GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
        # List of all videos for each speaker
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        totalNumOfImages = len(vidDirs)
        # To shuffle directories before splitting into train and validate
        fullListIdx = list(range(totalNumOfImages))
        np.random.shuffle(fullListIdx)
        # Append training directories
        for i in fullListIdx[:int((1 - valSplit) * totalNumOfImages)]:
            for j in range(WORDS_PER_VIDEO):
                trainDirs.append(vidDirs[i])
                trainWordNumbers.append(j)
        # Append val directories
        for i in fullListIdx[int((1 - valSplit) * totalNumOfImages):]:
            for j in range(WORDS_PER_VIDEO):
                valDirs.append(vidDirs[i])
                valWordNumbers.append(j)
    # Numbers
    print("No. of training words: " + str(len(trainDirs)))
    print("No. of val words: " + str(len(valDirs)))
    # SPEAKER INDEPENDENT
    siDirs = []
    siWordNumbers = []
    for speaker in sorted(tqdm.tqdm(siList)):
        speakerDir = os.path.join(GRID_DATA_DIR, 's' + '{0:02d}'.format(speaker))
        vidDirs = sorted(glob.glob(os.path.join(speakerDir, '*/')))
        for i in fullListIdx:
                for j in range(WORDS_PER_VIDEO):
                    siDirs.append(vidDirs[i])
                    siWordNumbers.append(j)
    # Numbers
    print("No. of speaker-independent words: " + str(len(siDirs)))
    # Return
    return np.array(trainDirs), np.array(trainWordNumbers), np.array(valDirs), np.array(valWordNumbers), np.array(siDirs), np.array(siWordNumbers)

