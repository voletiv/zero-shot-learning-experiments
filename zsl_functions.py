import numpy as np

#############################################################
# LEARN V AND CALCULATE ACCURACIES
#############################################################


def learn_by_ESZSL_and_calc_accs(train_num_of_words, word_to_attr_matrix,
                                 features, one_hot_words,
                                 si_features=None, si_one_hot_words=None,
                                 optG=1e-6, optL=1e-3, fix_seed=True):

    ########################################
    # Split data into in_vocabulary (iv),
    # i.e those containing training_words,
    # and out_of_vocabulary (oov) data
    ########################################

    # Choose words for training
    training_words_idx = choose_words_for_training(
        train_num_of_words, vocab_size=one_hot_words.shape[1], fix_seed=fix_seed)

    # Split data into in_vocabulary (iv) and out_of_vocabulary (oov)
    iv_features, iv_one_hot_words, oov_features, oov_one_hot_words \
        = split_data_into_iv_and_oov(training_words_idx,
                                           features, one_hot_words)

    # Split SPEAKER-INDEPENDENT data into iv and oov
    if si_features is not None and si_one_hot_words is not None:
        si_iv_features, si_iv_one_hot_words, \
            si_oov_features, si_oov_one_hot_words \
            = split_data_into_iv_and_oov(training_words_idx,
                                               si_features, si_one_hot_words)
    else:
        si_iv_features = None
        si_iv_one_hot_words = None
        si_oov_features = None
        si_oov_one_hot_words = None

    ########################################
    # Split embedding matrix into iv and oov
    ########################################

    iv_word_to_attr_matrix, oov_word_to_attr_matrix \
        = split_embedding_matrix_into_iv_and_oov(
            training_words_idx, word_to_attr_matrix)

    ########################################
    # EMBARRASSINGLY SIMPLE LEARNING
    ########################################
    # predV = ((X.X^T + gI)^(-1)).X.Y.S^T.((S.S^T + lI)^(-1))
    # === dxa = (dxd) . dxm . mxz . zxa . (axa)
    pred_V = np.dot(np.dot(np.dot(np.dot(np.linalg.inv(np.dot(
        iv_features.T, iv_features)
        + optG * np.eye(iv_features.shape[1])),
        iv_features.T), iv_one_hot_words), iv_word_to_attr_matrix),
        np.linalg.inv(np.dot(iv_word_to_attr_matrix.T,
                             iv_word_to_attr_matrix)
                      + optL * np.eye(iv_word_to_attr_matrix.shape[1])))

    # Using my method
    my_pred_V = np.zeros((iv_features.shape[1], iv_word_to_attr_matrix.shape[1]))
    for i in range(iv_features.shape[1]):
        for j in range(iv_word_to_attr_matrix.shape[1]):
            my_pred_V[i, j] = np.dot(
                iv_features[:, i],
                np.dot(iv_one_hot_words,
                       iv_word_to_attr_matrix[:, j]))/((np.dot(iv_word_to_attr_matrix[:, j],
                                                        iv_word_to_attr_matrix[:, j]) + optL) \
                                                       * (np.dot(x[i], x[i]) + optG))

    ########################################
    # ACCURACY CALCULATION
    ########################################

    iv_acc, oov_acc, si_iv_acc, si_oov_acc, si_acc \
        = calc_accs(pred_V, iv_features, iv_one_hot_words, oov_features,
                    oov_one_hot_words, si_iv_features, si_iv_one_hot_words,
                    si_oov_features, si_oov_one_hot_words,
                    iv_word_to_attr_matrix, oov_word_to_attr_matrix, word_to_attr_matrix)


    my_iv_acc, my_oov_acc, my_si_iv_acc, my_si_oov_acc, si_acc \
        = calc_accs(my_pred_V, iv_features, iv_one_hot_words, oov_features,
                    oov_one_hot_words, si_iv_features, si_iv_one_hot_words,
                    si_oov_features, si_oov_one_hot_words,
                    iv_word_to_attr_matrix, oov_word_to_attr_matrix, word_to_attr_matrix)

    return pred_V, iv_acc, oov_acc, si_iv_acc, si_oov_acc, si_acc, \
        my_pred_V, my_iv_acc, my_oov_acc, my_si_iv_acc, my_si_oov_acc, si_acc

#############################################################
# ACCURACY CALCULATION
#############################################################


def calc_accs(pred_V, iv_features, iv_one_hot_words, oov_features,
              oov_one_hot_words, si_iv_features, si_iv_one_hot_words,
              si_oov_features, si_oov_one_hot_words,
              iv_word_to_attr_matrix, oov_word_to_attr_matrix, word_to_attr_matrix):

    # Train Acc
    y_iv_preds = np.argmax(
        np.dot(np.dot(iv_features, pred_V), iv_word_to_attr_matrix.T), axis=1)
    iv_acc = np.sum(y_iv_preds == np.argmax(
        iv_one_hot_words, axis=1)) / len(iv_one_hot_words)

    # Test Acc
    y_oov_preds = np.argmax(
        np.dot(np.dot(oov_features, pred_V), oov_word_to_attr_matrix.T), axis=1)
    oov_acc = np.sum(y_oov_preds == np.argmax(
        oov_one_hot_words, axis=1)) / len(oov_one_hot_words)

    if si_iv_features is not None:
        # SI in vocab Acc
        y_si_iv_preds = np.argmax(
            np.dot(np.dot(si_iv_features, pred_V), iv_word_to_attr_matrix.T), axis=1)
        si_iv_acc = np.sum(y_si_iv_preds == np.argmax(
            si_iv_one_hot_words, axis=1)) / len(si_iv_one_hot_words)

        # SI OOV Acc
        y_si_oov_preds = np.argmax(
            np.dot(np.dot(si_oov_features, pred_V), oov_word_to_attr_matrix.T), axis=1)
        si_oov_acc = np.sum(y_si_oov_preds == np.argmax(
            si_oov_one_hot_words, axis=1)) / len(si_oov_one_hot_words)

        # SI Acc
        # y_si_preds = np.append(y_si_iv_preds, y_si_oov_preds)
        y_si_preds = np.argmax(
            np.dot(np.dot(np.vstack((si_iv_features, si_oov_features)), pred_V), word_to_attr_matrix.T), axis=1)
        si_acc = np.sum(y_si_preds == np.append(np.argmax(
            si_iv_one_hot_words, axis=1), np.argmax(si_oov_one_hot_words, axis=1))) / len(y_si_preds)
    else:
        si_iv_acc = -1
        si_oov_acc = -1
        si_acc = -1

    return iv_acc, oov_acc, si_iv_acc, si_oov_acc, si_acc

#############################################################
# CHOOSE WORDS FOR TRAINING, REST FOR OOV TESTING
#############################################################


def choose_words_for_training(train_num_of_words, vocab_size, fix_seed=True):
    # Choose words to keep in training data - training words
    if fix_seed:
        np.random.seed(29)
    return np.sort(np.random.choice(
        vocab_size, train_num_of_words, replace=False))

#############################################################
# SPLIT DATA INTO IN_VOCAB AND OOV
#############################################################


def split_data_into_iv_and_oov(training_words_idx, features, one_hot_words):

    oov_words_idx = np.delete(
        np.arange(one_hot_words.shape[1]), training_words_idx)

    # Choose those rows in data that contain training words, i.e.
    # in_vocabulary (inv)
    in_vocab_data_idx = np.array([i for i in range(
        len(one_hot_words)) if np.argmax(one_hot_words[i]) in training_words_idx])

    # Make the rest of the rows as testing data
    oov_data_idx = np.delete(
        np.arange(len(one_hot_words)), in_vocab_data_idx)

    # IN_VOCAB
    iv_features = features[in_vocab_data_idx]
    iv_one_hot_words = one_hot_words[in_vocab_data_idx][:, training_words_idx]

    # (SPEAKER-DEPENDENT) TEST DATA
    oov_features = features[oov_data_idx]
    oov_one_hot_words = one_hot_words[oov_data_idx][:, oov_words_idx]

    return iv_features, iv_one_hot_words, oov_features, oov_one_hot_words


#############################################################
# SPLIT EMBEDDING MATRIX INTO IN_VOCAB AND OOV
#############################################################


def split_embedding_matrix_into_iv_and_oov(training_words_idx,
                                           word_to_attr_matrix):
    iv_word_to_attr_matrix = word_to_attr_matrix[training_words_idx]
    oov_word_to_attr_matrix = word_to_attr_matrix[np.delete(
        np.arange(len(word_to_attr_matrix)), training_words_idx)]
    return iv_word_to_attr_matrix, oov_word_to_attr_matrix
