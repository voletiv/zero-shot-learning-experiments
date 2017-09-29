
def load_lrw_vocab_list(LRW_VOCAB_LIST_FILE):
    lrw_vocab = []
    with open(LRW_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().lower()
            # LABOUR is not in word2vec, LABOR is
            if word == 'labour':
                word = 'labor'
            lrw_vocab.append(word)
    return lrw_vocab

def load_gridcorpus_vocab_list(GRID_VOCAB_LIST_FILE):
    grid_vocab = []
    with open(GRID_VOCAB_LIST_FILE) as f:
        for line in f:
            word = line.rstrip().split()[-1]
            grid_vocab.append(word)
    return grid_vocab

