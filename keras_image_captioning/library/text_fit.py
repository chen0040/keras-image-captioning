from collections import Counter
import nltk


def fit_text(data, max_vocab_size):
    counter = Counter()
    max_seq_length = 0
    for t in data:
        _, txt = t
        txt = 'START ' + txt.lower() + ' END'
        words = nltk.word_tokenize(txt)
        seq_length = len(words)
        max_seq_length = max(seq_length, max_seq_length)
        for w in words:
            counter[w] += 1

    word2idx = dict()
    for idx, word in enumerate(counter.most_common(max_vocab_size)):
        word2idx[word] = idx

    config = dict()
    config['max_seq_length'] = max_seq_length
    config['word2idx'] = word2idx
    config['vocab_size'] = len(word2idx)
    config['idx2word'] = dict([(idx, word) for word, idx in word2idx.items()])

    return config
