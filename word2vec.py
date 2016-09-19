from gensim.models import Word2Vec
from joblib import Memory
import numpy as np

import re
import heapq
from tqdm import tqdm

memory = Memory(cachedir='./cache/', verbose=1)


@memory.cache
def vocab_subsample(model, top_k, max_length=20):
    nlargest = []
    for word, value in tqdm(model.vocab.items()):
        item = (value.count, (word, value))
        if len(nlargest) >= top_k:
            heapq.heappushpop(nlargest, item)
        else:
            heapq.heappush(nlargest, item)
    indexes = []
    model.vocab = {}
    model.index2word = []
    acceptable = re.compile("^[a-z ]{," + str(max_length) + "}$")
    i = 0
    for _, (word, value) in nlargest:
        if acceptable.match(word) is not None:
            indexes.append(value.index)
            value.index = i
            model.vocab[word] = value
            model.index2word.append(word)
            i += 1
    model.syn0 = model.syn0[indexes]
    return model


def trim_model(model, words):
    words = set(words)
    mask = np.ones(model.syn0.shape[0], dtype=np.bool)
    indexes = []
    for w in words:
        i = model.vocab[w].index
        mask[i] = False
        indexes.append(i)
    model.syn0 = model.syn0[mask]
    model.index2word = [w for w in model.index2word if w not in words]
    model.vocab = {k: v for k, v in model.vocab.items() if k not in words}
    for i, w in enumerate(model.index2word):
        model.vocab[w].index = i
    return model


@memory.cache
def load_word2vec_model(truncate_vector=None, top_k=50000, limit=None):
    model = Word2Vec.load_word2vec_format(
        './data/word2vec_googlenews_negative300.bin',
        binary=True, limit=limit
    )
    model.syn0 = model.syn0[:, :truncate_vector]
    model.syn0 /= np.linalg.norm(model.syn0, axis=1)[:, np.newaxis]
    model = vocab_subsample(model, top_k=top_k)
    print("Final vocab size:", len(model.vocab))
    return model

