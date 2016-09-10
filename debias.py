""""
Implementation of http://arxiv.org/abs/1607.06520
"""

import random
from gensim.models import Word2Vec
import numpy as np
import cvxpy as cvx
from tqdm import tqdm
from joblib import Memory

import re
from functools import wraps
import time
import heapq

memory = Memory(cachedir='./cache/', verbose=1)


def timer(fxn):
    @wraps(fxn)
    def _(*args, **kwargs):
        print("Starting {}".format(fxn.__name__))
        start = time.time()
        result = fxn(*args, **kwargs)
        runtime = time.time() - start
        print("{}: took {:0.3f}s".format(fxn.__name__, runtime))
        return result
    return _


@timer
def gender_subspace(model, k=10):
    W = model.syn0
    normalization = (1.0 / np.linalg.norm(W, axis=1)).sum()
    mu = W * normalization
    C = (W - mu).T @ (W - mu) * normalization
    _, _, svdC = np.linalg.svd(C)
    return svdC[:k]


@timer
def soft_bias_correction(model, gender_subspace, neutral_words, tuning=0.2):
    neutral_indexes = [model.vocab[w].index for w in neutral_words]
    N = model.syn0.T[:, neutral_indexes]
    # slice the svd to ignore the unitary matricies
    U, E = np.linalg.svd(model.syn0.T, full_matrices=False)[:-1]
    E = np.diag(E)
    I = np.eye(E.shape[0])
    UE = U @ E

    X = cvx.Semidef(N.shape[0], name='T')
    objective = sum((
        cvx.Minimize(cvx.sum_squares(UE.T * (X - I) * UE)),
        cvx.Minimize(tuning * cvx.sum_squares(N.T * X * B.T))
    ))
    constraints = [X >> 0]
    prob = cvx.Problem(objective, constraints)

    print("Large constants:")
    for c in prob.constants():
        try:
            print("\tConstant of size:", c.value.shape, type(c.value))
        except AttributeError:
            pass

    prob.solve(solver=cvx.CVXOPT, verbose=True)

    return X.value, prob.value


@timer
def vocab_subsample(model, top_k=5000, max_length=20):
    nlargest = []
    acceptable = re.compile("^[a-z ]{," + str(max_length) + "}$")
    for word, value in tqdm(model.vocab.items()):
        if acceptable.match(word) is not None:
            item = (value.count, (word, value))
            if len(nlargest) >= top_k:
                heapq.heappushpop(nlargest, item)
            else:
                heapq.heappush(nlargest, item)
    indexes = []
    model.vocab = {}
    for i, (_, (word, value)) in enumerate(nlargest):
        indexes.append(value.index)
        value.index = i
        model.vocab[word] = value
    model.syn0 = model.syn0[indexes]
    return model


@timer
@memory.cache
def load_word2vec_model():
    model = timer(Word2Vec.load_word2vec_format)(
        './data/word2vec_googlenews_negative300.bin',
        binary=True, limit=None
    )
    model.syn0 = model.syn0[:, :100]
    model.syn0 /= np.linalg.norm(model.syn0)  # ensure normalied
    model = vocab_subsample(model, top_k=50000)
    return model


if __name__ == "__main__":
    model = load_word2vec_model()
    B = gender_subspace(model)
    neutral_words = random.sample(model.vocab.keys(), 5)
    X, result = soft_bias_correction(model, B, neutral_words)
