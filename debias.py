""""
Implementation of http://arxiv.org/abs/1607.06520
"""

import random
import numpy as np
import cvxpy as cvx

from functools import wraps
import itertools as it
import time

from word2vec import load_word2vec_model


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
def gender_subspace(model, word_groups, k=10):
    """
    the definition of C on page 12 makes no sense.... that isn't a matrix!
    """
    W = model.syn0
    C = np.zeros_like(W)
    mu = np.zeros(len(word_groups)+1)

    indexes = np.ones(W.shape[0], dtype=np.bool)
    for i, words in enumerate(word_groups):
        idx = [model.vocab[w].index for w in words]
        mu = np.mean(W[idx])
        D = len(words)
        C[idx] = (W[idx] - mu) / D
        indexes[i] = False

    # get the rest of the words not described in a word group
    mu = np.mean(W[indexes])
    D = sum(indexes)
    C[indexes] = (W[indexes] - mu) / D

    _, _, svdC = np.linalg.svd(C, full_matrices=True)
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

    prob.solve(solver=cvx.SCS, verbose=True)

    return X.value, prob.value


if __name__ == "__main__":
    gendered_words = [{w.strip().split(',')[0]
                      for w in it.chain(open("gendered_words_classifier.txt"),
                                        open("gendered_words.txt"))}]
    model = load_word2vec_model(truncate_vector=150)
    B = gender_subspace(model, gendered_words)
    neutral_words = random.sample(model.vocab.keys(), 5)
    X, result = soft_bias_correction(model, B, neutral_words)
