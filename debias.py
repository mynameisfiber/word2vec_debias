""""
Implementation of http://arxiv.org/abs/1607.06520
"""

import random
import numpy as np
import cvxpy as cvx
from scipy import optimize

from functools import wraps
import itertools as it
import time

from word2vec import load_word2vec_model, trim_model


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
    - the definition of C on page 12 makes no sense.... that isn't a matrix!
    - is there only a group for gendered words or also for neutral ones?
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
    B = gender_subspace

    return _solve_soft_bias_correction_scipy(UE, I, N, B, tuning)


def _solve_soft_bias_correction_cvxpy(UE, I, N, B, tuning):
    X = cvx.Semidef(N.shape[0], name='T')
    objective = (
        cvx.Minimize(cvx.sum_squares(UE.T * (X - I) * UE) +
                     cvx.quad_over_lin(N.T * X * B.T), tuning)
    )
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


def _solve_soft_bias_correction_scipy(UE, I, N, B, tuning):
    def objective(x, X, UE, I, N, B, tuning):
        X[np.triu_indices(N.shape[0], -1)] = 0
        X[np.triu_indices(N.shape[0])] = x
        X = X + X.T - np.diag(X.diagonal())
        return (np.linalg.norm(UE.T @ (X - I) @ UE)**2 +
                tuning * np.linalg.norm(N.T @ X @ B.T)**2)

    n_elements = N.shape[0] * (N.shape[0] + 1)/2
    x = np.random.random(n_elements)
    X = np.zeros((N.shape[0],)*2)

    constraints = ({'type': 'ineq', 'fun': lambda x: x})
    result = optimize.minimize(objective, x, args=(X, UE, I, N, B, tuning),
                               constraints=constraints,
                               options={'disp': True, 'maxiter': 1000000},
                               method='COBYLA')

    x = result.x
    X[np.triu_indices(N.shape[0], -1)] = 0
    X[np.triu_indices(N.shape[0])] = x
    X = X + X.T - np.diag(X.diagonal())
    return X, objective(result.x)


if __name__ == "__main__":
    gendered_words = {w.strip().split(',')[0]
                      for w in it.chain(open("gendered_words_classifier.txt"),
                                        open("gendered_words.txt"))}
    model = load_word2vec_model(truncate_vector=50)
    gendered_words = list(filter(model.vocab.__contains__, gendered_words))
    neutral_words = set(model.vocab) - set(gendered_words)
    model = trim_model(model, neutral_words[1000:])
    neutral_words = neutral_words[:1000]

    B = gender_subspace(model, [gendered_words], k=4)
    X, result = soft_bias_correction(model, B, neutral_words)
