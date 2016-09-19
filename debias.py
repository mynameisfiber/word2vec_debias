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
    mu = np.zeros(len(word_groups)+1)
    Wnorm = np.zeros_like(W)

    indexes = np.ones(W.shape[0], dtype=np.bool)
    for i, words in enumerate(word_groups):
        idx = [model.vocab[w].index for w in words]
        mu = np.mean(W[idx])
        D = len(words)
        Wnorm[idx] = (W[idx] - mu) / D
        indexes[i] = False
    # get the rest of the words not described in a word group
    mu = np.mean(W[indexes])
    D = sum(indexes)
    Wnorm[indexes] = (W[indexes] - mu) / D

    C = np.cov(Wnorm, rowvar=False)
    _, s, Vt = np.linalg.svd(C, full_matrices=False)
    B = np.diag(np.sqrt(s[:k])) @ Vt[:k, :]
    return B


@timer
def soft_bias_correction(model, gender_subspace, neutral_words, tuning=0.2):
    neutral_indexes = [model.vocab[w].index for w in neutral_words]
    N = model.syn0.T[:, neutral_indexes]
    # slice the svd to ignore the unitary matricies
    U, E = np.linalg.svd(model.syn0.T, full_matrices=False)[:-1]
    E = np.diag(E)
    I = np.eye(E.shape[0])
    UE = U @ E
    EUT = E @ U.T

    return _solve_soft_bias_correction_tensorflow(UE, EUT, I, N, B, tuning)


def _solve_soft_bias_correction_cvxpy(UE, EUT, I, N, B, tuning):
    X = cvx.Semidef(N.shape[0], name='T')
    objective = (
        cvx.Minimize(cvx.sum_squares(EUT * (X - I) * UE) +
                     cvx.quad_over_lin(N.T * X * B.T, tuning))
    )
    constraints = [X >> 0]
    prob = cvx.Problem(objective, constraints)

    print("Large constants:")
    for c in prob.constants():
        try:
            print("\tConstant of size:", c.value.shape, type(c.value))
        except AttributeError:
            pass

    prob.solve(solver=cvx.SCS, verbose=True, gpu=True)
    return X.value, prob.value


def _solve_soft_bias_correction_scipy(UE, EUT, I, N, B, tuning):
    def objective(x, X, UE, EUT, I, N, B, tuning):
        X[np.triu_indices(N.shape[0], -1)] = 0
        X[np.triu_indices(N.shape[0])] = x
        X = X + X.T - np.diag(X.diagonal())
        return (np.linalg.norm(EUT @ (X - I) @ UE)**2 +
                tuning * np.linalg.norm(N.T @ X @ B.T)**2)

    n_elements = N.shape[0] * (N.shape[0] + 1)/2
    x = np.random.random(n_elements)
    X = np.zeros((N.shape[0],)*2)

    constraints = ({'type': 'ineq', 'fun': lambda x: x})
    result = optimize.minimize(objective, x, args=(X, UE, EUT, I, N, B, tuning),
                               constraints=constraints,
                               options={'disp': True, 'maxiter': 1000000},
                               method='COBYLA')

    x = result.x
    X[np.triu_indices(N.shape[0], -1)] = 0
    X[np.triu_indices(N.shape[0])] = x
    X = X + X.T - np.diag(X.diagonal())
    return X, objective(result.x)


def _solve_soft_bias_correction_tensorflow(UE, EUT, I, N, B, tuning):
    import tensorflow as tf

    B = B.astype(np.float32)

    n = N.shape[0]
    n_elements = int(n * (n + 1) / 2.0)
    xv = tf.Variable(np.random.rand(n_elements).astype(np.float32))
    x = tf.SparseTensor(indices=list(zip(*np.triu_indices(n, m=n))),
                        values=xv,
                        shape=(n, n))
    X = tf.sparse_tensor_to_dense(x)
    X += tf.transpose(X)
    X -= tf.diag(tf.diag_part(X))
    A = tf.matmul(EUT, tf.matmul((X - I), UE))
    B = tf.matmul(N.T, tf.matmul(X, B.T))
    loss = (tf.reduce_sum(tf.mul(A, A))**2 +
            tuning * tf.reduce_sum(tf.mul(B, B))**2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for step in range(8001):
        sess.run(train)
        if step % 100 == 0:
            print(step, sess.run(loss), sess.run(x))
    import IPython; IPython.embed()


if __name__ == "__main__":
    num_neutral = 714
    gendered_words = {w.strip().split(',')[0]
                      for w in it.chain(open("gendered_words_classifier.txt"),
                                        open("gendered_words.txt"))}
    model = load_word2vec_model(truncate_vector=None)
    model.syn0 = model.syn0[:, :100]
    gendered_words = list(filter(model.vocab.__contains__, gendered_words))
    neutral_words = list(set(model.vocab) - set(gendered_words))
    model = trim_model(model, neutral_words[num_neutral:])
    neutral_words = neutral_words[:num_neutral]

    B = gender_subspace(model, [gendered_words], k=4)
    X, result = soft_bias_correction(model, B, neutral_words)
