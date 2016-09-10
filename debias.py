""""
Implementation of http://arxiv.org/abs/1607.06520
"""
import random
from gensim.models import Word2Vec
import numpy as np
import cvxpy as cvx
import cvxopt


def gender_subspace(model, k=10):
    W = model.syn0
    normalization = (1.0 / np.linalg.norm(W, axis=1)).sum()
    mu = W * normalization
    C = (W - mu).T @ (W - mu) * normalization
    _, _, svdC = np.linalg.svd(C)
    return svdC[:k]

def soft_bias_correction(model, gender_subspace, neutral_words, tuning=0.2):
    neutral_indexes = [model.vocab[w].index for w in neutral_words]
    W = model.syn0.T
    U, E, Vt = np.linalg.svd(W)
    E = E * np.eye(*W.shape)
    N = W[:, neutral_indexes]
    I = np.eye(W.shape[0])
    UE = U @ E

    print("E: ",E.shape)
    print("U: ",U.shape)
    print("W: ",W.shape)
    print("N: ",W.shape)
    print("B: ",B.shape)

    X = cvx.Semidef(W.shape[0], name='X')

    objective = sum((
        cvx.Minimize(cvx.sum_squares(UE.T * (X-I) * UE)),
        cvx.Minimize(tuning * cvx.sum_squares(N.T * X * B.T))
    ))
    constraints = [0 <= X]
    prob = cvx.Problem(objective, constraints)
    
    import IPython; IPython.embed()




if __name__ == "__main__":
    model = Word2Vec.load_word2vec_format('./data/word2vec_googlenews_negative300.bin',
                                          binary=True, limit=20)
    B = gender_subspace(model)
    neutral_words = random.sample(model.vocab.keys(), 5)
    soft_bias_correction(model, B, neutral_words)
