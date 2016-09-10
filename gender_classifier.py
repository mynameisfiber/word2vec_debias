from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import f1_score, make_scorer
import numpy as np

from debias import load_word2vec_model


def gender_data_labels(model, gendered):
    labels = np.zeros(len(model.vocab))
    for word in gendered:
        idx = w2v.vocab[word].index
        labels[idx] = 1
    return labels


if __name__ == "__main__":
    gendered_words = [line.strip()
                      for line in open("gendered_words.txt")]
    w2v = load_word2vec_model()
    labels = gender_data_labels(w2v, gendered_words)

    svm = LinearSVC(C=1.0, class_weight="balanced")
    skf = StratifiedKFold(labels, 3)
    scorer = make_scorer(f1_score, average='weighted')
    score, permutation_scores, pvalue = permutation_test_score(
        estimator=svm,
        X=w2v.syn0,
        y=labels,
        scoring=scorer,
        cv=skf,
        n_permutations=10,
        n_jobs=1
    )
