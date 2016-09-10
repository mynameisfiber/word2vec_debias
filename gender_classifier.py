from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, permutation_test_score
from sklearn.metrics import f1_score, accuracy_score

from gensim.models import Word2Vec
import numpy as np

from word2vec import trim_model


def gender_data_labels(model, gendered):
    labels = np.zeros(len(model.vocab), dtype=np.bool)
    for word in gendered:
        idx = w2v.vocab[word].index
        labels[idx] = True
    return labels


def extract_words_matrix(model, words):
    indexes = [model.vocab[w].index for w in words]
    matrix = model.syn0[indexes]
    return matrix

if __name__ == "__main__":
    w2v = Word2Vec.load_word2vec_format(
        './data/word2vec_googlenews_negative300.bin',
        binary=True, limit=100000
    )
    w2v.syn0 /= np.linalg.norm(w2v.syn0, axis=1)[:, np.newaxis]
    used_words = [w.strip() for w in open("used_words.txt")]
    used_words_matrix = extract_words_matrix(w2v, used_words)
    gendered_words = [line.strip()
                      for line in open("gendered_words.txt")]
    w2v = trim_model(w2v, set(used_words) - set(gendered_words))

    print("Making labels")
    labels = gender_data_labels(w2v, gendered_words)

    print("Starting KFold")
    skf = StratifiedKFold(labels, 10, shuffle=True)
    svms = []
    scores = []
    for trainidx, testidx in skf:
        X_train = w2v.syn0[trainidx]
        X_test = w2v.syn0[testidx]
        y_train = labels[trainidx]
        y_test = labels[testidx]

        print("fitting")
        svm = LinearSVC(C=1.0, class_weight="balanced")
        svm.fit(X_train, y_train)
        svms.append(svm)
        predicted = svm.predict(X_test)
        scores.append((
            f1_score(y_test, predicted, average='binary'),
            accuracy_score(y_test, predicted)
        ))
        print("scores: ", scores[-1])

    gendered_dist = np.mean([s.decision_function(w2v.syn0[labels]) for s in svms], axis=0)
    threshold = gendered_dist.mean() - gendered_dist.std()
    print("Threshold: ", threshold)

    distances = np.mean([s.decision_function(used_words_matrix) for s in svms], axis=0)
    s = np.sign(threshold)
    gendered_indicies = np.where(s*distances > s*threshold)[0]
    with open("gendered_words_classifier.txt", "w+") as fd:
        for i in gendered_indicies:
            word = used_words[i]
            d = distances[i]
            fd.write("{},{}\n".format(word, d))
