import numpy as np
from math import sqrt
from collections import Counter
from .metrics import accuracy_score

class KNNClassifier:

    def __init__(self, k):
        assert k >= 1, ""
        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], ""
        assert self.k <= X_train.shape[0], ""
        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_pridect):
        assert self._X_train is not None and self._y_train is not None, ""
        assert X_pridect.shape[1] == self._X_train.shape[1], ""
        y_predict = [self._predict(x) for x in X_pridect]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "kNN(k=%d)" % self.k