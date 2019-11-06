import numpy as np

from sklearn.model_selection import GridSearchCV

def accuracy_score(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0], ""

    return sum(y_true == y_predict) / len(y_true)

def grid_search():
    param_grid = [
        {
            'weights':['uniform'],
            'n_neighbors':[i for i in range[1, 11]]
        },
        {
            'weights': ['distance'],
            'n_neighbors': [i for i in range[1, 11]],
            'p': [i for i in range[1, 6]]
        }
    ]

    knn_clf = KNeighborClassifier()
    grid_search = GridSearchCV(knn_clf, param_grid)