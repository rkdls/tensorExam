from sklearn import clone
import matplotlib

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original", data_home='./mnist_dataset/')

X, y = mnist['data'], mnist['target']

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

print(y[36000])  # 라벨값 확인.

# 셔플해보자.
import numpy as np

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)  # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)
print('y_train_5', y_train_5)

# 훈련시킴
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

# cross_val_score 확인.
print('cross_val_score(sgd_clf,X_train, y_train_5, cv=3, scoring="accuracy"):',
      cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5 = Never5Classifier()
print('cross_val_score(sgd_clf,X_train, y_train_5, cv=3, scoring="accuracy"):',
      cross_val_score(never_5, X_train, y_train_5, cv=3, scoring="accuracy"))

from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=5)

print('y_train_pred', type(y_train_pred))
# confusion matrix 확인해보자. (TP,TN, FP,FN)
print(confusion_matrix(y_train_5, y_train_pred))
skfolds = StratifiedKFold(n_splits=3, random_state=42)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
print('y_scores', y_scores)
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

print(precisions)
print(recalls)
# print(thresholds)
from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))  # == 3719 / (3719 + 984)
print(recall_score(y_train_5, y_train_pred))  # == 3719 / (3719+ 1702)

from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    print('len', len(thresholds), len(precisions), len(recalls))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    # plt.plot(recalls[:-1], precisions[:-1], "g-", label="Precision,Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="center left")
    plt.ylim([0, 1])


# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2, label="roc curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)

from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probs_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
y_scores_forest = y_probs_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)


# plot_roc_curve(fpr, tpr)
plt.show()
