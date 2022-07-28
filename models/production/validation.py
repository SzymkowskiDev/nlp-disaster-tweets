import datetime
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


def generate_perf_report(
    X: Iterable, y: Iterable,  # training vectors and target values (class labels)
    name: str,
    description: str,
    clf: Any = None,  # classification instance, by default `sklearn.svm.SVC()`
    test_size: float = .15,
    date_fmt="%Y-%m-%d %H:%M:%S",
) -> pd.Series:
    """
    Generate a report of certain model performance metrics, returned as an instance
    of `pandas.Series`.

    See example 1. in the README for usage details.
    """
    date = datetime.datetime.now().strftime(date_fmt)

    clf = clf or svm.SVC()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = clf.fit(X_train, y_train)
    predictions = model.predict(X_test)  # type: ignore

    # Compute and return performance metrics in a human-readable Series format.
    return pd.Series({
        "Date": date,
        "Name": name,
        "Description": description,
        "Test Size": test_size,
        "Precision": metrics.precision_score(y_test, predictions),
        "Recall": metrics.recall_score(y_test, predictions),
        # F-measure
        "F1 Score": metrics.f1_score(y_test, predictions),
        "Accuracy": metrics.accuracy_score(y_test, predictions),
        # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
        "Roc_auc_score": metrics.roc_auc_score(y_test, predictions)
    }, name="Performance Report")
