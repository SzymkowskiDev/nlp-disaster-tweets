import datetime
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


def generate_perf_report(
    X: Iterable,
    y: Iterable,
    *,
    name: str = "sample",
    description: str = "no description",
    clf: Any = None,
    date_fmt: str = "%Y-%m-%d %H:%M:%S",
    test_size: float = 0.15,
) -> pd.Series:
    """
    Generate a report gathering main model classification metrics.

    Parameters
    ----------
    X : iterable of shape (n_samples, n_features)
        Sparse matrix of shape (n_samples, n_features).
    y : iterable
        Target values (class labels).
    name : str
        Name of the report, by default "sample".
    description : str
        Optional description for better understanding of the report.
    clf : Any
        Vector classification. Defaults to C-Support Vector Classification
        (`sklearn.svm.SVC`).
    test_size : float or int, default=0.15
        See `sklearn.model_selection.train_test_split` documentation
        for details on this parameter.
    date_fmt : str, default="%Y-%m-%d %H:%M:%S"
        Date format.

    Returns
    -------
    report : pd.Series
        Report with specified name, date, description, test size,
        precision score, recall score, f-measure, accuracy and ROC
        AUC score in human-friendly format.
    """  # needs review
    date = datetime.datetime.now().strftime(date_fmt)

    clf = clf or svm.SVC()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    model = clf.fit(X_train, y_train)
    predictions = model.predict(X_test)  # type: ignore

    return pd.Series(
        {
            "Date": date,
            "Description": description,
            "Test Size": test_size,
            "Precision": metrics.precision_score(y_test, predictions),
            "Recall": metrics.recall_score(y_test, predictions),
            # F-measure
            "F1 Score": metrics.f1_score(y_test, predictions),
            "Accuracy": metrics.accuracy_score(y_test, predictions),
            # Area Under the Receiver Operating Characteristic Curve (ROC AUC)
            "Roc_auc_score": metrics.roc_auc_score(y_test, predictions),
            "Confusion Matrix": metrics.confusion_matrix(y_test, predictions),
            "Roc curve": metrics.roc_curve(y_test, predictions),
        },
        name=name,
    )
