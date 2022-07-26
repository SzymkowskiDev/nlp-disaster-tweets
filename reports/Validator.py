import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import datetime
from typing import Any, Iterable


class Validator:
    """
    A class for validating various methods of preprocessing data on 
    the same model/models

    Needs documentation
    """
    
    def __init__(
        self,
        X: Iterable,
        y: Iterable,
        name: str,
        description: str,
        estimator: Any = svm.SVC,
    test_size: float = .15) -> None: 
        self._date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._X = X
        self._y = y
        self._name = name
        self._description = description
        self._test_size = test_size
        self._clf = estimator()
           
    def generate_raport(self) -> None:
        """
        Function to compute selected metric of the model and

        Needs better documentation
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self._X,
            self._y,
            test_size = self._test_size)

        model = self._clf.fit(X_train, y_train)
        predictions = model.predict(X_test)
        self._precision = metrics.precision_score(y_test, predictions)
        self._recall = metrics.recall_score(y_test, predictions)
        self._f1 = metrics.f1_score(y_test, predictions)
        self._accuracy = metrics.accuracy_score(y_test, predictions)
        self._roc_auc_score = metrics.roc_auc_score(y_test, predictions)
        self._confusion_matrix = metrics.confusion_matrix(y_test, predictions)
        print('Model test:')
        print(f'\t\tPrec: {self._precision} \
                \n\t\t Rec: {self._recall} \
                \n\t\t F1: {self._f1} \
                \n\t\t Acc: {self._accuracy} \
                \n\t\t ROC_AUC: {self._roc_auc_score}')
        print(f' {self._confusion_matrix}')
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Function to get models metrics in readable format

        Needs better documentation
        """
        tmp = {
            "Date": self._date,
            "Name": self._name, 
            "Description": self._description,
            "Test Size": self._test_size,
            "Precision": self._precision,
            "Recall": self._recall,
            "F1 Score": self._f1,
            "Accuracy": self._accuracy,
            "Roc_auc_score": self._roc_auc_score
        }
        return pd.DataFrame.from_dict(tmp, orient = 'index').T
