import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

def calculate_score(model:BaseEstimator, 
                    scorer,
                    X:pd.DataFrame, 
                    y:pd.Series, 
                    kf:KFold) -> np.ndarray:
    y_pred_all, y_test_all = [], []

    for train_index, test_index in kf.split(X):
        model_ = clone(model)
        
        # Split the data into training and testing sets
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.to_numpy()[train_index], y.to_numpy()[test_index]
    
        model_.fit(X_train, y_train)
        
        y_pred         = model_.predict(X_test)

        y_pred_all += y_pred.tolist()
        y_test_all += y_test.tolist()

    score         = scorer(y_test_all, y_pred_all)

    return np.array(score)