import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold


from typing import Tuple

def pad_values(feature_:list, scores:list, max_val:int) -> Tuple[list, list]:
    pad_value = max_val - len(feature_)
    feature_  = feature_ + ['']*pad_value
    scores    = scores   + [0]*pad_value

    return feature_, scores

class perError:
    def __init__(self, y_LOD) -> None:
        self.y_LOD = y_LOD
        
    def __call__(self, y_test:pd.Series, y_pred:np.array)->float:
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
    
        mask           = (y_test != 0)    # Non Zero Concentration
        zero_mask      = ~(mask)          # Zero Concentration

        y_pred         = np.maximum(y_pred, 0.0)

        # Only for non zero concentration
        non_zero_per_error = np.abs(y_test[mask] - y_pred[mask])/(0.5*(y_test[mask] + y_pred[mask]))
    
        # zero concentration
        zero_per_error     = np.abs(y_test[zero_mask] - y_pred[zero_mask]) / self.y_LOD

        assert not(np.isnan(zero_per_error).any())
        assert not(np.isnan(non_zero_per_error).any())

        per_error         = np.concatenate((non_zero_per_error, zero_per_error))
        per_error         = np.mean(per_error) * 100

        return per_error

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

    score         = scorer(y_test=y_test_all, y_pred=y_pred_all)

    return np.array(score)