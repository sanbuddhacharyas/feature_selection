import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, clone

from typing import Tuple

from src.utils import calculate_score

class FeatureModelSelection():
    def __init__(self, 
                model:BaseEstimator, 
                scorer,
                X_train:pd.DataFrame, 
                y_train:pd.Series,
                higher_good:bool,
                n_splits=5):          
        
        self.model  = model                            # Feature selection model
        self.scorer = scorer                           # Performance Metrics
        self.higher_good = higher_good                 # If higher value is better or not
        self.X_train, self.y_train = X_train, y_train  # Training dataset for feature selection
        self.all_feature_scores = []                   # Selected features

        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Save the trained model
    def save(self, path:str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.model, path) 

    def find_score(self, kf:KFold, features:list) -> np.ndarray:
        return np.array(calculate_score(self.model, self.scorer, self.X_train[features], self.y_train, kf))
    
    def fit(self, features:list) -> None:
        self.model.fit(self.X_train[features], self.y_train)

    def forward_feature_selection(self):
        model                   = clone(self.model)
        all_features            = self.X_train.columns.values

        self.all_feature_scores = []
        flag                    = False

        self.selected_features  = []
        best_score              = [0, 0] if self.higher_good else 100.0
        
        while len(self.selected_features) != len(all_features):
            one_line_score    = []
            one_line_features = []
            
            for feature in all_features:
                
                if feature not in self.selected_features:
                    testing_feature = self.selected_features + [feature]
                    
                    score = self.scorer(model, self.X_train[testing_feature], self.y_train, self.kf)
                                    
                    one_line_score.append(score)
                    one_line_features.append(feature)
           
            one_line_score = np.array(one_line_score) if self.higher_good else one_line_score
            
            if self.higher_good:
                best_socre_ind      = np.argmax(one_line_score[:,0])
                one_line_best_score = one_line_score[best_socre_ind]

            else:
                best_socre_ind, one_line_best_score = np.argmin(one_line_score), np.min(one_line_score)

            sel_one_line_feature    = one_line_features[best_socre_ind] 

            temp = {}
            for key, score in zip(one_line_features, one_line_score):
                key = self.selected_features + [key]
                temp[str(key)] = score
                
            if self.higher_good:
                if one_line_best_score[0] > best_score[0]:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True
                        
            else:
                if one_line_best_score <= best_score:
                    best_score = one_line_best_score
                    self.selected_features.append(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False
                else: flag = True

            if flag: break
        
        self.best_score = best_score
        return self.all_feature_scores
    
    def backward_feature_selection(self):
        model = clone(self.model)

        all_features            = self.X_train.columns.values
        self.selected_features  = all_features.copy().tolist()
        self.all_feature_scores = []

        best_score              = self.scorer(model, self.X_train[all_features], self.y_train, self.kf)
        flag                    = False

        while len(self.selected_features) != 0:
            one_line_score    = []
            one_line_features = []
            
            for feature in all_features:
                if feature in self.selected_features: 
                    testing_feature = [i for i in self.selected_features if i != feature] # Remove the feature from the set
                    
                    score = self.scorer(model, self.X_train[testing_feature], self.y_train, self.kf)
                  
                    one_line_score.append(score)
                    one_line_features.append(feature)
           
            if self.higher_good:
                best_socre_ind      = np.argmax(one_line_score[:,0])
                one_line_best_score = one_line_score[best_socre_ind]

            else:
                best_socre_ind, one_line_best_score = np.argmin(one_line_score), np.min(one_line_score)

            sel_one_line_feature    = one_line_features[best_socre_ind] 

            temp = {}
            for key, score in zip(one_line_features, one_line_score):
                temp[str(key)] = score

            if self.higher_good:
                if one_line_best_score[0] > best_score[0]:
                    best_score = one_line_best_score
                    self.selected_features.remove(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True
                        
            else:
                if one_line_best_score <= best_score:
                    best_score = one_line_best_score
                    self.selected_features.remove(sel_one_line_feature)
                    self.all_feature_scores.append(temp)
                    flag = False

                else: flag = True

            if flag: break
        
        self.best_score = best_score
        if self.all_feature_scores == []: self.all_feature_scores.append({str(self.selected_features): best_score[0]})
        return self.all_feature_scores

    def find_best_features(self,  feature_selection_type:str) -> list: # Forward or Backward feature selection
        return  self.forward_feature_selection() if feature_selection_type=='forward' else self.backward_feature_selection()

    def find_testing_score(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Tuple[list, list]:

        # Fit the model with selected features
        self.model.fit(self.X_train[self.selected_features], self.y_train)

        # Return both training and testing r2 score
        return self.model.score(self.X_train[self.selected_features], self.y_train), \
               self.model.score(X_test[self.selected_features], y_test)