# Feature Selection

This repository contains code to perform wrapper-based feature selection (i.e. forward and backward). 

## Install requirements:
```
pip install -r requirements.txt
```

## How to use?
Load model selection class from the package
```
from src.featureSelector import FeatureModelSelection
```


```
FeatureModelSelection(model=model,
                      scorer=r2_score,
                      X_train=X_train,
                      y_train=y_train,
                      higher_good=False,
                      )

model:   sklearn BaseEstimator(Machine Learning Models)(Eg: KNeighborsRegressor())
scorer:  Scorer like r2 square, % error or Custom Scorer
X_train: np.ndarray (Training input data (X,d), 
y_train: np.ndarray (Target Labels)
higher_good: Boolean (True when the performance is better for higher values and False when the performance is better for lower value)
```
Find the best features using the forward or backward feature selection method using ```feature_selection_type=('forward' or 'backward')``` 
```
best_features = model_selection.find_best_features(feature_selection_type='forward')
```

Save each step of the feature selection method in an Excel file.
```
model_selection.feature_selection_tabularize(best_features, './feature_selected.xlsx')
```

Find the example at  ```main.py ```

