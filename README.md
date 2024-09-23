# Feature Selection

This repository contains code to perform wrapper-based feature selection (i.e forward and backward). 

## Install requirements:
```
pip install -r requirements.txt
```

```bash
FeatureModelSelection(model=model,
                      scorer=r2_score,
                      X_train=X_train,
                      y_train=y_train,
                      higher_good=False,
                      )

model: sklearn BaseEstimator
```

