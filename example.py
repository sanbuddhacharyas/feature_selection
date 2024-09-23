import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.featureSelector import FeatureModelSelection
from src.load_dataset import load_dataset
from src.utils import perError

from sklearn.metrics import make_scorer


from sklearn.metrics import r2_score

if __name__ == '__main__':

    # Load Training Dataset
    (X_train, X_test, y_train, y_test), _ = load_dataset(dataset_path='/Users/sangam/Desktop/Epilepsey/Code/vgramreg/dataset/ML1_ML2',
                                                       normalization=True,
                                                       standardize_type='mean_std')
    
    model           = KNeighborsRegressor()
    model_selection = FeatureModelSelection(model=model,
                                            scorer=perError(y_LOD=0.91),
                                            X_train=X_train,
                                            y_train=y_train,
                                            higher_good=False,
                                            )
    
    best_features = model_selection.find_best_features(feature_selection_type='forward')
    model_selection.feature_selection_tabularize(best_features, './feature_selected.xlsx')