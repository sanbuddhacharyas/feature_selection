import os
import pandas as pd
from glob import glob

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from typing import Tuple

# from src.config import DATASET_PATH, OUTPUT_PATH

def select_normalizer(standardize_type):
    if standardize_type   == 'mean_std':scaler = StandardScaler() 
    elif standardize_type == 'min_max': scaler = MinMaxScaler()
    elif standardize_type == 'robust':  scaler = RobustScaler()

    return scaler

def load_dataset(dataset_path=None, normalization=True, normalize_blanks=False, standardize_type='', eval_correl_matrix=False, split=True, test_nor_separate=False, showFileName=False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # if dataset_path==None: dataset_path = DATASET_PATH

    if 'ML1_ML2'in os.path.basename(dataset_path):
        datasets = sorted([f"{i}/extracted_features.xlsx" for i in glob(f'{dataset_path}/*')])
    
        df  = [pd.read_excel(dataset) for dataset in datasets]
        df  = pd.concat(df)
    
    else: df = pd.read_excel(f'{dataset_path}/extracted_features.xlsx')
    
    X   = df[["peak area", "peak curvature", "peak V", "vcenter", "PH", "signal_mean", "signal_std", \
                                "dS_dV_max_peak", "dS_dV_min_peak", "dS_dV_peak_diff", "dS_dV_max_V", "dS_dV_min_V", "dS_dV_area"]]
    if showFileName: 
        y = df['file']
        stratified_y = df['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz','')))

    else: 
        y   = df['file'].apply(lambda x: int(x.split('_')[-2].replace('cbz','')))
        stratified_y = y

    # Copy the dataframe to remove the warning caused because of slicing view
    X   = X.copy()

    X.rename(columns={"PH": 'univariate, max(S)', 'signal_std':'univariate, std(S)', 'signal_mean':'univariate, mean(S)', 'peak area':'univariate, area(S)', \
                        'dS_dV_area':'univariate, area(dS/dV)', 'dS_dV_max_peak':'univariate, max(dS/dV)', 'dS_dV_min_peak':'univariate, min(dS/dV)',\
                    'dS_dV_peak_diff':'univariate, max(dS/dV) - min(dS/dV)', \
                    'peak V':'univariate, V_max(S)', 'dS_dV_max_V':'univariate, V_max(dS/dV)', 'dS_dV_min_V':'univariate, V_min(dS/dV)',\
        }, inplace = True)

    if not(split): 
        if not(normalization): return X, y
        else:
            scaler = select_normalizer(standardize_type)
            scaler.fit(X[y==0]) if (normalize_blanks and (standardize_type=='mean_std')) else scaler.fit(X)
            X_ = scaler.transform(X)

            return pd.DataFrame(X_, columns=X.columns), y

    # Split the total dataset into training (60%) and testing (40%) dataset
    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.4, shuffle=True, random_state=20, stratify=stratified_y)

    scaler = None
    if normalization:
        # Initialize the StandardScaler
        scaler = select_normalizer(standardize_type)

        # Fit the scaler to only the blank of the training dataset
        if (normalize_blanks and standardize_type=='mean_std'): scaler.fit(X_train[y_train==0].copy())

        # Fit the scaler to the training dataset
        else: scaler.fit(X_train)

        # Transform the data
        X_train_normalize = scaler.transform(X_train)
        
        # Transform the data
        X_test_normalize =   scaler.fit_transform(X_test) if test_nor_separate else scaler.transform(X_test)


        X_train = pd.DataFrame(X_train_normalize,  columns=X.columns)
        X_test  = pd.DataFrame(X_test_normalize,   columns=X.columns)

    else:  X_train, X_test = pd.DataFrame(X_train, columns=X.columns), pd.DataFrame(X_test, columns=X.columns)

    return (X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True)), scaler