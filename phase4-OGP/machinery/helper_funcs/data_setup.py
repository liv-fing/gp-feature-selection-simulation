# data_setup
'''
load functions with
from helper_funcs.data_setup import
    make_run_dir
    diabetes_data_init
    synthetic_data_init
    starting_points 

'''


# IMPORTS
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from pathlib import Path


# RUN DIRECTORY
def make_run_dir(sampler="pymc", 
                 mechanism="orig", 
                 numdraws=2000, 
                 numtune=1000, 
                 numchains = 6, 
                 steptype = 'Metropolis', 
                 seed=1, 
                 test_lambda=False, 
                 fixed_lambda_val=0, 
                 data = 'diabetes'):
    if test_lambda:
        day_month = datetime.now().strftime("%d-%m")
        path =  f"results/{mechanism}/{day_month}_{data}/{steptype}/lbda{fixed_lambda_val}/dr{numdraws}_t{numtune}_ch{numchains}/sd{seed}"
        
    else:
        path =  f"results/{mechanism}/{day_month}_{data}/{steptype}/dr{numdraws}_t{numtune}_ch{numchains}/sd{seed}"
    os.makedirs(path, exist_ok=True)
    metadata = { # create metadata json file
        "data": data,
        "mechanism": mechanism,
        "sampler": sampler,
        "draws": numdraws,
        "tune": numtune,
        "chains": numchains,
        "seed": seed,
        "step": steptype,
        "data": data,
        "datetime": datetime.now().isoformat() # capture run time
        }
    
    if test_lambda:
        metadata["test_lambda"] = test_lambda
        metadata["fixed_lambda_val"] = fixed_lambda_val

    with open(f"{path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path




# DIABETES DATA
def diabetes_data_init(choose_features = 'all'):
    '''
    load and prepare diabetes data from sklearn
    '''
    
    diabetes = load_diabetes(as_frame=True)

    if choose_features == 'all':
        selected_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    else:
        selected_features = ['bmi', 'bp', 's1']

    X = diabetes.data
    X = X[selected_features]
    y = diabetes.target

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=22)

    Xtrain = Xtrain.to_numpy()
    Xtest  = Xtest.to_numpy()
    ytrain = np.asarray(ytrain)
    ytest  = np.asarray(ytest)  

    return Xtrain, Xtest, ytrain, ytest

# SYNTHETIC DATA
def synthetic_data_init(
        size = 2000,
        active_proportion = 50,
        noise = 0.1,
        seed = 6,
        rep = 1,
        features = 'all'):

    '''
    load and set up the synthetic datasets
    need to add a way to save meta data for synthetic data
    
    '''
    # set up path according to input params
    base_path = Path("/Users/liviafingerson/Desktop/GitHub/IEMS399-GP/synthetic_data_large/simulated_datasets_large_coef") # main folder
    folder_name = f"N11000_AP{active_proportion}_noise{noise}_seed{seed}" # first folder
    subfolder_name = f"Size{size}" # second folder
    path = base_path / folder_name / subfolder_name / f"Rep{rep}.csv"

    df = pd.read_csv(path)

    if features == 'all':
        X = df.drop(columns=['y']).values
    else:
        X = df[features].values

    y = df['y'].values

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=22)

    # scale
    # scaler = StandardScaler()
    # Xtrain = scaler.fit_transform(Xtrain) # fit and scale training data
    # Xtest = scaler.transform(Xtest) # scale test data
    return Xtrain, Xtest, ytrain, ytest

# STARTING POINTS FOR OLS
def starting_points(Xtrain, ytrain):
    '''
    set starting points for sampler using OLS estimates for beta
    '''
    start = {}

    # OLS estimates for beta
    XTX_inv = np.linalg.inv(Xtrain.T @ Xtrain)
    beta_ols = XTX_inv @ Xtrain.T @ ytrain
    start['beta'] = beta_ols

    #MLE estimate for sigma2_noise
    residuals = ytrain - Xtrain @ beta_ols
    sigma2_noise_mle = np.var(residuals, ddof=1) 
    start['sigma2_noise'] = sigma2_noise_mle

    return start