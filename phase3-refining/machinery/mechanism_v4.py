
# mechanism_v4.py

'''
main differences: 
- regularization handled only in prior (no lambda term in likelihood)
- likelihood only uses gp structure
- adding prediction and rmse function

- double checked for C vs K in covariance matrix
- double checked which sigma

/// to run:
cd Desktop/GitHub/IEMS399-GP/phase3-refining
conda activate venv
python machinery/mechanism_v4.py --test_lambda True --fixed_lambda_val 2.0 --numtune 0 --numdraws 10000
/// 

'''

# imports
from datetime import datetime
from bleach import clean
from networkx import sigma
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

import pytensor.tensor as pt
import pymc as pm
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.datasets import load_diabetes


# make directory for the run
def make_run_dir(sampler="pymc", numdraws=2000, numtune=1000, numchains = 6, steptype = 'Metropolis', seed=1, test_lambda=False, fixed_lambda_val=0, data = 'diabetes'):
    if test_lambda:
        path =  f"results/v4_large/{data}_{sampler}/lbda{fixed_lambda_val}/dr{numdraws}_t{numtune}/ch{numchains}_{steptype}/sd{seed}"
        
    else:
        path =  f"results/v4_large/{data}_{sampler}/dr{numdraws}_t{numtune}/ch{numchains}_{steptype}/sd{seed}"
    os.makedirs(path, exist_ok=True)
    metadata = { # create metadata json file
        "data": data,
        "sampler": sampler,
        "draws": numdraws,
        "tune": numtune,
        "chains": numchains,
        "seed": seed,
        "step": steptype,
        "datetime": datetime.now().isoformat() # capture run time
        }
    with open(f"{path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path

def process_pymc_results(outdir, trace, X):
    
    num_features = X.shape[1]

    posterior_ds = trace.posterior.stack(sample=("chain", "draw"))

    clean_df = pd.DataFrame()

    clean_df['sigma2_noise'] = (posterior_ds['sigma2_noise']).values
    clean_df['sigma2_gp'] = (posterior_ds['sigma2_gp']).values
    clean_df['lambda'] = np.sqrt(posterior_ds['lambda2'].values)

    for i in range(num_features):
        ell_i = posterior_ds['ell'].isel(ell_dim_0=i).values
        beta_i = posterior_ds['beta'].isel(beta_dim_0=i).values

        # add to dataframe
        clean_df[f'beta{i}'] = beta_i
        clean_df[f'ell{i}'] = ell_i

    # --- save to CSV ---
    os.makedirs(outdir, exist_ok=True)
    clean_df.to_csv(os.path.join(outdir, "posterior_summary.csv"), index=False)
    print(f"Posterior summary saved to {outdir}/posterior_summary.csv")

    return clean_df


def make_trace_plots(outdir, trace, X):
    num_features = X.shape[1]

    # plot: beta, sigma2_noise, sigma2_gp, lambda, ell
    posterior = trace.posterior

    total_axes = 3 + num_features + num_features  # scalars + betas + ells
    fig, axes = plt.subplots(total_axes, 1, figsize=(10, 3 * total_axes))
    
    if num_features + 4 == 1:
        axes = [axes]

    scalars = {
        'sigma2_noise': posterior['sigma2_noise'],
        'sigma2_gp': posterior['sigma2_gp'],
        'lambda': np.sqrt(posterior['lambda2'])
    }

    for i, (name, val) in enumerate(scalars.items()):
        ax = axes[i]
        for chain in range(val.sizes['chain']):
            ax.plot(val.isel(chain=chain).values, alpha=0.5)
        ax.set_title(f'Trace plot for {name}')
        ax.set_ylabel(name)

    # betas
    for i in range(num_features):
        ax = axes[i + 3]  # offset by number of scalars
        beta_i = posterior['beta'].isel(beta_dim_0=i)

        for chain in range(beta_i.sizes['chain']):
            ax.plot(beta_i.isel(chain=chain).values, alpha=0.4)
            #ax.set_ylim(-50, 50)  

        ax.set_title(f'Trace plot for beta{i}')
        ax.set_ylabel(f'beta{i}')

    # ells
    for i in range(num_features):
        ax = axes[i + 3 + num_features]  # offset after scalars + betas
        ell_i = posterior['ell'].isel(ell_dim_0=i)
        for chain in range(ell_i.sizes['chain']):
            ax.plot(ell_i.isel(chain=chain).values, alpha=0.4)
        ax.set_title(f'Trace plot for ell{i}')
        ax.set_ylabel(f'ell{i}')

    #axes[-1].set_xlabel('Draw')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'trace_plots.png'))
    plt.close()

    print(f"Trace plots saved to {outdir}/trace_plots.png")



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
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain) # fit and scale training data
    Xtest = scaler.transform(Xtest) # scale test data

    return Xtrain, Xtest, ytrain, ytest


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



def rbf_kernel(X1, X2, ell, sigma2):
        """ 
        compute RBF kernel using numpy 
        """
        X1_scaled = X1 / ell
        X2_scaled = X2 / ell
        diff = X1_scaled[:, None, :] - X2_scaled[None, :, :]
        squared_dist = np.sum(diff ** 2, axis=2)
        K = sigma2 * np.exp(-0.5 * squared_dist)
        return K


def predict_and_save(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    ''' 
    make predictions
    uses posterior means to make predictions
    computes rmse
    saves results to file

    '''

    # using mean to test
    beta_hat = trace.posterior["beta"].mean(dim=("chain", "draw")).values
    ell_hat = trace.posterior["ell"].mean(dim=("chain", "draw")).values
    sigma2_gp_hat = trace.posterior["sigma2_gp"].mean(dim=("chain", "draw")).values
    sigma2_noise_hat = trace.posterior["sigma2_noise"].mean(dim=("chain", "draw")).values

    # extract residuals
    residuals = ytrain - Xtrain @ beta_hat

    # make covariance matrices
    Ktrain = rbf_kernel(Xtrain, Xtrain, ell_hat, sigma2_gp_hat)
    Ktest = rbf_kernel(Xtest, Xtrain, ell_hat, sigma2_gp_hat)

    Ctrain = Ktrain + sigma2_noise_hat * np.eye(len(Xtrain))
    alpha = np.linalg.solve(Ctrain, residuals)

    # create predictions
    ftrain = Ktrain @ alpha
    ftest = Ktest @ alpha

    ytrain_pred = Xtrain @ beta_hat + ftrain
    ytest_pred = Xtest @ beta_hat + ftest

    train_rmse = root_mean_squared_error(ytrain, ytrain_pred)
    test_rmse = root_mean_squared_error(ytest, ytest_pred)

    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')


    run_dir = Path(run_dir)

    # create a file to save predictions
    df_train = pd.DataFrame({
        "split": "train",
        "y_true": ytrain,
        "y_pred": ytrain_pred
    })
    df_test = pd.DataFrame({
        "split": "test",
        "y_true": ytest,
        "y_pred": ytest_pred
    })
    df_predictions = pd.concat([df_train, df_test], ignore_index=True)
    df_predictions.to_csv(run_dir / "predictions_post_mean.csv", index=False)

    # create a file to save rmse
    df_rmse = pd.DataFrame({
        "split": ["train", "test"],
        "rmse": [train_rmse, test_rmse]
    })
    df_rmse.to_csv(run_dir / "rmse_results.csv", index=False)



class GPLikelihood_pymc:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = X.shape

    def rbf_kernel(self, X, ell, sigma2_gp):
        """ 
        compute RBF kernel using pytensor 

        """
        X_scaled = X / ell
        diff = X_scaled[:, None, :] - X_scaled[None, :, :]
        squared_dist = pt.sum(diff ** 2, axis=2)
        K = sigma2_gp * pt.exp(-0.5 * squared_dist)
        return K # returns K, not C = K + \sigma^2_GP * I

    def logProbability(self, sigma2_noise, sigma2_gp, beta, ell):
        """ 
        GP log-likelihood only        
        y(*) = f(*) + g(*) + \epsilon

        """
        residuals = self.y - pt.dot(self.X, beta) # y - XB
        K = self.rbf_kernel(self.X, ell, sigma2_gp) # C = K + \sigma^2_GP * I
        C = K + sigma2_noise * pt.eye(self.n)

        log_lik_gp = (
            -0.5 * self.n * pt.log(2 * pt.pi)
            -0.5 * pt.nlinalg.slogdet(C)[1]
            -0.5 * pt.dot(residuals, pt.slinalg.solve(C, residuals))
            )
        
        return log_lik_gp
    
# main call
def main(numdraws=1000, numtune=500, numchains = 4, steptype = 'Metropolis', seed=1, test_lambda=True, fixed_lambda_val=5, data = 'diabetes'):

    # make output directory
    outdir = make_run_dir(
        data = data,
        sampler="pymc", 
        numdraws=numdraws, 
        numtune=numtune, 
        numchains=numchains, 
        steptype=steptype, 
        seed=seed, 
        test_lambda=test_lambda, 
        fixed_lambda_val=fixed_lambda_val)

    # set up data
    if data == 'diabetes':
        Xtrain, Xtest, ytrain, ytest = diabetes_data_init(choose_features = 'all')
    elif data == 'synthetic':
        Xtrain, Xtest, ytrain, ytest = synthetic_data_init(
            size = 1000,
            active_proportion = 10,
            noise = 0.1,
            seed = 0,
            rep = 1,
            features = 'all')

    with pm.Model() as model:

        # lambda (deterministic or gamma)
        if test_lambda:
            fixed_lambda2_val = fixed_lambda_val **2
            lambda2 = pm.Deterministic("lambda2", pm.math.constant(fixed_lambda2_val)) 
        else:
            lambda2 = pm.Gamma("lambda2", 1.0, 1.78) # hyperparameters from bayesian lasso

        # priors
        #sigma2_noise = pm.InverseGamma("sigma2_noise", 3.0, 1.0)
        sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
        sigma2_gp = pm.InverseGamma("sigma2_gp", 3.0, 1.0)
        tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
        tau = pm.math.sqrt(tau2)
        beta = pm.Normal("beta", 0.0, pm.math.sqrt(sigma2_noise * tau), shape=Xtrain.shape[1])  # depends on tau, sigma2_noise
        ell = pm.Lognormal("ell", 0.0, 1.0, shape=Xtrain.shape[1])

        gp_likelihood = GPLikelihood_pymc(Xtrain, ytrain) 

        
        # step type
        if steptype == 'Metropolis':
            step = pm.Metropolis()
        elif steptype == 'NUTS':
            step = pm.NUTS()

        # likelihood function 
        pm.Potential(
            "gp_likelihood",
            gp_likelihood.logProbability(
                sigma2_noise,
                sigma2_gp,
                beta,
                ell) )

        # run model
        trace = pm.sample(
            draws=numdraws,
            tune=numtune,
            chains=numchains,
            step=step,
            random_seed=seed,
            progressbar=True
        )

    # save results

    posterior_df = process_pymc_results(outdir, trace, Xtrain)
    posterior_df.to_csv(os.path.join(outdir, 'posterior_results.csv'), index=False)
    make_trace_plots(outdir, trace, Xtrain)
    

    summary = pm.summary(trace, hdi_prob=0.95)  # 95% credible interval
    print('95% Credible Interval', summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))

    # make predictions and save
    predict_and_save(trace, Xtrain, ytrain, Xtest, ytest, outdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCMC sampler for penalized GP regression on diabetes data.")
    parser.add_argument('--numdraws', type=int, default=2000, help='Number of draws (default: 2000)')
    parser.add_argument('--numtune', type=int, default=1000, help='Number of tuning steps (default: 1000)')
    parser.add_argument('--numchains', type=int, default=6, help='Number of chains (default: 6)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--step', type=str, default='Metropolis', help='Sampler step type (default: Metropolis)')
    parser.add_argument('--test_lambda', type=bool, default=True, help = 'Experiment with a set lambda?')
    parser.add_argument('--fixed_lambda_val', type=float, default=5, help = 'If testing lambda, choose fixed val')
    parser.add_argument('--data', type=str, default='diabetes', help = 'Choose data: diabetes or synthetic')
    args = parser.parse_args()

    main(
        numdraws = args.numdraws,
        numtune = args.numtune,
        numchains = args.numchains,
        seed = args.seed,
        steptype = args.step,
        test_lambda = args.test_lambda,
        fixed_lambda_val = args.fixed_lambda_val,
        data = args.data
        )