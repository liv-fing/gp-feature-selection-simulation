
# mechanism_v4.py

'''
main differences: 
- regularization handled only in prior (no lambda term in likelihood)
- likelihood only uses gp structure
- adding rmse function


/// to activate venv:
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

import pytensor.tensor as pt
import pymc as pm
import arviz as az
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# make directory for the run
def make_run_dir(sampler="pymc", numdraws=2000, numtune=1000, numchains = 6, steptype = 'Metropolis', seed=1, test_lambda=False, fixed_lambda_val=0):
    if test_lambda:
        path =  f"results/v4/{sampler}/lbda{fixed_lambda_val}/dr{numdraws}_t{numtune}/ch{numchains}_{steptype}/sd{seed}"
        
    else:
        path =  f"results/v4/{sampler}/dr{numdraws}_t{numtune}/ch{numchains}_{steptype}/sd{seed}"
    os.makedirs(path, exist_ok=True)
    metadata = { # create metadata json file
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


def compute_rmse(Xtrain, ytrain, Xtest, ytest):
    ...


def diabetes_data_init(choose_features = 'all'):
    '''
    load and prepare diabetes data from sklearn
    '''
    from sklearn.datasets import load_diabetes
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
        return K

    def logProbability(self, sigma2_noise, sigma2_gp, beta, ell):
        """ 
        GP log-likelihood only        
        y(*) = f(*) + g(*) + \epsilon

        """
        residuals = self.y - pt.dot(self.X, beta) # y - XB
        C = self.rbf_kernel(self.X, ell, sigma2_gp)  

        log_lik_gp = (
            -0.5 * self.n * pt.log(2 * pt.pi)
            -0.5 * pt.nlinalg.slogdet(C + sigma2_noise * pt.eye(self.n))[1]
            -0.5 * pt.dot(residuals, pt.slinalg.solve(C + sigma2_noise * pt.eye(self.n), residuals))
            )
        
        return log_lik_gp #+ log_lik_beta
    


    

# main call
def main(numdraws=1000, numtune=500, numchains = 4, steptype = 'Metropolis', seed=1, test_lambda=True, fixed_lambda_val=5):

    # make output directory
    outdir = make_run_dir(
        sampler="pymc", 
        numdraws=numdraws, 
        numtune=numtune, 
        numchains=numchains, 
        steptype=steptype, 
        seed=seed, 
        test_lambda=test_lambda, 
        fixed_lambda_val=fixed_lambda_val)

    # set up sklearn diabetes data
    Xtrain, Xtest, ytrain, ytest = diabetes_data_init(choose_features = 'all')

    with pm.Model() as model:

        # lambda (deterministic or gamma)
        if test_lambda:
            fixed_lambda2_val = fixed_lambda_val **2
            lambda2 = pm.Deterministic("lambda2", pm.math.constant(fixed_lambda2_val)) 
        else:
            lambda2 = pm.Gamma("lambda2", 1.0, 1.78) # hyperparameters from bayesian lasso

        # priors
        sigma2_noise = pm.InverseGamma("sigma2_noise", 3.0, 1.0)
        sigma2_gp = pm.InverseGamma("sigma2_gp", 3.0, 1.0)
        tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
        tau = pm.math.sqrt(tau2)
        beta = pm.Normal("beta", 0.0, sigma2_noise * tau, shape=Xtrain.shape[1])  # depends on tau, sigma2_noise
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
    args = parser.parse_args()

    main(
        numdraws = args.numdraws,
        numtune = args.numtune,
        numchains = args.numchains,
        seed = args.seed,
        steptype = args.step,
        test_lambda = args.test_lambda,
        fixed_lambda_val = args.fixed_lambda_val
        )