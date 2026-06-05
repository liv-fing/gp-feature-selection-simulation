
'''
Clean code for Gaussian Process regression with zero mean function, using pymc


/// to run:
    cd Desktop/GitHub/IEMS399-GP/phase5-writing
    conda activate venv

    python model0_ZeroMeanGP.py  --numtune 0 --numdraws 10000

'''

# IMPORTS
from datetime import date, datetime
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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# CUSTOM IMPORTS
from helper_funcs.data_setup import make_run_dir, diabetes_data_init, synthetic_data_init, new_data_init

from helper_funcs.predicting import predictions, plot_predictions, rbf_kernel
from helper_funcs.zero_mean import ZERO_process_pymc_results, ZERO_make_trace_plots, ZERO_predict_and_save


class ZERO_GPLikelihood_pymc:
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
    

    def logProbability(self, sigma2_noise, sigma2_gp, ell):
        """ 
        GP log-likelihood only        
        y(*) = f(*) + g(*) + \epsilon

        """
        residuals = self.y #- pt.dot(self.X, beta) # y - XB
        K = self.rbf_kernel(self.X, ell, sigma2_gp) # C = K + \sigma^2_GP * I
        C = K + sigma2_noise * pt.eye(self.n)

        log_lik_gp = (
            -0.5 * self.n * pt.log(2 * pt.pi)
            -0.5 * pt.nlinalg.slogdet(C)[1]
            -0.5 * pt.dot(residuals, pt.slinalg.solve(C, residuals))
            )
        
        return log_lik_gp
    
# main call
def main(numdraws=1000, 
         numtune=500, 
         numchains = 4, 
         steptype = 'Metropolis', 
         seed=1, 
         test_lambda=True, 
         fixed_lambda_val=5, 
         predict = False,
         data = 'diabetes', 
         mechanism = 'ZEROMEAN_GP'):

    # make output directory
    outdir = make_run_dir(
        mechanism=mechanism,
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
    if data == 'new':
        Xtrain, Xtest, ytrain, ytest = new_data_init()
    elif data == 'synthetic':
        Xtrain, Xtest, ytrain, ytest = synthetic_data_init(
            size = 1000,
            active_proportion = 10,
            noise = 0.1,
            seed = 0,
            rep = 1,
            features = 'all')

    with pm.Model() as model:


        # priors
        sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
        sigma2_gp = pm.InverseGamma("sigma2_gp", 3.0, 1.0)
        ell = pm.Lognormal("ell", mu=-2, sigma=1, shape=Xtrain.shape[1])

        gp_likelihood = ZERO_GPLikelihood_pymc(Xtrain, ytrain) 

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

    posterior_df = ZERO_process_pymc_results(outdir, trace, Xtrain)
    posterior_df.to_csv(os.path.join(outdir, 'posterior_results.csv'), index=False)
    ZERO_make_trace_plots(outdir, trace, Xtrain)
    

    summary = pm.summary(trace, hdi_prob=0.95)  # 95% credible interval
    print('95% Credible Interval', summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))

    # make predictions and save
    ZERO_predict_and_save(trace, Xtrain, ytrain, Xtest, ytest, outdir)


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
    parser.add_argument('--data', type=str, default='diabetes', help = 'Choose data: diabetes, or synthetic')
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