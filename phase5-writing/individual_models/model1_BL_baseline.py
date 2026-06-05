'''
Clean code for baseline Bayesian Lasso with pymc

/// to run:
    cd Desktop/GitHub/IEMS399-GP/phase5-writing
    conda activate venv

    python model1_BL_baseline.py --test_lambda True --fixed_lambda_val 2.0 --numtune 0 --numdraws 10000

    for lbda in  0.001 0.01 0.1; do
    python model1_BL_baseline.py \
        --test_lambda True \
        --fixed_lambda_val "$lbda" \
        --numtune 0 \
        --numdraws 15000 \
        --numchains 4 \
        --data synthetic \
        --step metropolis 
    done
'''

# IMPORTS
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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# CUSTOM IMPORTS
from helper_funcs.data_setup import make_run_dir, diabetes_data_init, synthetic_data_init, starting_points, new_data_init
from helper_funcs.result_processing import process_pymc_results, make_trace_plots, make_trace_plots_ogp
from helper_funcs.predicting import predictions_lasso, predictions_ogp_betanosample, predictions, plot_predictions, rbf_kernel

# MAIN CALL
def main(
    numdraws=1000, 
    numtune=500, 
    numchains = 4, 
    steptype = 'Metropolis', 
    seed=1, 
    test_lambda=False, 
    fixed_lambda_val=0, 
    data = 'diabetes', 
    predict = False,
    mechanism = 'BL_baseline'
):
    
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
    
    # data 
    if data == 'diabetes':
        Xtrain, Xtest, ytrain, ytest = diabetes_data_init(choose_features = 'all')
    if data == 'new':
        Xtrain, Xtest, ytrain, ytest = new_data_init()
    elif data == 'synthetic':
        Xtrain, Xtest, ytrain, ytest = synthetic_data_init(size = 1000, 
                                                           active_proportion = 10, 
                                                           noise = 0.1, 
                                                           seed = 0, 
                                                           rep = 1, 
                                                           features = 'all')

    with pm.Model() as model:

        # priors
        sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
        if test_lambda: # deterministic lambda
            fixed_lambda2_val = fixed_lambda_val **2
            lambda2 = pm.Deterministic("lambda2", pm.math.constant(fixed_lambda2_val)) 
        else: # sampled lambda
            lambda2 = pm.Gamma("lambda2", 1.0, 1.78) # hyperparameters from bayesian lasso
        tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
        beta = pm.Normal("beta", mu = 0.0, sigma = pt.sqrt(sigma2_noise * tau2), shape=Xtrain.shape[1])  # depends on tau^2, sigma2_noise, but expects sd not var

        # likelihood function
        y_obs = pm.Normal(
            "y_obs", 
            mu=pt.dot(Xtrain, beta), 
            sigma=pt.sqrt(sigma2_noise), 
            observed=ytrain)

        # step type
        if steptype == 'Metropolis':
            step = pm.Metropolis()
        elif steptype == 'NUTS':
            step = pm.NUTS(
                target_accept=0.9
            )
        
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

    # trace plots
    make_trace_plots(outdir, trace, Xtrain, mechanism)

    # summary
    summary = pm.summary(trace, hdi_prob=0.95)
    print('\n95% Credible Interval\n', summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))

    if predict:
        predictions_lasso(trace, Xtrain, ytrain, Xtest, ytest, outdir)
        print("Predictions calculated and saved, but no plots generated for lasso mechanism yet")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCMC sampler for penalized GP regression on diabetes data.")
    parser.add_argument('--numdraws', type=int, default=2000, help='Number of draws (default: 2000)')
    parser.add_argument('--numtune', type=int, default=1000, help='Number of tuning steps (default: 1000)')
    parser.add_argument('--numchains', type=int, default=6, help='Number of chains (default: 6)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--step', type=str, default='Metropolis', help='Sampler step type (default: Metropolis)')
    args = parser.parse_args()

    main(
        numdraws = args.numdraws,
        numtune = args.numtune,
        numchains = args.numchains,
        seed = args.seed,
        steptype = args.step
        )