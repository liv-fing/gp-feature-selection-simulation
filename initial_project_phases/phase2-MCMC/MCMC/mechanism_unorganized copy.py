#!/usr/bin/env python
# coding: utf-8

'''
input: 
- output_file
- desc (descriptive label for the run)
- numwalkers
- numsteps
- numburn
- a_val

output:
- creates output directory labelled by number of walkers used:
    - {numwalkers}walkers/{sampler_name}_{desc}/
        - posterior.pkl
        - posterior_summary.csv
        - sampler.pickle
    - {numwalkers}walkers/{sampler_name}_{sampler_name}{desc}/\
        - chain.dat
        - sampler.pickle
    - {numwalkers}walkers/{sampler_name}_{desc}_result.json


'''

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core.utils import random
import json
import scipy.special
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# define RBF kernel function
def rbf_kernel(X, ell, sigma_gp):
    """
    compute the RBF kernel
    """
    N = X.shape[0]
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            diff = (X[i] - X[j]) / ell
            K[i, j] = sigma_gp**2 * np.exp(-0.5 * np.dot(diff, diff))
    return K

# define custom likelihood class for penalized GP regression
class PenalizedGPLikelihood(bilby.Likelihood):
    def __init__(self, X, y):
        # store data
        self.X = np.asarray(X)
        self.y = np.asarray(y)

        # define parameters
        parameters = {}

        for i in range(self.X.shape[1]):
            parameters[f"ell{i}"] = None # lengthscales (ells)     
            parameters[f"nu{i}"] = None # nu: random variable for beta calculation
            parameters[f"zeta{i}"] = None # zeta: random variable for tau calculation

        # noise parameters
        parameters["inv_sigma_noise"] = None # remember this is gamma not inverse gamma, so use (alpha, 1/b)
        parameters["inv_sigma_gp"] = None # remember this is gamma not inverse gamma, so use (alpha, 1/b)
        parameters["lambda2"] = None # lambda_squared 

        super().__init__(parameters=parameters)

    def log_likelihood(self):

        # extract parameters
        inv_sigma_noise = self.parameters["inv_sigma_noise"]
        inv_sigma_gp = self.parameters["inv_sigma_gp"]
        ells = np.array([self.parameters[f"ell{i}"] for i in range(self.X.shape[1])])
        lbda2 = self.parameters["lambda2"] # sample lambda squared

        # --- transformers ---
        nus = np.array([self.parameters[f"nu{i}"] for i in range(self.X.shape[1])]) # random var for beta
        zetas = np.array([self.parameters[f"zeta{i}"] for i in range(self.X.shape[1])]) 
        
        # calculate n and p
        n = self.X.shape[0]
        p = self.X.shape[1]

        # --- transform sampled gamma to desired inverse gamma ---
        sigma_gp = 1/inv_sigma_gp if inv_sigma_gp is not None else None
        sigma_noise = 1/inv_sigma_noise if inv_sigma_noise is not None else None

        # --- transform zetas to tau^2 ---
        # each tau^2 \sim expoential(\lambda^2 / 2)
        if (zetas is not None and lbda2 is not None):
            tau_sqs = (2 * zetas) / lbda2

        # --- transform nu to beta ---                   
        if (nus is not None and tau_sqs is not None and sigma_noise is not None):
            betas = nus * np.sqrt(tau_sqs) * sigma_noise 
        
        C = rbf_kernel(self.X, ells, sigma_gp) # calculate covariance matrix C
        residuals = self.y - self.X @ betas  # calculate residuals
        D = np.diag(tau_sqs) # calculate diagonal matrix D

        # compute ahead of time
        sigma2_D = sigma_noise**2 * D
        
        # --- log likelihood components ---
        log_lik_gp = (
            -0.5 * n * np.log(2 * np.pi) 
            -0.5 * np.linalg.slogdet(C + sigma_noise**2 * np.eye(n))[1]
            -0.5 *(residuals).T @ np.linalg.solve(C + sigma_noise**2 * np.eye(n), residuals))

        # rewritten to avoid inversion of D
        if np.any(sigma2_D == 0):
            log_lik_beta = (
                -0.5 * p * np.log(2 * np.pi)
                -0.5 * np.linalg.slogdet(sigma_noise**2 * D)[1]
                -0.5 * betas.T @ np.linalg.solve(sigma_noise**2 * D, betas)
            )
        else:  
            inv_sigma2_D = 1 / sigma2_D
            quadratic = np.dot((betas.ravel()**2), inv_sigma2_D)
            logdet = np.sum(np.log(sigma2_D))
            
            log_lik_beta = (
                -0.5 * p * np.log(2 * np.pi)
                -0.5 * logdet
                -0.5 * quadratic)
        
        return log_lik_gp + log_lik_beta


def main(output_file = "result.txt", desc = "baseline", numwalkers = 50, numsteps = 10000, numburn = 1000, a_val = 2.0):
    """
    running mcmc sampler for penalized GP regression on diabetes data
    """

    # set up sampler
    sampler_name = "emcee"
    label = f"{sampler_name}_{desc}"
    outdir = f"{numwalkers}walkers"    
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    run_dir = os.path.join(outdir, label)
    os.makedirs(run_dir, exist_ok=True)
    random.seed(123)

    # load diabetes data
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    selected_features = ['bmi', 'bp', 's1']
    X = X[selected_features]
    label_names = diabetes.feature_names
    y = diabetes.target
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=22)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain) # fit and scale training data
    Xtest = scaler.transform(Xtest) # scale test data

    # make priors
    priors = dict()
    priors["inv_sigma_noise"] = bilby.core.prior.Gamma(1, 1, "inv_sigma_noise")  
    priors["inv_sigma_gp"] = bilby.core.prior.Gamma(1, 1, "inv_sigma_gp")  
    priors["lambda2"] = bilby.core.prior.Gamma(1.0, 1.78, name="lambda2") # positive

    for i in range(Xtrain.shape[1]):
        priors[f"zeta{i}"] = bilby.core.prior.Exponential(1, f"zeta{i}") # exponential with mean 1 = rate 1
        priors[f"ell{i}"] = bilby.core.prior.LogNormal(0, 1, f"ell{i}") # define log-normal priors for each lengthscale
        priors[f"nu{i}"] = bilby.core.prior.Normal(0,1, f"nu{i}") # standard normal for nu

    # define the likelihood function that we defined earlier
    likelihood = PenalizedGPLikelihood(
        X = Xtrain,
        y = ytrain)

    # run MCMC sampler
    result = bilby.run_sampler(
        likelihood=likelihood, # likelihood function
        priors=priors, # prior distributions
        sampler=sampler_name, # other options for mcmc are emcee, zeus, , pyemcee, ptemcee, bilby-mcmc 
        nwalkers = numwalkers , # need > 2 x number of parameters
        nsteps = numsteps,
        nburn = numburn,
        sampler_kwargs = dict(a=a_val),
        outdir=outdir,
        label=label)


    ### --- process results ---


    # create new dataframe to store results
    result_tab = result.posterior.copy()

    # calculate tau_sq from samples
    result_tab["lambda"] = np.sqrt(result_tab["lambda2"].values)
    for i in range(X.shape[1]):
        zetas = result_tab[f"zeta{i}"].values
        tau_sqs = (2 * zetas) / result_tab["lambda2"].values
        result_tab[f"tau_sq{i}"] = tau_sqs

    # calculate beta from samples
    result_tab["sigma_noise"] = 1 / result_tab["inv_sigma_noise"].values

    for i in range(X.shape[1]):
        nus = result_tab[f"nu{i}"].values
        tau_sqs = result_tab[f"tau_sq{i}"].values
        betas = nus * np.sqrt(tau_sqs) * result_tab["sigma_noise"].values
        result_tab[f"beta{i}"] = betas

    # clean up dataframe for easy viewing
    clean_tab = result_tab.copy()
    drop_cols = [f"zeta{i}" for i in range(X.shape[1])] + ["lambda2", "inv_sigma_noise", "inv_sigma_gp"] + [f"nu{i}" for i in range(X.shape[1])]
    clean_tab = clean_tab.drop(columns=drop_cols) # drop unneeded columns
    clean_tab = clean_tab.reindex(sorted(clean_tab.columns), axis=1) # sort columns alphabetically
        
    # save to csv
    clean_tab.to_csv(os.path.join(run_dir, "posterior_summary.csv"), index=False)

    # save full posterior
    with open(os.path.join(run_dir, "posterior.pkl"), "wb") as f:
        pickle.dump(result.posterior, f)
    
    # compute and save acceptance fraction
    with open(os.path.join(run_dir, "sampler.pickle"), "wb") as f:
        pickle.dump(result.sampler, f)
    af = result.sampler.acceptance_fraction
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write("Per-walker acceptance fractions:\n")
        f.write(", ".join([f"{x:.3f}" for x in af]) + "\n")
        f.write(f"Mean acceptance fraction: {af.mean():.3f}\n")
        f.write(f"Number of walkers: {numwalkers}, steps: {numsteps}, burn-in: {numburn}\n")
        f.write(f"Output directory: {run_dir}\n")

    # plot and save trace plots
    fig, axes = plt.subplots(X.shape[1], 1, figsize=(10, 3*X.shape[1]), sharex=True)

    for i in range(X.shape[1]):
        ax = axes[i]
        traces = clean_tab[f"beta{i}"].to_numpy().reshape(50, 9000).T
        ax.plot(traces, alpha=0.5, linewidth=0.5)  # matplotlib will plot each walker as a line
        ax.set_ylim([-60, 60])
        ax.set_title(f"Trace plot for beta{i}")
        ax.set_ylabel(f"beta{i}")

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "beta_trace_plots.png"))
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MCMC sampler for penalized GP regression on diabetes data.")
    
    parser.add_argument('--output_file', type=str, default='result.txt', help='Output file name (default: result.txt)')
    parser.add_argument('--desc', type=str, default='baseline', help='Description label for the run (default: baseline)')
    parser.add_argument('--numwalkers', type=int, default=50, help='Number of walkers (default: 50)')
    parser.add_argument('--numsteps', type=int, default=10000, help='Number of steps (default: 10000)')
    parser.add_argument('--numburn', type=int, default=1000, help='Number of burn-in steps (default: 1000)')
    parser.add_argument('--a_val', type=float, default=2.0, help='Stretch move scale parameter a (default: 2.0)')
    args = parser.parse_args()


    main(output_file = args.output_file, 
         numwalkers = args.numwalkers,
            numsteps = args.numsteps,
            numburn = args.numburn,
            a_val = args.a_val)