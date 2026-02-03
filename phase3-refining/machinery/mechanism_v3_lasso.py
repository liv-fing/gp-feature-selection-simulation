
# mechanism_v3_lasso.py

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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# make directory for the run
def make_run_dir(sampler="pymc", numdraws=2000, numtune=1000, numchains = 4, steptype = 'Metropolis', seed=1):
    
    path =  f"results/lasso/all_features/v3/{sampler}_dr{numdraws}_t{numtune}/ch{numchains}_{steptype}/sd{seed}"
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


def process_pymc_results(outdir, trace, X): # process pymc trace into a clean dataframe
    num_features = X.shape[1]
    posterior_ds = trace.posterior.stack(sample=("chain", "draw"))
    clean_df = pd.DataFrame()
    clean_df['sigma2_noise'] = (posterior_ds['sigma2_noise']).values
    # clean_df['sigma2_gp'] = (posterior_ds['sigma2_gp']).values
    clean_df['lambda'] = np.sqrt(posterior_ds['lambda2'].values)
    for i in range(num_features):
        # ell_i = posterior_ds['ell'].isel(ell_dim_0=i).values
        beta_i = posterior_ds['beta'].isel(beta_dim_0=i).values
        # add to dataframe
        clean_df[f'beta{i}'] = beta_i
        # clean_df[f'ell{i}'] = ell_i
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
        # 'sigma2_gp': posterior['sigma2_gp'],
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

            # set automatic y-limits for better visualization
            ax.set_ylim(beta_i.min().values * 1.1, beta_i.max().values * 1.1)

        ax.set_title(f'Trace plot for beta{i}')
        ax.set_ylabel(f'beta{i}')

    # ells
    # for i in range(num_features):
    #     ax = axes[i + 3 + num_features]  # offset after scalars + betas
    #     ell_i = posterior['ell'].isel(ell_dim_0=i)
    #     for chain in range(ell_i.sizes['chain']):
    #         ax.plot(ell_i.isel(chain=chain).values, alpha=0.4)
    #     ax.set_title(f'Trace plot for ell{i}')
    #     ax.set_ylabel(f'ell{i}')

    #axes[-1].set_xlabel('Draw')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'trace_plots.png'))
    plt.close()

    print(f"Trace plots saved to {outdir}/trace_plots.png")




# main call
def main(numdraws=1000, numtune=500, numchains = 4, steptype = 'Metropolis', seed=1):

    # make output directory
    outdir = make_run_dir(
        sampler="pymc", 
        numdraws=numdraws, 
        numtune=numtune, 
        numchains=numchains, 
        steptype=steptype, 
        seed=seed)

    # load diabetes data from sklearn
    diabetes = load_diabetes(as_frame=True)
    X = diabetes.data
    y = diabetes.target

    # select features
    selected_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    X = X[selected_features]
    
    # center y
    y_centered = y - np.mean(y) # center y like in the bl paper

    # split and scale
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_centered, test_size=0.2, random_state=22)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(Xtrain) # fit and scale training data
    Xtest = scaler.transform(Xtest) # scale test data

    with pm.Model() as model:

        # priors

        #sigma2_noise = pm.InverseGamma("sigma2_noise", 5.0, 1.5)

        sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)

        lambda2 = pm.Gamma("lambda2", 1.0, 1.78) # hyperparameters from bayesian lasso
        tau2 = pm.Exponential("tau2", lambda2/2.0, shape=X.shape[1]) # depends on lambda2
        beta = pm.Normal("beta", mu = 0.0, sigma = pt.sqrt(sigma2_noise * tau2), shape=X.shape[1])  # depends on tau^2, sigma2_noise, but expects sd not var

        # using standard likelihood for testing
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
    make_trace_plots(outdir, trace, Xtrain)
    

    summary = pm.summary(trace, hdi_prob=0.95)  # 95% credible interval
    print(summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))


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