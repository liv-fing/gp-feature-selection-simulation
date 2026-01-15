
# mechanism_v2.py


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
def make_run_dir(sampler="pymc", numdraws=2000, numtune=1000, numchains = 6, steptype = 'Metropolis', seed=1):
    path =  f"results/{sampler}/dr{numdraws}/ch{numchains}_t{numtune}_{steptype}/sd{seed}"
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

    clean_df['sigma_noise'] = (1 / posterior_ds['inv_sigma_noise']).values
    clean_df['sigma_gp'] = (1 / posterior_ds['inv_sigma_gp']).values
    clean_df['lambda'] = np.sqrt(posterior_ds['lambda2'].values)

    for i in range(num_features):
        zeta_i = posterior_ds['zeta'].isel(zeta_dim_0=i).values
        nu_i = posterior_ds['nu'].isel(nu_dim_0=i).values
        ell_i = posterior_ds['ell'].isel(ell_dim_0=i).values

        tau_sq_i = 2 * zeta_i / posterior_ds['lambda2'].values
        beta_i = nu_i * np.sqrt(tau_sq_i) * clean_df['sigma_noise'].values

        # add to dataframe
        clean_df[f'zeta{i}'] = zeta_i
        clean_df[f'tau_sq{i}'] = tau_sq_i
        clean_df[f'nu{i}'] = nu_i
        clean_df[f'beta{i}'] = beta_i
        clean_df[f'ell{i}'] = ell_i

    # --- save to CSV ---
    os.makedirs(outdir, exist_ok=True)
    clean_df.to_csv(os.path.join(outdir, "posterior_summary.csv"), index=False)
    print(f"Posterior summary saved to {outdir}/posterior_summary.csv")

    return clean_df


def make_trace_plots(outdir, trace, X):
    num_features = X.shape[1]

    # plot: beta, sigma_noise, sigma_gp, lambda, ell
    posterior = trace.posterior

    total_axes = 3 + num_features + num_features  # scalars + betas + ells
    fig, axes = plt.subplots(total_axes, 1, figsize=(10, 3 * total_axes))
    
    if num_features + 4 == 1:
        axes = [axes]

    scalars = {
        'sigma_noise': 1 / posterior['inv_sigma_noise'],
        'sigma_gp': 1 / posterior['inv_sigma_gp'],
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
        # compute tau_sq and beta
        zeta_i = posterior['zeta'].isel(zeta_dim_0=i)
        nu_i = posterior['nu'].isel(nu_dim_0=i)
        lambda_val = posterior['lambda2']
        sigma_noise = 1 / posterior['inv_sigma_noise']
        tau_sq_i = 2 * zeta_i / lambda_val
        beta_i = nu_i * np.sqrt(tau_sq_i) * sigma_noise

        for chain in range(beta_i.sizes['chain']):
            ax.plot(beta_i.isel(chain=chain).values, alpha=0.4)
            ax.set_ylim(-100, 100)  

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


# custom log likelihood class
class PenalizedGPLikelihood_pymc:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = X.shape

    def rbf_kernel(self, X, ell, sigma2_gp):
        """ compute RBF kernel using pytensor """
        X_scaled = X / ell
        diff = X_scaled[:, None, :] - X_scaled[None, :, :]
        squared_dist = pt.sum(diff ** 2, axis=2)
        K = sigma2_gp * pt.exp(-0.5 * squared_dist)
        return K

    def logProbability(self, inv_sigma_noise, inv_sigma_gp, lambda2, zeta, ell, nu):
        """ compute custom log likelihood using pytensor """
        # transform
        sigma2_noise = 1 / inv_sigma_noise # variance
        sigma_noise = pt.sqrt(sigma2_noise) # std dev

        sigma2_gp = 1 / inv_sigma_gp # variance
        sigma_gp = pt.sqrt(sigma2_gp) # std dev

        tau_sq = 2 * zeta / lambda2
        beta = nu * pt.sqrt(tau_sq) * sigma_noise

        C = self.rbf_kernel(self.X, ell, sigma2_gp) 
        residuals = self.y - pt.dot(self.X, beta)
        D = pt.eye(self.p) * tau_sq
        sigma2_D = sigma2_noise * D

        log_lik_gp = (
            -0.5 * self.n * pt.log(2 * pt.pi)
            -0.5 * pt.nlinalg.slogdet(C + sigma2_noise * pt.eye(self.n))[1]
            -0.5 * pt.dot(residuals, pt.slinalg.solve(C + sigma2_noise * pt.eye(self.n), residuals))
            )
        
        #  add epsilon of 1e-12 for stability
        inv_sigma2_D = 1 / (sigma2_D + 1e-12)
        logdet = pt.sum(pt.log(sigma2_D + 1e-12))
        quadratic = pt.sum((beta**2) * inv_sigma2_D)

        log_lik_beta = (
            -0.5 * self.p * pt.log(2 * pt.pi)
            -0.5 * logdet
            -0.5 * quadratic
            )
        
        return log_lik_gp + log_lik_beta
    

# main call
def main(numdraws=1000, numtune=500, numchains = 4, steptype = 'Metropolis', seed=1):

    # make output directory
    outdir = make_run_dir(sampler="pymc", numdraws=numdraws, numtune=numtune, numchains=numchains, steptype=steptype, seed=seed)

    # load diabetes data from sklearn
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


    gp_likelihood = PenalizedGPLikelihood_pymc(Xtrain, ytrain)
    with pm.Model() as model:

        # priors
        zeta = pm.Exponential("zeta", 1.0, shape=X.shape[1])
        ell = pm.Lognormal("ell", 0.0, 1.0, shape=X.shape[1])
        nu = pm.Normal("nu", 0.0, 1.0, shape=X.shape[1])
        inv_sigma_noise = pm.Gamma("inv_sigma_noise", 1.0, 1.0)
        inv_sigma_gp = pm.Gamma("inv_sigma_gp", 1.0, 1.0)
        lambda2 = pm.Gamma("lambda2", 1.0, 1.78)

        # step type
        if steptype == 'Metropolis':
            step = pm.Metropolis()
        elif steptype == 'NUTS':
            step = pm.NUTS()

        # define likelihood function
        pm.Potential(
            "gp_likelihood",
            gp_likelihood.logProbability(
                inv_sigma_noise,
                inv_sigma_gp,
                lambda2,
                zeta,
                ell,
                nu ) )

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
    posterior_df.to_csv(os.path.join(outdir, 'posterior_summary.csv'), index=False)
    make_trace_plots(outdir, trace, Xtrain)
    


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