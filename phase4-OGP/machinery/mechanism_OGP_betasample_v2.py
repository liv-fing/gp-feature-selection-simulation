


'''

trying to fix





Just OGP version of original GP implementation
This one samples beta

to run: 
cd Desktop/GitHub/IEMS399-GP/phase4-OGP
conda activate venv
python machinery/mechanism_OGP_betasample_v2.py \
--test_lambda True \
--fixed_lambda_val 1.0 \
--numtune 0 \
--numdraws 1000 \
--numchains 4 \
--data diabetes \
--mechanism ogp \
--predict 0 
'''


# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
from pytensor.graph.op import Op
import os
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cho_factor, cho_solve

# MY IMPORTS
from helper_funcs.data_setup import make_run_dir, diabetes_data_init, synthetic_data_init, starting_points
from helper_funcs.result_processing import process_pymc_results, make_trace_plots
from helper_funcs.predicting import predictions_lasso, predictions, plot_predictions, rbf_kernel
from helper_funcs.make_c_star import make_c_star_matrix

# ORTHOGONAL GP LIKELIHOOD (PYTENSOR OP VERSION)
class OGPLikelihood_OP(Op):
    itypes = [pt.dscalar, # sigma2_noise
              pt.dscalar, # sigma2_gp
              pt.dvector, # beta
              pt.dvector] # ell
    
    otypes = [pt.dscalar] # log likelihood value

    def __init__(self, X, y, terms):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = X.shape
        self.terms = terms
        self.counter = 0 # for debugging

    def perform(self, node, inputs, outputs):
        sigma2_noise, sigma2_gp, beta, ell = inputs
        residuals = self.y - self.X @ beta


        self.counter += 1
        print(
                "perform called",
                self.counter,
                "sigma2_noise=", sigma2_noise,
                "sigma2_gp=", sigma2_gp,
                "beta0=", beta[0],
                "ell0=", ell[0],
        )

        # compute C = C_star + \sigma2_noise * I
        C = make_c_star_matrix(self.X, self.X, psi=ell, sigma2=sigma2_gp, terms=self.terms)
        C = C + sigma2_noise * np.eye(self.n) # C + \sigma2 * I

        try:
            L, lower = cho_factor(C, lower=True, check_finite=False)
            logdetC = 2.0 * np.sum(np.log(np.diag(L)))

            Cinv_resid = cho_solve((L, lower), residuals, check_finite=False)
            qf = residuals.T @ Cinv_resid # quadratic form

            log_lik = -0.5 * (
                logdetC
                + qf
                + self.n * np.log(2.0 * np.pi)
            )

        except Exception:
            log_lik = -np.inf

        outputs[0][0] = np.asarray(log_lik, dtype=np.float64)       



# MAIN
def main(numdraws=1000, 
         numtune=500, 
         numchains = 4, 
         steptype = 'Metropolis', 
         seed=1, 
         test_lambda=True, 
         fixed_lambda_val=5, 
         data = 'diabetes', 
         mechanism = 'ogp', 
         predict = False):

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
    elif data == 'synthetic':
        Xtrain, Xtest, ytrain, ytest = synthetic_data_init(size = 1000, active_proportion = 10, noise = 0.1, seed = 0, rep = 1, features = 'all')
        
    with pm.Model() as model:

         # lambda (deterministic or sampled)
        if test_lambda:
            fixed_lambda2_val = fixed_lambda_val **2
            lambda2 = pm.Deterministic("lambda2", pm.math.constant(fixed_lambda2_val)) 
        else:
            lambda2 = pm.Gamma("lambda2", 1.0, 1.78) # hyperparameters from bayesian lasso

        # OGP + ORIGINAL GP 
        if mechanism == 'ogp': 

            # priors for \sigma^2_noise, \sigma^2_gp, ell
            sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
            sigma2_gp = pm.InverseGamma("sigma2_gp", 3.0, 1.0)
            tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
            # tau = pm.math.sqrt(tau2)
            beta = pm.Normal("beta", 0.0, pm.math.sqrt(sigma2_noise * tau2), shape=Xtrain.shape[1])
            ell = pm.Lognormal("ell", mu=2, sigma=0.5, shape=Xtrain.shape[1]) 

            # likelihood function with custom GPLikelihood class
            terms = [None] + list(range(1, Xtrain.shape[1] + 1))
            ogp_likelihood = OGPLikelihood_OP(Xtrain, ytrain, terms) 

            # step type (only use Metropolis now)
            step = pm.Metropolis()
       
             # likelihood function 
            pm.Potential(
                "ogp_likelihood",
                ogp_likelihood(
                    sigma2_noise,
                    sigma2_gp,
                    beta,
                    ell) )   
            
            # run model
            trace = pm.sample(
            initvals=None,
            draws=numdraws,
            tune=numtune,
            chains=numchains,
            step=step,
            random_seed=seed,
            progressbar=True
            )
            
        
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}.")
        

    # save results
    posterior_df = process_pymc_results(outdir, trace, Xtrain)
    posterior_df.to_csv(os.path.join(outdir, 'posterior_results.csv'), index=False)

    # trace plots
    make_trace_plots(outdir, trace, Xtrain, mechanism)

    # summary
    summary = pm.summary(trace, hdi_prob=0.95)
    print('\n95% Credible Interval\n', summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))

    # plot predictions

    if predict:
        if mechanism == 'lasso':
            predictions_lasso(trace, Xtrain, ytrain, Xtest, ytest, outdir)
            print("Predictions calculated and saved, but no plots generated for lasso mechanism yet")

        if mechanism == 'ogp':
            predictions_ogp(trace, Xtrain, ytrain, Xtest, ytest, outdir)
            prediction_summary_path = os.path.join(outdir, 'prediction_summary.csv')
            plot_predictions(prediction_summary_path)

        else:
            predictions(trace, Xtrain, ytrain, Xtest, ytest, outdir)
            prediction_summary_path = os.path.join(outdir, 'prediction_summary.csv')
            plot_predictions(prediction_summary_path)
        

# ARG PARSE
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
    parser.add_argument('--mechanism', type=str, default='ogp', help = 'Choose mechanism: orig, lasso, ols, ogp')
    parser.add_argument('--predict', type=int, default=0, help = 'Whether to calculate predictions and rmse from posterior samples (1 true / 0 false)')
    args = parser.parse_args()

    main(
        numdraws = args.numdraws,
        numtune = args.numtune,
        numchains = args.numchains,
        seed = args.seed,
        steptype = args.step,
        test_lambda = args.test_lambda,
        fixed_lambda_val = args.fixed_lambda_val,
        data = args.data,
        mechanism = args.mechanism,
        predict = args.predict
        )