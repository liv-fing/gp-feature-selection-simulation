


'''
Just OGP version of original GP implementation
'''


# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import pytensor.tensor as pt
import argparse
import os
from sklearn.preprocessing import StandardScaler

# MY IMPORTS
from helper_funcs.data_setup import make_run_dir, diabetes_data_init, synthetic_data_init, starting_points
from helper_funcs.result_processing import process_pymc_results, make_trace_plots
from helper_funcs.predicting import predictions_lasso, predictions, plot_predictions, rbf_kernel


# GP LIKELIHOOD
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
    
# ORTHOGONAL GP LIKELIHOOD
class OGPLikelihood_pymc:
    def __init__(self, X, y, terms):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = X.shape
        self.terms = terms

    def rbf_kernel(self, X, ell, sigma2_gp):
        """ 
        compute RBF kernel using pytensor 
        """
        X_scaled = X / ell
        diff = X_scaled[:, None, :] - X_scaled[None, :, :]
        squared_dist = pt.sum(diff ** 2, axis=2)
        K = sigma2_gp * pt.exp(-0.5 * squared_dist)
        return K # returns K, not C = K + \sigma^2_GP * I
    
    def make_c_star_matrix(self, X, psi)
    
    def compute_c_star(self, ell, sigma2_gp):
        psi = np.array(ell) # convert from pt to np
        C = make_c_star_matrix(self.X, self.X, psi=psi, sigma2=float(sigma2_gp), terms=self.terms)
        return pt.as_tensor_variable(C)


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
    '''
    options for 
    '''

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

            # priors for \sigma^2_noise, \sigma^2_gp, tau^2, beta, ell
            sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
            sigma2_gp = pm.InverseGamma("sigma2_gp", 3.0, 1.0)
            tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
            tau = pm.math.sqrt(tau2)
            beta = pm.Normal("beta", 0.0, pm.math.sqrt(sigma2_noise * tau), shape=Xtrain.shape[1])  # depends on tau, sigma2_noise
            ell = pm.Lognormal("ell", mu=2, sigma=0.5, shape=Xtrain.shape[1]) 

            # likelihood function with custom GPLikelihood class
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
    make_trace_plots(outdir, trace, Xtrain, mechanism)
    
    summary = pm.summary(trace, hdi_prob=0.95)  # 95% credible interval
    print('95% Credible Interval\n', summary)
    summary.to_csv(os.path.join(outdir, 'posterior_summary.csv'))

    if predict:
        # make predictions and save
        if mechanism == 'lasso':
            predictions_lasso(trace, Xtrain, ytrain, Xtest, ytest, outdir)
        else:
            predictions(trace, Xtrain, ytrain, Xtest, ytest, outdir)

    # plot predictions
    if predict and mechanism != 'lasso': 
        prediction_summary_path = os.path.join(outdir, 'prediction_summary.csv')
        plot_predictions(prediction_summary_path)

    if predict and mechanism == 'lasso':
        print("Predictions calculated and saved, but no plots generated for lasso mechanism yet")




# ARG PARSE
if __name__ == "__main__":
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
    parser.add_argument('--predict', type=bool, default=False, help = 'Whether to calculate predictions and rmse from posterior samples')
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