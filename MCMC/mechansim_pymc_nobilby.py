
"""
mechansim_pymc_nobilby.py


to run:
conda activate pymc310
python mechansim_pymc_nobilby.py

"""

# --- imports ---
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- load and preprocess data ---
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

# --- custom log likelihood class ---
class PenalizedGPLikelihood_pymc:
    def __init__(self, X, y):
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n, self.p = X.shape

    def rbf_kernel(self, X, ell, sigma_gp):
        """ compute RBF kernel using pytensor """
        X_scaled = X / ell
        diff = X_scaled[:, None, :] - X_scaled[None, :, :]
        squared_dist = pt.sum(diff ** 2, axis=2)
        K = sigma_gp**2 * pt.exp(-0.5 * squared_dist)
        return K

    def logProbability(self, inv_sigma_noise, inv_sigma_gp, lambda2, zeta, ell, nu):
        """ compute custom log likelihood using pytensor """
        # transform
        sigma_noise = 1 / inv_sigma_noise
        sigma_gp = 1 / inv_sigma_gp
        tau_sq = 2 * zeta / lambda2
        beta = nu * pt.sqrt(tau_sq) * sigma_noise

        C = self.rbf_kernel(self.X, ell, sigma_gp) 
        residuals = self.y - pt.dot(self.X, beta)
        D = pt.diag(tau_sq)
        sigma2_D = sigma_noise**2 * D

        log_lik_gp = (
            -0.5 * self.n * pt.log(2 * pt.pi)
            -0.5 * pt.nlinalg.slogdet(C + sigma_noise**2 * pt.eye(self.n))[1]
            -0.5 * pt.dot(residuals, pt.slinalg.solve(C + sigma_noise**2 * pt.eye(self.n), residuals))
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



# --- build and sample from model ---

gp_likelihood = PenalizedGPLikelihood_pymc(Xtrain, ytrain)
with pm.Model() as model:

    # priors
    zeta = pm.Exponential("zeta", 1.0, shape=X.shape[1])
    ell = pm.Lognormal("ell", 0.0, 1.0, shape=X.shape[1])
    nu = pm.Normal("nu", 0.0, 1.0, shape=X.shape[1])
    inv_sigma_noise = pm.Gamma("inv_sigma_noise", 1.0, 1.0)
    inv_sigma_gp = pm.Gamma("inv_sigma_gp", 1.0, 1.0)
    lambda2 = pm.Gamma("lambda2", 1.0, 1.78)

    # custom log likelihood
    pm.Potential(
        "gp_likelihood",
        gp_likelihood.logProbability(inv_sigma_noise, inv_sigma_gp, lambda2, zeta, ell, nu)
    )

    trace = pm.sample(
        draws = 1000, 
        tune = 500, 
        chains = 4, 
        step = pm.Metropolis())
    
# --- analyze results ---
az.plot_trace(trace)
plt.savefig("trace_plot.png")