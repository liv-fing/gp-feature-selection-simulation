
# mechanism_mega.py

'''
specify with command line args: --mechanism [lasso, original, ols]

new form of predictions
- wrote code, but have to test (takes a very long time)
- add plots as well
- do we want rmse distribution or from posterior mean?


updated how results save (file structure):
results/
    mechanism_name/ (lasso, orig, ols)
        date_data/ (ex. 06-15_diabetes)
            step_type/ (Metropolis, NUTS)
                lambda_val/ (if testing lambda, ex. lbda5.0)
                    dr{numdraws}_t{numtune}_ch{numchains}/
                        sd{seed}/



to run:
cd Desktop/GitHub/IEMS399-GP/phase3-refining
conda activate venv
python machinery/mechanism_mega.py 
--test_lambda True 
--fixed_lambda_val 1.0 
--numtune 0 
--numdraws 10000 
--numchains 4 
--data diabetes
--mechanism orig
--predict True


Command line structure (with defaults): 
python machinery/mechanism_mega.py
--test_lambda True
--fixed_lambda_val 1.0
--numtune 0
--numdraws 10000
--numchains 4
--data diabetes
--mechanism orig
--predict True




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
def make_run_dir(sampler="pymc", 
                 mechanism="orig", 
                 numdraws=2000, 
                 numtune=1000, 
                 numchains = 6, 
                 steptype = 'Metropolis', 
                 seed=1, 
                 test_lambda=False, 
                 fixed_lambda_val=0, 
                 data = 'diabetes'):
    if test_lambda:
        day_month = datetime.now().strftime("%d-%m")
        path =  f"results/{mechanism}/{day_month}_{data}/{steptype}/lbda{fixed_lambda_val}/dr{numdraws}_t{numtune}_ch{numchains}/sd{seed}"
        
    else:
        path =  f"results/{mechanism}/{day_month}_{data}/{steptype}/dr{numdraws}_t{numtune}_ch{numchains}/sd{seed}"
    os.makedirs(path, exist_ok=True)
    metadata = { # create metadata json file
        "data": data,
        "mechanism": mechanism,
        "sampler": sampler,
        "draws": numdraws,
        "tune": numtune,
        "chains": numchains,
        "seed": seed,
        "step": steptype,
        "data": data,
        "datetime": datetime.now().isoformat() # capture run time
        }
    
    if test_lambda:
        metadata["test_lambda"] = test_lambda
        metadata["fixed_lambda_val"] = fixed_lambda_val

    with open(f"{path}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return path

# make clean dataframe
def process_pymc_results(outdir, trace, X):
    '''
    takes the pymc trace and processes it into a clean dataframe with one row per sample and columns for each parameter
    
    '''
    num_features = X.shape[1]

    posterior_ds = trace.posterior.stack(sample=("chain", "draw"))

    clean_df = pd.DataFrame()

    clean_df['sigma2_gp'] = (posterior_ds['sigma2_gp']).values if 'sigma2_gp' in posterior_ds else None # if lasso
    clean_df['sigma2_noise'] = (posterior_ds['sigma2_noise']).values
    clean_df['lambda'] = np.sqrt(posterior_ds['lambda2'].values)
    
    for i in range(num_features):
        ell_i = posterior_ds['ell'].isel(ell_dim_0=i).values if 'ell' in posterior_ds else None # if lasso
        beta_i = posterior_ds['beta'].isel(beta_dim_0=i).values

        # add to dataframe
        clean_df[f'beta{i}'] = beta_i
        clean_df[f'ell{i}'] = ell_i if 'ell' in posterior_ds else None # is lasso

    # --- save to CSV ---
    os.makedirs(outdir, exist_ok=True)
    clean_df.to_csv(os.path.join(outdir, "posterior_summary.csv"), index=False)
    print(f"Posterior summary saved to {outdir}/posterior_summary.csv")

    return clean_df


def make_trace_plots(outdir, trace, X, mechanism):
    '''
    create trace plots for all parameters in the model
    for lasso, only plot beta, sigma2_noise, and lambda
    for original and ols, plot beta, sigma2_noise, sigma2_gp, lambda, and ell
    use the same plotting code, but loop through only parameters that exist based on mechanism type
    '''

    posterior = trace.posterior
    num_features = X.shape[1]

    # choose vars by mechanism
    if mechanism == 'lasso':
        scalars = ["sigma2_noise", "lambda"] # lambda derived from lambda2
        plot_ell = False

    elif mechanism in ('orig','ols'):
        scalars = ["sigma2_noise", "sigma2_gp", "lambda"] # lambda derived from lambda2
        plot_ell = True

    panels = []

    # scalars
    for scalar in scalars: # 
        if scalar == "lambda":
            if "lambda2" in posterior:
                panels.append(("scalar", "lambda")) # add lambda as a scalar panel, even though it's derived from lambda2
        else:
            if scalar in posterior:
                panels.append(("scalar", scalar))


    # betas
    if "beta" in posterior:
        for j in range(num_features):
            panels.append((f'beta', j))

    # ells
    if plot_ell and "ell" in posterior:
        for j in range(num_features):
            panels.append((f'ell', j))
    
    # total axes is length of panels
    total_axes = len(panels)
    if total_axes == 0:
        print("Nothing to plot.")
        return
    fig, axes = plt.subplots(total_axes, 1, figsize=(10, 3 * total_axes))
    if total_axes == 1: # if only one plot, axes is not a list, so make it a list
        axes = [axes]

    # plot
    for i, (kind, key) in enumerate(panels): # loop through panels
        ax = axes[i]

        if kind == "scalar":
            if key == "lambda":
                val = np.sqrt(posterior['lambda2'])
            else:
                val = posterior[key]

            for chain in range(val.sizes['chain']):
                ax.plot(val.isel(chain=chain).values, alpha=0.5)
            ax.set_title(f'Trace plot for {key}')
            ax.set_ylabel(key)

        elif kind == "beta":
            j = key
            beta_j = posterior['beta'].isel(beta_dim_0=j) # beta for feature j
            for chain in range(beta_j.sizes['chain']):
                ax.plot(beta_j.isel(chain=chain).values, alpha=0.4)
            ax.set_title(f'Trace plot for beta{j}')
            ax.set_ylabel(f'beta{j}')

        elif kind == "ell":
            j = key
            ell_j = posterior['ell'].isel(ell_dim_0=j)
            for chain in range(ell_j.sizes['chain']):
                ax.plot(ell_j.isel(chain=chain).values, alpha=0.4)
            ax.set_title(f'Trace plot for ell{j}')
            ax.set_ylabel(f'ell{j}')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'trace_plots.png'))
    print(f"Trace plots saved to {outdir}/trace_plots.png")
    plt.close()


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

    Xtrain = Xtrain.to_numpy()
    Xtest  = Xtest.to_numpy()
    ytrain = np.asarray(ytrain)
    ytest  = np.asarray(ytest)  

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


def predictions_lasso(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    '''
    if mechanism == "lasso"
    predictions are just X @ beta, no gp component
    '''
    # make directory for predictions
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # get posterior samples
    posterior = trace.posterior
    beta = posterior["beta"].values # shape (chains, draws, features)
    beta_flat = beta.reshape(-1, beta.shape[-1]) # shape (samples, features)

    samples = beta_flat.shape[0]

    ytrain_preds = beta_flat @ Xtrain.T # shape (samples, n_train) 
    ytest_preds = beta_flat @ Xtest.T # shape (samples, n_test)

    # mean predictions per observation
    ytrain_pred_mean = ytrain_preds.mean(axis=0) # shape (n_train,)
    ytest_pred_mean = ytest_preds.mean(axis=0) # shape (n_test,)

    # calculate overall rmse
    train_rmse = root_mean_squared_error(ytrain, ytrain_pred_mean)
    test_rmse = root_mean_squared_error(ytest, ytest_pred_mean)

    print(f'Predictions and RMSEs calculated for {samples} samples.')

    # --- outputs ---

    # print(f'shape of ytrain_preds: {ytrain_preds.shape}')
    # print(f'shape of ytrain: {ytrain.shape}')
    # print(f'shape of ytest_preds: {ytest_preds.shape}')
    # print(f'shape of ytest: {ytest.shape}')
    # print(f'shape of train_rmse: {train_rmse.shape if isinstance(train_rmse, np.ndarray) else "scalar"}')
    # print(f'shape of test_rmse: {test_rmse.shape if isinstance(test_rmse, np.ndarray) else "scalar"}')
          

    # lasso, so just linear regression predictions, no credible intervals
    predictions_df_train = pd.DataFrame({
        "split": ["train"] * len(ytrain),
        "y_true": ytrain,
        "y_pred": ytrain_pred_mean,
        "rmse": [train_rmse] * len(ytrain) # same rmse for all rows, since no credible intervals}
    })

    predictions_df_test = pd.DataFrame({
        "split": ["test"] * len(ytest),
        "y_true": ytest,
        "y_pred": ytest_pred_mean,
        "rmse": [test_rmse] * len(ytest) # same rmse for all rows, since no credible intervals
    })
        
    df_summary = pd.concat([predictions_df_train, predictions_df_test], ignore_index=True)
    df_summary.to_csv(run_dir / "prediction_summary.csv", index=False)

    return { "prediction_summary_path": run_dir / "prediction_summary.csv"}


def predictions(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    '''
    for each draw for (beta, ell, sigma2_gp, sigma2_noise) 
    calculate predictions for train and test
    using y_pred = X * beta + f_pred
    each draw is its own prediction, so we get a distribution of predictions
    also calculate rmse for each draw
    '''

    print('Calculating predictions for each posterior sample...')

    # make directory for predictions
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


    # get posterior samples
    posterior = trace.posterior
    beta = posterior["beta"].values # shape (chains, draws, features)
    ell = posterior["ell"].values # shape (chains, draws, features)
    sigma2_gp = posterior["sigma2_gp"].values # shape (chains, draws)
    sigma2_noise = posterior["sigma2_noise"].values # shape (chains, draws)

    n_train = len(Xtrain)
    n_test = len(Xtest)
    I = np.eye(n_train)

    chains, draws, features = beta.shape
    samples = chains * draws

    # flatten
    beta_flat = beta.reshape(samples, features) # shape (samples, features)
    ell_flat = ell.reshape(samples, ell.shape[-1]) # shape (samples, features)
    sigma2_gp_flat = sigma2_gp.reshape(samples) # shape (samples,)
    sigma2_noise_flat = sigma2_noise.reshape(samples) # shape (samples,)

    # # check shapes
    # print(f"beta shape: {beta_flat.shape}")
    # print(f"ell shape: {ell_flat.shape}")
    # print(f"sigma2_gp shape: {sigma2_gp_flat.shape}")
    # print(f"sigma2_noise shape: {sigma2_noise_flat.shape}")

    # store y = X * beta + f_pred
    ytrain_preds = np.zeros((samples, n_train)) 
    ytest_preds = np.zeros((samples, n_test))

    # store f_pred (just gp component)
    ftrain_preds = np.zeros((samples, n_train)) 
    ftest_preds = np.zeros((samples, n_test)) 

    # store X * beta (just linear component)
    Xbeta_train_preds = np.zeros((samples, n_train))
    Xbeta_test_preds = np.zeros((samples, n_test))

    # store rmse for each sample
    train_rmses = np.zeros(samples)
    test_rmses = np.zeros(samples)

    # loop through each sample and make predictions
    for s in range(samples):
        beta_s = beta_flat[s]
        ell_s = ell_flat[s]
        sigma2_gp_s = float(sigma2_gp_flat[s])
        sigma2_noise_s = float(sigma2_noise_flat[s])

        residuals = ytrain - Xtrain @ beta_s

        # K
        Ktrain = rbf_kernel(Xtrain, Xtrain, ell_s, sigma2_gp_s)
        Ktest = rbf_kernel(Xtest, Xtrain, ell_s, sigma2_gp_s)

        # C = K + sigma2_noise * I
        Ctrain = Ktrain + sigma2_noise_s * I
        alpha = np.linalg.solve(Ctrain, residuals)

        # gp predictions
        ftrain = Ktrain @ alpha
        ftest = Ktest @ alpha

        # y_pred = X * beta + f_pred
        ytrain_pred = Xtrain @ beta_s + ftrain
        ytest_pred = Xtest @ beta_s + ftest

        # save 
        ytrain_preds[s] = ytrain_pred
        ytest_preds[s] = ytest_pred
        ftrain_preds[s] = ftrain
        ftest_preds[s] = ftest
        Xbeta_train_preds[s] = Xtrain @ beta_s
        Xbeta_test_preds[s] = Xtest @ beta_s
        train_rmses[s] = root_mean_squared_error(ytrain, ytrain_pred)
        test_rmses[s] = root_mean_squared_error(ytest, ytest_pred)


    # --- outputs ---

    # posterior means
    ytrain_means = ytrain_preds.mean(axis=0) # shape (n_train,)
    ytest_means = ytest_preds.mean(axis=0) # shape (n_test,)

    # 90% credible intervals
    ytrain_p05, ytrain_p95 = np.quantile(ytrain_preds, [0.05, 0.95], axis=0) # shape (n_train,)
    ytest_p05,  ytest_p95  = np.quantile(ytest_preds,  [0.05, 0.95], axis=0) # shape (n_test,)

    # gp predictions
    ftrain_means = ftrain_preds.mean(axis=0) # shape (n_train,)
    ftest_means = ftest_preds.mean(axis=0) # shape (n_test,)

    # linear predictions
    Xbeta_train_means = Xbeta_train_preds.mean(axis=0) # shape (n_train,)
    Xbeta_test_means = Xbeta_test_preds.mean(axis=0) # shape (n_test,)

    # rmse from posterior mean
    train_rmse = root_mean_squared_error(ytrain, ytrain_means) # scalar rmse
    test_rmse = root_mean_squared_error(ytest, ytest_means) # scalar rmse
    print(f'Train RMSE (posterior mean): {train_rmse}')
    print(f'Test RMSE (posterior mean): {test_rmse}')

    predictions_df_train = pd.DataFrame({ # ------- add in individual gp and beta parts ------
        "split": "train",
        "y_true": ytrain,
        "y_pred_mean": ytrain_means,
        "y_pred_p05": ytrain_p05,
        "y_pred_p95": ytrain_p95,
        "f_pred_mean": ftrain_means,
        "Xbeta_pred_mean": Xbeta_train_means

    })

    predictions_df_test = pd.DataFrame({
        "split": "test",
        "y_true": ytest,
        "y_pred_mean": ytest_means,
        "y_pred_p05": ytest_p05,
        "y_pred_p95": ytest_p95,
        "f_pred_mean": ftest_means,
        "Xbeta_pred_mean": Xbeta_test_means
    })

    df_summary = pd.concat([predictions_df_train, predictions_df_test], ignore_index=True)
    df_summary.to_csv(run_dir / "prediction_summary.csv", index=False)

    return {"prediction_summary_path": run_dir / "prediction_summary.csv"}


def plot_predictions(prediction_summary_path):
    '''
    1. plot predicted vs true values with 90% credible intervals
    2. plot gp prediction and linear component vs true values 


    '''
    prediction_summary_path = Path(prediction_summary_path)
    df = pd.read_csv(prediction_summary_path)
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']

    # plot 1: true vs predicted values with 90% credible intervals
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train['y_true'], train['y_pred_mean'], alpha=0.5, label='Mean Prediction')
    plt.scatter(train['y_true'], train['y_pred_p05'], alpha=0.5, label='5th Percentile', color='orange')
    plt.scatter(train['y_true'], train['y_pred_p95'], alpha=0.5, label='95th Percentile', color='green')
    plt.plot([train['y_true'].min(), train['y_true'].max()], [train['y_true'].min(), train['y_true'].max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Train Set')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(test['y_true'], test['y_pred_mean'], alpha=0.5, label='Mean Prediction')
    plt.scatter(test['y_true'], test['y_pred_p05'], alpha=0.5, label='5th Percentile', color='orange')
    plt.scatter(test['y_true'], test['y_pred_p95'], alpha=0.5, label='95th Percentile', color='green')
    plt.plot([test['y_true'].min(), test['y_true'].max()], [test['y_true'].min(), test['y_true'].max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Set')
    plt.legend()
    plt.tight_layout()
    plt.suptitle('True vs Predicted Values with 90% Credible Intervals', fontsize=14) 
    plt.subplots_adjust(top=0.88) # adjust the top of the plots to make room for the title
    plt.savefig(prediction_summary_path.parent / "predicted_vs_true.png") # save plot to file

    # plot 2: gp prediction and linear component vs true values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train['y_true'], train['f_pred_mean'], alpha=0.5, label='GP Prediction')
    plt.scatter(train['y_true'], train['Xbeta_pred_mean'], alpha=0.5, label='Linear Prediction', color='green')
    plt.plot([train['y_true'].min(), train['y_true'].max()], [train['y_true'].min(), train['y_true'].max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Train Set')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(test['y_true'], test['f_pred_mean'], alpha=0.5, label='GP Prediction')
    plt.scatter(test['y_true'], test['Xbeta_pred_mean'], alpha=0.5, label='Linear Prediction', color='green')
    plt.plot([test['y_true'].min(), test['y_true'].max()], [test['y_true'].min(), test['y_true'].max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Set')
    plt.legend()
    plt.tight_layout()
    plt.suptitle('GP vs Linear Components of Prediction for Train and Test', fontsize=14) 
    plt.subplots_adjust(top=0.88) # adjust the top of the plots to make room for the title
    plt.savefig(prediction_summary_path.parent / "gp_vs_linear.png") # save plot to file

    print(f"\nPrediction plots saved to {prediction_summary_path.parent}/predicted_vs_true.png and {prediction_summary_path.parent}/gp_vs_linear.png")



def starting_points(Xtrain, ytrain):
    '''
    set starting points for sampler using OLS estimates for beta
    '''
    start = {}

    # OLS estimates for beta
    XTX_inv = np.linalg.inv(Xtrain.T @ Xtrain)
    beta_ols = XTX_inv @ Xtrain.T @ ytrain
    start['beta'] = beta_ols

    #MLE estimate for sigma2_noise
    residuals = ytrain - Xtrain @ beta_ols
    sigma2_noise_mle = np.var(residuals, ddof=1) 
    start['sigma2_noise'] = sigma2_noise_mle

    return start


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
def main(numdraws=1000, 
         numtune=500, 
         numchains = 4, 
         steptype = 'Metropolis', 
         seed=1, 
         test_lambda=True, 
         fixed_lambda_val=5, 
         data = 'diabetes', 
         mechanism = 'orig', 
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


        if mechanism == 'lasso': # lasso penalty only, no gp

            # priors for \sigma^2_noise, tau^2, beta
            sigma2_noise = pm.HalfNormal("sigma2_noise", sigma=1)
            tau2 = pm.Exponential("tau2", lambda2/2.0, shape=Xtrain.shape[1]) # depends on lambda2
            tau = pm.math.sqrt(tau2)
            beta = pm.Normal("beta", 0.0, pm.math.sqrt(sigma2_noise * tau), shape=Xtrain.shape[1])  # depends on tau, sigma2_noise

            # standard likelihood
            y_obs = pm.Normal(
                "y_obs", 
                mu=pt.dot(Xtrain, beta), 
                sigma=pt.sqrt(sigma2_noise), 
                observed=ytrain)
            
            # step type
            if steptype == 'Metropolis':
                step = pm.Metropolis()
            elif steptype == 'NUTS':
                step = pm.NUTS()

            # run model
            trace = pm.sample(
            initvals=start_points if mechanism == 'ols' else None,
            draws=numdraws,
            tune=numtune,
            chains=numchains,
            step=step,
            random_seed=seed,
            progressbar=True
            )
            

        elif mechanism == 'orig': # original model

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
            initvals=start_points if mechanism == 'ols' else None,
            draws=numdraws,
            tune=numtune,
            chains=numchains,
            step=step,
            random_seed=seed,
            progressbar=True
            )
            
        elif mechanism == 'ols': # gp with ols initialization
                
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
            
            start_points = starting_points(Xtrain, ytrain)

            # run model
            trace = pm.sample(
            initvals=start_points if mechanism == 'ols' else None,
            draws=numdraws,
            tune=numtune,
            chains=numchains,
            step=step,
            random_seed=seed,
            progressbar=True
            )

        # if not lasso, orig, or ols, raise error
        else:
            raise ValueError(f"Invalid mechanism: {mechanism}. Choose from 'lasso', 'orig', 'ols'.")
        

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
    parser.add_argument('--mechanism', type=str, default='orig', help = 'Choose mechanism: orig, lasso, ols')
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
    
