

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import root_mean_squared_error

from helper_funcs.predicting import predictions_lasso, predictions_ogp_betanosample, predictions, plot_predictions, rbf_kernel


def ZERO_process_pymc_results(outdir, trace, X):
    
    num_features = X.shape[1]

    posterior_ds = trace.posterior.stack(sample=("chain", "draw"))

    clean_df = pd.DataFrame()

    clean_df['sigma2_noise'] = (posterior_ds['sigma2_noise']).values
    clean_df['sigma2_gp'] = (posterior_ds['sigma2_gp']).values
    # clean_df['lambda'] = np.sqrt(posterior_ds['lambda2'].values)

    for i in range(num_features):
        ell_i = posterior_ds['ell'].isel(ell_dim_0=i).values
        # beta_i = posterior_ds['beta'].isel(beta_dim_0=i).values

        # add to dataframe
        # clean_df[f'beta{i}'] = beta_i
        clean_df[f'ell{i}'] = ell_i

    # --- save to CSV ---
    os.makedirs(outdir, exist_ok=True)
    clean_df.to_csv(os.path.join(outdir, "posterior_summary.csv"), index=False)
    print(f"Posterior summary saved to {outdir}/posterior_summary.csv")

    return clean_df



def ZERO_make_trace_plots(outdir, trace, X):
    num_features = X.shape[1]

    # plot: beta, sigma2_noise, sigma2_gp, lambda, ell
    posterior = trace.posterior

    total_axes = 2 + num_features + num_features  # scalars + betas + ells
    fig, axes = plt.subplots(total_axes, 1, figsize=(10, 3 * total_axes))
    
    if num_features + 4 == 1:
        axes = [axes]

    scalars = {
        'sigma2_noise': posterior['sigma2_noise'],
        'sigma2_gp': posterior['sigma2_gp'],
    }

    for i, (name, val) in enumerate(scalars.items()):
        ax = axes[i]
        for chain in range(val.sizes['chain']):
            ax.plot(val.isel(chain=chain).values, alpha=0.5)
        ax.set_title(f'Trace plot for {name}')
        ax.set_ylabel(name)

    # ells
    for i in range(num_features):
        ax = axes[i + 2 + num_features]  # offset after scalars + betas
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


def ZERO_predict_and_save(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    ''' 
    make predictions
    uses posterior means to make predictions
    computes rmse
    saves results to file

    '''

    # using mean to test
    # beta_hat = trace.posterior["beta"].mean(dim=("chain", "draw")).values
    ell_hat = trace.posterior["ell"].mean(dim=("chain", "draw")).values
    sigma2_gp_hat = trace.posterior["sigma2_gp"].mean(dim=("chain", "draw")).values
    sigma2_noise_hat = trace.posterior["sigma2_noise"].mean(dim=("chain", "draw")).values

    # extract residuals
    residuals = ytrain #- Xtrain @ beta_hat

    # make covariance matrices
    Ktrain = rbf_kernel(Xtrain, Xtrain, ell_hat, sigma2_gp_hat)
    Ktest = rbf_kernel(Xtest, Xtrain, ell_hat, sigma2_gp_hat)

    Ctrain = Ktrain + sigma2_noise_hat * np.eye(len(Xtrain))
    alpha = np.linalg.solve(Ctrain, residuals)

    # create predictions
    ftrain = Ktrain @ alpha
    ftest = Ktest @ alpha

    ytrain_pred = ftrain
    ytest_pred = ftest

    # ytrain_pred = Xtrain @ beta_hat + ftrain
    # ytest_pred = Xtest @ beta_hat + ftest

    train_rmse = root_mean_squared_error(ytrain, ytrain_pred)
    test_rmse = root_mean_squared_error(ytest, ytest_pred)

    print(f'Train RMSE: {train_rmse}')
    print(f'Test RMSE: {test_rmse}')


    run_dir = Path(run_dir)

    # create a file to save predictions
    df_train = pd.DataFrame({
        "split": "train",
        "y_true": ytrain,
        "y_pred": ytrain_pred
    })
    df_test = pd.DataFrame({
        "split": "test",
        "y_true": ytest,
        "y_pred": ytest_pred
    })
    df_predictions = pd.concat([df_train, df_test], ignore_index=True)
    df_predictions.to_csv(run_dir / "predictions_post_mean.csv", index=False)

    # create a file to save rmse
    df_rmse = pd.DataFrame({
        "split": ["train", "test"],
        "rmse": [train_rmse, test_rmse]
    })
    df_rmse.to_csv(run_dir / "rmse_results.csv", index=False)
