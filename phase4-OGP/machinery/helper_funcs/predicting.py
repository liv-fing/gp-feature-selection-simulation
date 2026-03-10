# prediction
'''
load functions with
from helper_funcs.predicting import
    predictions_lasso
    predictions
    plot_predictions
    rbf_kernel
'''
# IMPORTS
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error


# RBF KERNEL
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

# LASSO PREDICTIONS
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

# GP PREDICTIONS
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