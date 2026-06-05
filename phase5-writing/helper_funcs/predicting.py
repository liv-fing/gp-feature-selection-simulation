# prediction
'''
load functions with
from helper_funcs.predicting import
    predictions_lasso
    predictions_ogp
    predictions
    plot_predictions
    rbf_kernel
'''
# IMPORTS
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error

# MY IMPORTS
from helper_funcs.make_c_star import make_c_star_matrix, make_G, make_beta


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

# OGP PREDICTIONS
def predictions_ogp_betanosample(trace, Xtrain, ytrain, Xtest, ytest, run_dir, betas, terms):
    print('Calculating OGP predictions for each posterior sample...')
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    posterior = trace.posterior
    ell = posterior["ell"].values
    sigma2_gp = posterior["sigma2_gp"].values
    sigma2_noise = posterior["sigma2_noise"].values

    n_train = len(Xtrain)
    n_test = len(Xtest)
    chains, draws = sigma2_gp.shape
    samples = chains * draws

    # flatten
    beta_flat = betas # already (samples, p)
    ell_flat = ell.reshape(samples, ell.shape[-1])
    sigma2_gp_flat = sigma2_gp.reshape(samples)
    sigma2_noise_flat = sigma2_noise.reshape(samples)

    # design matrices (fixed, don't change per sample)
    G_train = make_G({"X_scaled": Xtrain}, terms)
    G_test  = make_G({"X_scaled": Xtest},  terms)

    ytrain_preds = np.zeros((samples, n_train))
    ytest_preds  = np.zeros((samples, n_test))
    ftrain_preds = np.zeros((samples, n_train))
    ftest_preds  = np.zeros((samples, n_test))
    Xbeta_train_preds = np.zeros((samples, n_train))
    Xbeta_test_preds  = np.zeros((samples, n_test))
    train_rmses = np.zeros(samples)
    test_rmses  = np.zeros(samples)

    for s in range(samples):
        beta_s      = beta_flat[s]
        ell_s       = ell_flat[s]
        sigma2_gp_s = float(sigma2_gp_flat[s])
        sigma2_noise_s = float(sigma2_noise_flat[s])

        # residual kernel instead of rbf
        Ctrain_clean = make_c_star_matrix(Xtrain, Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms)
        Ctest_clean  = make_c_star_matrix(Xtest,  Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms) 
        Ctrain_noisy = Ctrain_clean + sigma2_noise_s * np.eye(n_train)

        # residuals use G @ beta instead of X @ beta
        residuals = ytrain - G_train @ beta_s
        alpha = np.linalg.solve(Ctrain_noisy, residuals)

        # gp component
        ftrain = Ctrain_clean @ alpha
        ftest  = Ctest_clean  @ alpha

        # full predictions
        ytrain_pred = G_train @ beta_s + ftrain
        ytest_pred  = G_test  @ beta_s + ftest

        ytrain_preds[s] = ytrain_pred
        ytest_preds[s]  = ytest_pred
        ftrain_preds[s] = ftrain
        ftest_preds[s]  = ftest
        Xbeta_train_preds[s] = G_train @ beta_s
        Xbeta_test_preds[s]  = G_test  @ beta_s
        train_rmses[s] = root_mean_squared_error(ytrain, ytrain_pred)
        test_rmses[s]  = root_mean_squared_error(ytest,  ytest_pred)

    # outputs - same as original predictions function
    ytrain_means = ytrain_preds.mean(axis=0)
    ytest_means  = ytest_preds.mean(axis=0)
    ytrain_p05, ytrain_p95 = np.quantile(ytrain_preds, [0.05, 0.95], axis=0)
    ytest_p05,  ytest_p95  = np.quantile(ytest_preds,  [0.05, 0.95], axis=0)
    ftrain_means = ftrain_preds.mean(axis=0)
    ftest_means  = ftest_preds.mean(axis=0)
    Xbeta_train_means = Xbeta_train_preds.mean(axis=0)
    Xbeta_test_means  = Xbeta_test_preds.mean(axis=0)

    train_rmse = root_mean_squared_error(ytrain, ytrain_means)
    test_rmse  = root_mean_squared_error(ytest,  ytest_means)
    print(f'Train RMSE (posterior mean): {train_rmse}')
    print(f'Test RMSE (posterior mean): {test_rmse}')

    predictions_df_train = pd.DataFrame({
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