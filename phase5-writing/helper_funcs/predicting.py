# prediction
'''
load functions with
from helper_funcs.predicting import
    predictions_lasso
    predictions_ogp_betanosample
    predictions_gp
    plot_predictions
    rbf_kernel
'''
# IMPORTS
from pdb import run
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

# # OGP PREDICTIONS ANALYTICAL BETA 
# def predictions_ogp_betanosample(trace, Xtrain, ytrain, Xtest, ytest, run_dir, betas, terms):
#     """
#     Orthogonal GP predictions and posterior RMSE summaries.

#     Saves:
#     - rmse_summary.csv
#     - prediction_summary.csv needed by plot_predictions()
#     """
    

#     print("Calculating OGP predictions for each posterior sample...")

#     run_dir = Path(run_dir)
#     run_dir.mkdir(parents=True, exist_ok=True)

#     # convert inputs to numpy arrays
#     Xtrain = np.asarray(Xtrain)
#     Xtest = np.asarray(Xtest)
#     ytrain = np.asarray(ytrain)
#     ytest = np.asarray(ytest)

#     posterior = trace.posterior
#     ell = posterior["ell"].values
#     sigma2_gp = posterior["sigma2_gp"].values
#     sigma2_noise = posterior["sigma2_noise"].values

#     n_train = len(Xtrain)
#     n_test = len(Xtest)

#     chains, draws = sigma2_gp.shape
#     n_samples = chains * draws

#     # flatten posterior samples
#     beta_flat = np.asarray(betas)  # already expected to be shape (n_samples, p)
#     ell_flat = ell.reshape(n_samples, ell.shape[-1])
#     sigma2_gp_flat = sigma2_gp.reshape(n_samples)
#     sigma2_noise_flat = sigma2_noise.reshape(n_samples)

#     # optional sanity check
#     if beta_flat.shape[0] != n_samples:
#         raise ValueError(
#             f"betas has {beta_flat.shape[0]} samples, but trace has {n_samples} posterior samples."
#         )

#     # design matrices fixed across posterior samples
#     G_train = make_G({"X_scaled": Xtrain}, terms)
#     G_test = make_G({"X_scaled": Xtest}, terms)

#     # predictions from every posterior sample
#     ytrain_preds = np.zeros((n_samples, n_train))
#     ytest_preds = np.zeros((n_samples, n_test))

#     # save GP and linear components separately
#     ftrain_preds = np.zeros((n_samples, n_train))
#     ftest_preds = np.zeros((n_samples, n_test))
#     Xbeta_train_preds = np.zeros((n_samples, n_train))
#     Xbeta_test_preds = np.zeros((n_samples, n_test))

#     I = np.eye(n_train)

#     for s in range(n_samples):
#         beta_s = beta_flat[s]
#         ell_s = ell_flat[s]
#         sigma2_gp_s = float(sigma2_gp_flat[s])
#         sigma2_noise_s = float(sigma2_noise_flat[s])

#         # residual kernel
#         Ctrain_clean = make_c_star_matrix(
#             Xtrain, Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms
#         )
#         Ctest_clean = make_c_star_matrix(
#             Xtest, Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms
#         )

#         Ctrain_noisy = Ctrain_clean + sigma2_noise_s * I

#         # linear component
#         Gbeta_train = G_train @ beta_s
#         Gbeta_test = G_test @ beta_s

#         # residuals use G @ beta instead of X @ beta
#         residuals = ytrain - Gbeta_train
#         alpha = np.linalg.solve(Ctrain_noisy, residuals)

#         # GP posterior mean component
#         ftrain = Ctrain_clean @ alpha
#         ftest = Ctest_clean @ alpha

#         # full prediction
#         ytrain_pred = Gbeta_train + ftrain
#         ytest_pred = Gbeta_test + ftest

#         # save predictions
#         ytrain_preds[s] = ytrain_pred
#         ytest_preds[s] = ytest_pred

#         ftrain_preds[s] = ftrain
#         ftest_preds[s] = ftest

#         Xbeta_train_preds[s] = Gbeta_train
#         Xbeta_test_preds[s] = Gbeta_test

#     # posterior mean predictions
#     ytrain_pred_mean = ytrain_preds.mean(axis=0)
#     ytest_pred_mean = ytest_preds.mean(axis=0)

#     # RMSE of posterior mean prediction
#     train_rmse_mean_prediction = root_mean_squared_error(ytrain, ytrain_pred_mean)
#     test_rmse_mean_prediction = root_mean_squared_error(ytest, ytest_pred_mean)

#     # one RMSE per posterior sample
#     train_rmse_samples = np.sqrt(
#         np.mean((ytrain_preds - ytrain[None, :]) ** 2, axis=1)
#     )

#     test_rmse_samples = np.sqrt(
#         np.mean((ytest_preds - ytest[None, :]) ** 2, axis=1)
#     )

#     # posterior RMSE summaries
#     train_rmse_post_mean = train_rmse_samples.mean()
#     train_rmse_ci_lower, train_rmse_ci_upper = np.quantile(
#         train_rmse_samples, [0.025, 0.975]
#     )

#     test_rmse_post_mean = test_rmse_samples.mean()
#     test_rmse_ci_lower, test_rmse_ci_upper = np.quantile(
#         test_rmse_samples, [0.025, 0.975]
#     )

#     rmse_summary = pd.DataFrame({
#         "train": [
#             n_samples,
#             train_rmse_mean_prediction,
#             train_rmse_post_mean,
#             train_rmse_ci_lower,
#             train_rmse_ci_upper,
#         ],
#         "test": [
#             n_samples,
#             test_rmse_mean_prediction,
#             test_rmse_post_mean,
#             test_rmse_ci_lower,
#             test_rmse_ci_upper,
#         ],
#     }, index=[
#         "n_posterior_samples",
#         "rmse_of_posterior_mean_prediction",
#         "posterior_rmse_mean",
#         "posterior_rmse_ci_lower",
#         "posterior_rmse_ci_upper",
#     ])

#     rmse_summary_path = run_dir / "rmse_summary.csv"
#     rmse_summary.to_csv(rmse_summary_path, index=True)

#     # prediction summary for plot_predictions()
#     train_prediction_summary = pd.DataFrame({
#         "split": "train",
#         "y_true": ytrain,
#         "y_pred_mean": ytrain_pred_mean,
#         "y_pred_p05": np.quantile(ytrain_preds, 0.05, axis=0),
#         "y_pred_p95": np.quantile(ytrain_preds, 0.95, axis=0),
#         "f_pred_mean": ftrain_preds.mean(axis=0),
#         "Xbeta_pred_mean": Xbeta_train_preds.mean(axis=0),
#     })

#     test_prediction_summary = pd.DataFrame({
#         "split": "test",
#         "y_true": ytest,
#         "y_pred_mean": ytest_pred_mean,
#         "y_pred_p05": np.quantile(ytest_preds, 0.05, axis=0),
#         "y_pred_p95": np.quantile(ytest_preds, 0.95, axis=0),
#         "f_pred_mean": ftest_preds.mean(axis=0),
#         "Xbeta_pred_mean": Xbeta_test_preds.mean(axis=0),
#     })

#     prediction_summary = pd.concat(
#         [train_prediction_summary, test_prediction_summary],
#         ignore_index=True
#     )

#     prediction_summary_path = run_dir / "prediction_summary.csv"
#     prediction_summary.to_csv(prediction_summary_path, index=False)

#     print(f"Predictions and RMSEs calculated for {n_samples} posterior samples.")
#     print(f"Train RMSE of posterior mean prediction: {train_rmse_mean_prediction}")
#     print(f"Test RMSE of posterior mean prediction: {test_rmse_mean_prediction}")
#     print(f"RMSE summary saved to {rmse_summary_path}")
#     print(f"Prediction summary saved to {prediction_summary_path}")

#     return {
#         "rmse_summary_path": rmse_summary_path,
#         "prediction_summary_path": prediction_summary_path,
#     }





# OGP PREDICTIONS ANALYTICAL BETA 
def predictions_ogp_betanosample(trace, Xtrain, ytrain, Xtest, ytest, run_dir, betas=None, terms=None, mechanism='ogp'):
    """
    Orthogonal GP predictions and posterior RMSE summaries.

    Saves:
    - rmse_summary.csv
    - prediction_summary.csv needed by plot_predictions()
    """
    

    print("Calculating OGP predictions for each posterior sample...")

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # convert inputs to numpy arrays
    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain)
    ytest = np.asarray(ytest)

    n_train = len(Xtrain)
    n_test = len(Xtest)

    posterior = trace.posterior
    ell = posterior["ell"].values
    sigma2_gp = posterior["sigma2_gp"].values
    sigma2_noise = posterior["sigma2_noise"].values

    chains, draws = sigma2_gp.shape
    n_samples = chains * draws

    if mechanism == 'ogp_sample':
        print("Using sampled beta from trace for predictions.")
        beta = posterior["beta"].values
        beta_flat = beta.reshape(n_samples, beta.shape[-1])  # shape: (n_samples, n_terms)
    elif mechanism == 'ogp':
        if betas is None:
            raise ValueError("betas must be provided when mechanism='ogp'")
        print("Using analytically computed beta for predictions.")
        beta_flat = np.asarray(betas)  # shape: (n_samples, n_terms)

    # flatten posterior samples
    ell_flat = ell.reshape(n_samples, ell.shape[-1])
    sigma2_gp_flat = sigma2_gp.reshape(n_samples)
    sigma2_noise_flat = sigma2_noise.reshape(n_samples)

    # optional check
    if beta_flat.shape[0] != n_samples:
        raise ValueError(
            f"betas has {beta_flat.shape[0]} samples, but trace has {n_samples} posterior samples."
        )

    # design matrices fixed across posterior samples
    G_train = make_G({"X_scaled": Xtrain}, terms)
    G_test = make_G({"X_scaled": Xtest}, terms)

    # predictions from every posterior sample
    ytrain_preds = np.zeros((n_samples, n_train))
    ytest_preds = np.zeros((n_samples, n_test))

    # save GP and linear components separately
    ftrain_preds = np.zeros((n_samples, n_train))
    ftest_preds = np.zeros((n_samples, n_test))
    Xbeta_train_preds = np.zeros((n_samples, n_train))
    Xbeta_test_preds = np.zeros((n_samples, n_test))

    I = np.eye(n_train)

    for s in range(n_samples):
        beta_s = beta_flat[s]
        ell_s = ell_flat[s]
        sigma2_gp_s = float(sigma2_gp_flat[s])
        sigma2_noise_s = float(sigma2_noise_flat[s])

        # residual kernel
        Ctrain_clean = make_c_star_matrix(
            Xtrain, Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms
        )
        Ctest_clean = make_c_star_matrix(
            Xtest, Xtrain, psi=ell_s, sigma2=sigma2_gp_s, terms=terms
        )

        Ctrain_noisy = Ctrain_clean + sigma2_noise_s * I

        # linear component
        Gbeta_train = G_train @ beta_s
        Gbeta_test = G_test @ beta_s

        # residuals use G @ beta instead of X @ beta
        residuals = ytrain - Gbeta_train
        alpha = np.linalg.solve(Ctrain_noisy, residuals)

        # GP posterior mean component
        ftrain = Ctrain_clean @ alpha
        ftest = Ctest_clean @ alpha

        # full prediction
        ytrain_pred = Gbeta_train + ftrain
        ytest_pred = Gbeta_test + ftest

        # save predictions
        ytrain_preds[s] = ytrain_pred
        ytest_preds[s] = ytest_pred

        ftrain_preds[s] = ftrain
        ftest_preds[s] = ftest

        Xbeta_train_preds[s] = Gbeta_train
        Xbeta_test_preds[s] = Gbeta_test

    # posterior mean predictions
    ytrain_pred_mean = ytrain_preds.mean(axis=0)
    ytest_pred_mean = ytest_preds.mean(axis=0)

    # RMSE of posterior mean prediction
    train_rmse_mean_prediction = root_mean_squared_error(ytrain, ytrain_pred_mean)
    test_rmse_mean_prediction = root_mean_squared_error(ytest, ytest_pred_mean)

    # one RMSE per posterior sample
    train_rmse_samples = np.sqrt(
        np.mean((ytrain_preds - ytrain[None, :]) ** 2, axis=1)
    )

    test_rmse_samples = np.sqrt(
        np.mean((ytest_preds - ytest[None, :]) ** 2, axis=1)
    )

    # posterior RMSE summaries
    train_rmse_post_mean = train_rmse_samples.mean()
    train_rmse_ci_lower, train_rmse_ci_upper = np.quantile(
        train_rmse_samples, [0.025, 0.975]
    )

    test_rmse_post_mean = test_rmse_samples.mean()
    test_rmse_ci_lower, test_rmse_ci_upper = np.quantile(
        test_rmse_samples, [0.025, 0.975]
    )

    rmse_summary = pd.DataFrame({
        "train": [
            n_samples,
            train_rmse_mean_prediction,
            train_rmse_post_mean,
            train_rmse_ci_lower,
            train_rmse_ci_upper,
        ],
        "test": [
            n_samples,
            test_rmse_mean_prediction,
            test_rmse_post_mean,
            test_rmse_ci_lower,
            test_rmse_ci_upper,
        ],
    }, index=[
        "n_posterior_samples",
        "rmse_of_posterior_mean_prediction",
        "posterior_rmse_mean",
        "posterior_rmse_ci_lower",
        "posterior_rmse_ci_upper",
    ])

    rmse_summary_path = run_dir / "rmse_summary.csv"
    rmse_summary.to_csv(rmse_summary_path, index=True)

    # prediction summary for plot_predictions()
    train_prediction_summary = pd.DataFrame({
        "split": "train",
        "y_true": ytrain,
        "y_pred_mean": ytrain_pred_mean,
        "y_pred_p05": np.quantile(ytrain_preds, 0.05, axis=0),
        "y_pred_p95": np.quantile(ytrain_preds, 0.95, axis=0),
        "f_pred_mean": ftrain_preds.mean(axis=0),
        "Xbeta_pred_mean": Xbeta_train_preds.mean(axis=0),
    })

    test_prediction_summary = pd.DataFrame({
        "split": "test",
        "y_true": ytest,
        "y_pred_mean": ytest_pred_mean,
        "y_pred_p05": np.quantile(ytest_preds, 0.05, axis=0),
        "y_pred_p95": np.quantile(ytest_preds, 0.95, axis=0),
        "f_pred_mean": ftest_preds.mean(axis=0),
        "Xbeta_pred_mean": Xbeta_test_preds.mean(axis=0),
    })

    prediction_summary = pd.concat(
        [train_prediction_summary, test_prediction_summary],
        ignore_index=True
    )

    prediction_summary_path = run_dir / "prediction_summary.csv"
    prediction_summary.to_csv(prediction_summary_path, index=False)

    print(f"Predictions and RMSEs calculated for {n_samples} posterior samples.")
    print(f"Train RMSE of posterior mean prediction: {train_rmse_mean_prediction}")
    print(f"Test RMSE of posterior mean prediction: {test_rmse_mean_prediction}")
    print(f"RMSE summary saved to {rmse_summary_path}")
    print(f"Prediction summary saved to {prediction_summary_path}")

    return {
        "rmse_summary_path": rmse_summary_path,
        "prediction_summary_path": prediction_summary_path,
    }




def predictions_lasso(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    """
    Bayesian Lasso predictions and posterior RMSE summaries.

    Saves one compact CSV containing:
    - split
    - number of posterior samples
    - RMSE of posterior mean prediction
    - posterior mean RMSE
    - 95% credible interval for posterior RMSE
    """

    # make output directory
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # convert inputs to numpy arrays
    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain)
    ytest = np.asarray(ytest)

    # get posterior beta samples
    posterior = trace.posterior
    beta = posterior["beta"].values  # shape: (chains, draws, features)

    # flatten chains and draws
    beta_flat = beta.reshape(-1, beta.shape[-1])  # shape: (samples, features)
    n_samples = beta_flat.shape[0]

    # predictions from every posterior sample
    ytrain_preds = beta_flat @ Xtrain.T  # shape: (samples, n_train)
    ytest_preds = beta_flat @ Xtest.T    # shape: (samples, n_test)

    # posterior mean predictions
    ytrain_pred_mean = ytrain_preds.mean(axis=0)
    ytest_pred_mean = ytest_preds.mean(axis=0)

    # RMSE of posterior mean prediction
    train_rmse_mean_prediction = root_mean_squared_error(ytrain, ytrain_pred_mean)
    test_rmse_mean_prediction = root_mean_squared_error(ytest, ytest_pred_mean)

    # one RMSE per posterior sample
    train_rmse_samples = np.sqrt(
        np.mean((ytrain_preds - ytrain[None, :]) ** 2, axis=1)
    )

    test_rmse_samples = np.sqrt(
        np.mean((ytest_preds - ytest[None, :]) ** 2, axis=1)
    )

    # posterior RMSE summaries
    train_rmse_post_mean = train_rmse_samples.mean()
    train_rmse_ci_lower, train_rmse_ci_upper = np.quantile(
        train_rmse_samples, [0.025, 0.975]
    )

    test_rmse_post_mean = test_rmse_samples.mean()
    test_rmse_ci_lower, test_rmse_ci_upper = np.quantile(
        test_rmse_samples, [0.025, 0.975]
    )

    rmse_summary = pd.DataFrame({
        "train": [
            n_samples,
            train_rmse_mean_prediction,
            train_rmse_post_mean,
            train_rmse_ci_lower,
            train_rmse_ci_upper,
        ],
        "test": [
            n_samples,
            test_rmse_mean_prediction,
            test_rmse_post_mean,
            test_rmse_ci_lower,
            test_rmse_ci_upper,
        ],
    }, index=[
        "n_posterior_samples",
        "rmse_of_posterior_mean_prediction",
        "posterior_rmse_mean",
        "posterior_rmse_ci_lower",
        "posterior_rmse_ci_upper",
    ])


    rmse_summary_path = run_dir / "rmse_summary.csv"
    rmse_summary.to_csv(rmse_summary_path, index=True)

    print(f"Predictions and RMSEs calculated for {n_samples} posterior samples.")
    print(f"RMSE summary saved to {rmse_summary_path}")



def predictions_gp(trace, Xtrain, ytrain, Xtest, ytest, run_dir):
    """
    Gaussian Process predictions and posterior RMSE summaries.

    Saves:
    - rmse_summary.csv
    - prediction_summary.csv needed by plot_predictions()
    """

    print("Calculating GP predictions for each posterior sample...")

    # make output directory
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # convert inputs to numpy arrays
    Xtrain = np.asarray(Xtrain)
    Xtest = np.asarray(Xtest)
    ytrain = np.asarray(ytrain)
    ytest = np.asarray(ytest)

    # get posterior samples
    posterior = trace.posterior
    beta = posterior["beta"].values
    ell = posterior["ell"].values
    sigma2_gp = posterior["sigma2_gp"].values
    sigma2_noise = posterior["sigma2_noise"].values

    n_train = len(Xtrain)
    n_test = len(Xtest)
    I = np.eye(n_train)

    chains, draws, features = beta.shape
    n_samples = chains * draws

    # flatten chains and draws
    beta_flat = beta.reshape(n_samples, features)
    ell_flat = ell.reshape(n_samples, ell.shape[-1])
    sigma2_gp_flat = sigma2_gp.reshape(n_samples)
    sigma2_noise_flat = sigma2_noise.reshape(n_samples)

    # predictions from every posterior sample
    ytrain_preds = np.zeros((n_samples, n_train))
    ytest_preds = np.zeros((n_samples, n_test))

    # NEW: save GP and linear components separately
    ftrain_preds = np.zeros((n_samples, n_train))
    ftest_preds = np.zeros((n_samples, n_test))
    Xbeta_train_preds = np.zeros((n_samples, n_train))
    Xbeta_test_preds = np.zeros((n_samples, n_test))

    # loop through each posterior sample
    for s in range(n_samples):
        beta_s = beta_flat[s]
        ell_s = ell_flat[s]
        sigma2_gp_s = float(sigma2_gp_flat[s])
        sigma2_noise_s = float(sigma2_noise_flat[s])

        residuals = ytrain - Xtrain @ beta_s

        # GP covariance matrices
        Ktrain = rbf_kernel(Xtrain, Xtrain, ell_s, sigma2_gp_s)
        Ktest = rbf_kernel(Xtest, Xtrain, ell_s, sigma2_gp_s)

        # C = K + sigma2_noise * I
        Ctrain = Ktrain + sigma2_noise_s * I
        alpha = np.linalg.solve(Ctrain, residuals)

        # GP posterior mean component
        ftrain = Ktrain @ alpha
        ftest = Ktest @ alpha

        # linear component
        Xbeta_train = Xtrain @ beta_s
        Xbeta_test = Xtest @ beta_s

        # save components
        ftrain_preds[s] = ftrain
        ftest_preds[s] = ftest
        Xbeta_train_preds[s] = Xbeta_train
        Xbeta_test_preds[s] = Xbeta_test

        # full prediction: y_pred = X beta + f_pred
        ytrain_preds[s] = Xbeta_train + ftrain
        ytest_preds[s] = Xbeta_test + ftest

    # posterior mean predictions
    ytrain_pred_mean = ytrain_preds.mean(axis=0)
    ytest_pred_mean = ytest_preds.mean(axis=0)

    # RMSE of posterior mean prediction
    train_rmse_mean_prediction = root_mean_squared_error(ytrain, ytrain_pred_mean)
    test_rmse_mean_prediction = root_mean_squared_error(ytest, ytest_pred_mean)

    # one RMSE per posterior sample
    train_rmse_samples = np.sqrt(
        np.mean((ytrain_preds - ytrain[None, :]) ** 2, axis=1)
    )

    test_rmse_samples = np.sqrt(
        np.mean((ytest_preds - ytest[None, :]) ** 2, axis=1)
    )

    # posterior RMSE summaries
    train_rmse_post_mean = train_rmse_samples.mean()
    train_rmse_ci_lower, train_rmse_ci_upper = np.quantile(
        train_rmse_samples, [0.025, 0.975]
    )

    test_rmse_post_mean = test_rmse_samples.mean()
    test_rmse_ci_lower, test_rmse_ci_upper = np.quantile(
        test_rmse_samples, [0.025, 0.975]
    )

    rmse_summary = pd.DataFrame({
        "train": [
            n_samples,
            train_rmse_mean_prediction,
            train_rmse_post_mean,
            train_rmse_ci_lower,
            train_rmse_ci_upper,
        ],
        "test": [
            n_samples,
            test_rmse_mean_prediction,
            test_rmse_post_mean,
            test_rmse_ci_lower,
            test_rmse_ci_upper,
        ],
    }, index=[
        "n_posterior_samples",
        "rmse_of_posterior_mean_prediction",
        "posterior_rmse_mean",
        "posterior_rmse_ci_lower",
        "posterior_rmse_ci_upper",
    ])

    rmse_summary_path = run_dir / "rmse_summary.csv"
    rmse_summary.to_csv(rmse_summary_path, index=True)

    # NEW: save prediction summary for plot_predictions()
    train_prediction_summary = pd.DataFrame({
        "split": "train",
        "y_true": ytrain,
        "y_pred_mean": ytrain_preds.mean(axis=0),
        "y_pred_p05": np.quantile(ytrain_preds, 0.05, axis=0),
        "y_pred_p95": np.quantile(ytrain_preds, 0.95, axis=0),
        "f_pred_mean": ftrain_preds.mean(axis=0),
        "Xbeta_pred_mean": Xbeta_train_preds.mean(axis=0),
    })

    test_prediction_summary = pd.DataFrame({
        "split": "test",
        "y_true": ytest,
        "y_pred_mean": ytest_preds.mean(axis=0),
        "y_pred_p05": np.quantile(ytest_preds, 0.05, axis=0),
        "y_pred_p95": np.quantile(ytest_preds, 0.95, axis=0),
        "f_pred_mean": ftest_preds.mean(axis=0),
        "Xbeta_pred_mean": Xbeta_test_preds.mean(axis=0),
    })

    prediction_summary = pd.concat(
        [train_prediction_summary, test_prediction_summary],
        ignore_index=True
    )

    prediction_summary_path = run_dir / "prediction_summary.csv"
    prediction_summary.to_csv(prediction_summary_path, index=False)

    print(f"Predictions and RMSEs calculated for {n_samples} posterior samples.")
    print(f"RMSE summary saved to {rmse_summary_path}")
    print(f"Prediction summary saved to {prediction_summary_path}")

    return {
        "rmse_summary_path": rmse_summary_path,
        "prediction_summary_path": prediction_summary_path,
    }


def plot_predictions(run_dir):
    '''
    1. plot predicted vs true values with 90% credible intervals
    2. plot gp prediction and linear component vs true values 


    '''
    run_dir = Path(run_dir)
    prediction_summary_path = run_dir / "prediction_summary.csv"

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
    plt.close()

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
    plt.close()

    print(f"\nPrediction plots saved to {prediction_summary_path.parent}/predicted_vs_true.png and {prediction_summary_path.parent}/gp_vs_linear.png")