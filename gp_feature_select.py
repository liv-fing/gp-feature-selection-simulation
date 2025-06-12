#!/usr/bin/env python
# coding: utf-8

# # 10. GPFeatureSelect
# 
# Custom Class. Same code as '9. Simulation'

# In[2]:


# import libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gpflow as gpf
import json
import tensorflow as tf
import time
from tracemalloc import start


from sklearn.linear_model import LinearRegression, LassoLarsCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# In[391]:


class GPFeatureSelect:

    def __init__(self, model_type = 'std', cv = 5):
        self.model_type = model_type
        self.cv = cv
        self.gp_model = None
        self.scaler = StandardScaler()
        self.scaledX = None
        self.selected_features = None
        self.lasso_model = None
        self.lambda_val = None
        self.beta_hat = None
        self.lin_intercept = None
        self.runtime = None
        self.tunetime = None
        self.fittime = None
        self.predicttime = None
        self.opt = gpf.optimizers.Scipy()
        
    def training_loss_lasso(self):
        base_loss = self.gp_model.training_loss()
        l1_penalty = self.lambda_val * tf.reduce_sum(tf.abs(self.gp_model.mean_function.A))
        total_loss = base_loss + l1_penalty
        return total_loss

    def cv_lasso_lars(self, X, y):
        tune_start = time.time()
        if self.model_type in ['lasso_std', 'lasso_ard']:
            las = LassoLarsCV(cv = self.cv)
            las.fit(X, y.ravel())
            mask = np.abs(las.coef_) > 1e-4
            self.selected_features = np.where(mask)[0]
            self.beta_hat = las.coef_
            self.lin_intercept = las.intercept_
            self.lasso_model = las
            self.lambda_val = las.alpha_
            self.tunetime = time.time() - tune_start
            return X[:, self.selected_features]
            
        else:
            raise ValueError("cv_lasso_lars called on non-lasso model")

    def tune_lambda(self, X, y):
        start_tunetime = time.time()
        def run_cv(lambda_grid):
            best_lbda = None
            best_rmse = np.inf
            lambda_rmse_pairs = []


            for l in lambda_grid:
                rmses = []
                kf = KFold(n_splits=self.cv, shuffle=True, random_state=22)
                for train_index, val_index in kf.split(X):
                    X_train, X_val = X[train_index], X[val_index]
                    y_train, y_val = y[train_index], y[val_index]

                    self.init_ard_gp_mod(X_train, y_train)
                    self.lambda_val = l
                    self.opt.minimize(
                        lambda: self.training_loss_lasso(),
                        self.gp_model.trainable_variables
                    )

                    y_pred = self.gp_model.predict_f(X_val)[0].numpy().flatten()
                    rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
                    rmses.append(rmse)

                avg_rmse = np.mean(rmses)
                lambda_rmse_pairs.append((l, avg_rmse))

                #print('\nRMSE for this fold: ', avg_rmse)

                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_lbda = l
            self.tunetime = time.time() - start_tunetime
            return best_lbda, lambda_rmse_pairs

        coarse_grid = np.logspace(-1,0.5, 10)
        best_coarse, coarse_log = run_cv(coarse_grid)

        fine_grid = np.logspace(np.log10(best_coarse * 0.5), np.log10(best_coarse * 2), 5)
        best_fine, fine_log = run_cv(fine_grid)
        self.lambda_val = best_fine
        self.lambda_rmse_log = coarse_log + fine_log
        return best_fine

    def init_gp_mod(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        m = X.shape[1]
        if self.beta_hat is not None:
            A_init = self.beta_hat[self.selected_features].reshape(-1, 1)
            if A_init.shape[0] != m:
                raise ValueError(f"Shape mismatch: A_init.shape[0] = {A_init.shape[0]}, m = {m}")
        else:
            A_init = tf.zeros((m, 1), dtype=tf.float64)

        if self.lin_intercept is not None:
            b_init = tf.constant(self.lin_intercept, dtype=tf.float64)
        else:
            b_init = tf.zeros((1,), dtype=tf.float64)

        kernel = gpf.kernels.SquaredExponential(lengthscales=1)
        likelihood = gpf.likelihoods.Gaussian()
        mean_function = gpf.mean_functions.Linear(A=A_init, b = b_init) 

        self.gp_model = gpf.models.GPR(data = (X, y.reshape(-1,1)), kernel = kernel, likelihood = likelihood, mean_function = mean_function)

        # prevent mean function from being retrained
        gpf.set_trainable(self.gp_model.mean_function.A, False)
        gpf.set_trainable(self.gp_model.mean_function.b, False)

    def init_ard_gp_mod(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        m = X.shape[1]
        
        if self.beta_hat is not None:
            A_init = self.beta_hat[self.selected_features].reshape(-1, 1)
            if A_init.shape[0] != m:
                raise ValueError(f"Shape mismatch: A_init.shape[0] = {A_init.shape[0]}, m = {m}")
        else:
            A_init = tf.zeros((m, 1), dtype=tf.float64)

        if self.lin_intercept is not None:
            b_init = tf.constant(self.lin_intercept, dtype=tf.float64)
        else:
            b_init = tf.zeros((1,), dtype=tf.float64)

        kernel = gpf.kernels.SquaredExponential(lengthscales=np.ones(m))
        likelihood = gpf.likelihoods.Gaussian()
        mean_function = gpf.mean_functions.Linear(A=A_init, b = b_init) 

        self.gp_model = gpf.models.GPR(data = (X, y.reshape(-1,1)), kernel = kernel, likelihood = likelihood, mean_function = mean_function)

        # prevent mean function from being retrained
        gpf.set_trainable(self.gp_model.mean_function.A, False)
        gpf.set_trainable(self.gp_model.mean_function.b, False)


    def fit(self, X, y):
        fit_start_time = time.time()

        X = self.scaler.fit_transform(X)
        y = np.asarray(y).reshape(-1,1)

        if self.model_type == 'std':
            self.init_gp_mod(X, y)
            self.opt.minimize(
                self.gp_model.training_loss,
                self.gp_model.trainable_variables)
            self.beta_hat = self.gp_model.mean_function.A.numpy().flatten()

        elif self.model_type =='ard':
            self.init_ard_gp_mod(X, y)
            self.opt.minimize(
                self.gp_model.training_loss,
                self.gp_model.trainable_variables)
            self.beta_hat = self.gp_model.mean_function.A.numpy().flatten()

        elif self.model_type =='lasso_std':
            reducedX = self.cv_lasso_lars(X,y)
            self.init_gp_mod(reducedX, y)
            self.opt.minimize(
                self.gp_model.training_loss,
                self.gp_model.trainable_variables)
            self.beta_hat = self.gp_model.mean_function.A.numpy().flatten()
            
        elif self.model_type == 'lasso_ard':
            reducedX = self.cv_lasso_lars(X,y)
            self.init_ard_gp_mod(reducedX, y)
            self.opt.minimize(
                self.gp_model.training_loss,
                self.gp_model.trainable_variables)
            self.beta_hat = self.gp_model.mean_function.A.numpy().flatten()

        elif self.model_type == 'l1_gp': 
            if self.lambda_val is None:
                self.tune_lambda(X,y)

            # start by training on all features
            self.init_ard_gp_mod(X, y)
            self.opt.minimize(
                lambda: self.training_loss_lasso(),
                self.gp_model.trainable_variables   
            )

            # threshold coefficients to select features
            beta_full = self.gp_model.mean_function.A.numpy().flatten()

            threshold = 0.05 * np.max(np.abs(beta_full)) # using 5% of max magnitude
            mask = np.abs(beta_full) > threshold

            if np.sum(mask) == 0: # keep at least one feature
                mask[np.argmax(np.abs(beta_full))] = True

            beta_full[~mask] = 0.0
            self.selected_features = np.where(mask)[0]
            self.beta_hat = beta_full[mask]

            Anew = self.beta_hat.reshape(-1,1)
            b_same = self.gp_model.mean_function.b
            assert Anew.shape[0] == X[:, self.selected_features].shape[1], (
                f"Anew.shape = {Anew.shape}, X[:, selected_features].shape = {X[:, self.selected_features].shape}")

            self.gp_model.mean_function = gpf.mean_functions.Linear(A=Anew, b = b_same)
            self.gp_model.data = (X[:, self.selected_features], y)

            old_ls = self.gp_model.kernel.lengthscales.numpy()
            new_ls = old_ls[self.selected_features]
            self.gp_model.kernel = gpf.kernels.SquaredExponential(
                lengthscales=new_ls, 
                active_dims=list(range(len(self.selected_features))))
            
            if len(self.selected_features) == 0:
                print ('No features selected after thresholding')
                return
            
        else:
            raise ValueError("Unknown model type")
        
        self.fittime = (time.time() - fit_start_time - self.tunetime) if self.tunetime is not None else (time.time() - fit_start_time)
        
    def predict(self, X, istest=False):
        if istest:
            predict_start = time.time()

        X = self.scaler.transform(X)

        if self.selected_features is not None:
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, self.selected_features]
            else:
                X = X[:, self.selected_features]

        X = np.asarray(X)

        mean, _ = self.gp_model.predict_f(X)

        if istest:
            self.predicttime = time.time() - predict_start
        return mean.numpy().flatten()
    
    def get_metrics(self, Xtrain, ytrain, Xtest, ytest, beta_true=None):
        ytrain_pred = self.predict(Xtrain, istest = False)
        train_rmse = np.sqrt(np.mean((ytrain - ytrain_pred) ** 2))
        
        ytest_pred = self.predict(Xtest, istest = True)
        test_rmse = np.sqrt(np.mean((ytest - ytest_pred) ** 2))

        beta_error = None
        precision = None
        recall = None

        if beta_true is not None and self.beta_hat is not None:

            beta_hat_full = np.zeros(len(beta_true))

            if self.selected_features is not None:
                beta_hat_full[self.selected_features] = self.beta_hat
            else:
                #no selection -> full model used
                beta_hat_full = self.beta_hat
                self.selected_features = np.arange(len(beta_true)) 

            beta_true_bin = (beta_true != 0).astype(int)
            beta_hat_full_bin = (beta_hat_full != 0).astype(int)
            beta_error = np.sqrt(np.mean((beta_true_bin - beta_hat_full_bin) ** 2))

            [tn, fp, fn, tp] = confusion_matrix(beta_true_bin, beta_hat_full_bin, labels = [0, 1]).ravel()
                    
            if (tp + fp) == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)

            if (tp + fn) == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            
        self.runtime = (self.fittime or 0) + (self.predicttime or 0) + (self.tunetime or 0)
        
        return {
            'selected features': self.selected_features if self.selected_features is not None else 'All features',
            'precision': precision,
            'recall': recall,
            'training rmse': train_rmse.round(4),
            'testing rmse': test_rmse.round(4),
            'beta_error': beta_error.round(4),
            'tune time': round(self.tunetime, 4) if self.tunetime is not None else None,
            'fit time': round(self.fittime, 4) if self.fittime is not None else None,
            'prediction time (test)': round(self.predicttime, 4) if self.predicttime is not None else None,
            'total runtime': round(self.runtime, 4) if self.runtime is not None else None,
            'lambda': self.lambda_val.round(4) if self.lambda_val is not None else None
        }
    

