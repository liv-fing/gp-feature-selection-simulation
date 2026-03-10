# result processing
'''
load functions with
from helper_funcs.result_processing import
    process_pymc_results 
    make_trace_plots
'''


# IMPORTS
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# CLEAN DATAFRAME
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

# TRACE PLOTS
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
