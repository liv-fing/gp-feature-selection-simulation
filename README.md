# Sparse Feature Selection for Gaussian Process Regression

This project benchmarks sparse feature selection methods for Gaussian Process (GP) regression across synthetic datasets with varying levels of sparsity, noise, and sample size. 
It was completed as part of an independent study (IEMS 399) at Northwestern University, and it was advised by Professor of Industrial Engineering, Moses Chan. 

## Objective

To evaluate how different feature selection techniques impact GP regression performance in high-dimensional, sparse settings — both in terms of predictive accuracy and computational efficiency.

## Methods

- Generated 105 synthetic datasets with controlled sparsity, Gaussian noise, and variable sample sizes
- Implemented GP regression models using **TensorFlow** and **GPflow**
- Compared five feature selection strategies:
  - Standard GP with shared lengthscales (no feature selection)
  - Automatic Relevance Determination (ARD) GP
  - Lasso-based feature selection followed by standard GP
  - Lasso-based feature selection followed by ARD-enabled GP
  - GP with a custom L1-penalized penalty function (experimental modification to encourage sparsity)
    
The entire simulation pipeline was implemented using a custom Python class, allowing for clean, repeatable experimentation across all model types.


## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **Coefficient Error** (L2 norm between estimated and true coefficients)
- **Precision / Recall** on selected features
- **Runtime** for training, tuning, and inference

## Tools Used

- Python 3  
- TensorFlow  
- GPflow  
- NumPy, SciPy  
- Matplotlib / Seaborn for visualization

## Output

- All simulation results are saved to CSVs and visualized as comparative plots.
- A research paper (`IEMS399_Final_Report.pdf`) that summarizes the methodology, results, and implications for sparse regression with GP models. It is included in the root directory.

## Code Structure

- `gp_feature_select.py` – Contains a custom Python class that wraps feature selection and GP regression. Supports ARD, Lasso-based filtering, and L1-penalized GP models with modular training and evaluation methods.
- `notebooks/` – Jupyter notebooks used for simulation, analysis, and visualization.
- `synthetic_data/` – Synthetic datasets generated with controlled sparsity and noise.
- `figures/` – Plots comparing model performance across settings.
- `archive/` – Old experiments and notebooks kept for reference.
