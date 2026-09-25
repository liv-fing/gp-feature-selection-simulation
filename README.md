# Bayesian Lasso Priors for Linear Mean Gaussian Process Regression

**Independent Study — IEMS 399, Northwestern University**  
**Advisor:** Professor Moses Y.-H. Chan  
**Author:** Livia Fingerson

📄 [Read the full research paper](Research_Paper_Livia_Fingerson_June_12.pdf)

---

## Overview

Gaussian Process (GP) regression is a powerful, flexible tool for modeling complex nonlinear relationships. A common extension is to add a linear mean function to the GP, which essentially lets the model integrate a broad linear trend with nonlinear residual structure. This is appealing because the linear component gives interpretable coefficients, meaning that you can examine the relative importance of individual features. 

The problem is that in practice, the linear component and the GP compete to explain the same variation in the data. The GP is flexible enough to absorb whatever the linear component doesn't claim, but that also means it is flexible enough to absorb the everything, shrinking the linear coefficients to zero and interfering with interpretability. Standard regularization approaches designed for pure regression don't solve this, because they operate on the linear component without accounting for the GP's ability to compensate.

This research investigates whether **Bayesian Lasso priors**, a Bayesian approach to sparse linear regression, can recover meaningful, interpretable linear coefficients in this setting, either alone or in combination with recent orthogonal GP methods designed to enforce a structural separation between the linear and nonlinear components.

---

## Research Question

> *Can Bayesian Lasso priors, alone or combined with orthogonal Gaussian process methods, recover interpretable linear coefficients in GP regression models where the linear and nonlinear components compete to explain the same variance?*

---

## Methodology

Six models were implemented and compared on the [diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) (a standard regression benchmark with 10 features):

| Model | Description |
|---|---|
| **Model 0** — Zero-mean GP | Baseline: pure GP with no linear component |
| **Model 1** — Bayesian Lasso | Baseline: sparse Bayesian linear regression only, no GP |
| **Model 2a** — BL + GP (ARD) | Bayesian Lasso linear mean + GP with per-feature lengthscales (ARD kernel) |
| **Model 2b** — BL + GP (no ARD) | Bayesian Lasso linear mean + GP with a shared lengthscale |
| **Model 3** — Orthogonal GP, analytical β | Linear mean fit via orthogonal projection after GP sampling; β computed analytically |
| **Model 4** — Orthogonal GP, sampled β | Orthogonal GP with Bayesian Lasso priors on β, sampled jointly |

The **orthogonal GP** approach (Models 3 & 4) is based on the framework of [Plumlee & Roshan (2016)](https://www.tandfonline.com/doi/abs/10.1080/01621459.2015.1119157), which constructs a modified kernel (C\*) that is mathematically orthogonal to the linear subspace, preventing the GP from absorbing linear trends by construction.

All models were implemented in **Python** using **PyMC** for Bayesian inference, **ArviZ** for posterior diagnostics, and standard scientific Python libraries (NumPy, Pandas, scikit-learn, Matplotlib).

---

## Key Finding

The joint Bayesian inference approaches (Models 2 & 4) struggled in practice: Bayesian Lasso hyperparameters tended to force linear coefficients toward zero while the GP captured all variance, or produced very wide credible intervals when regularization was relaxed. The **two-stage orthogonal GP** (Model 3), which computes β analytically from posterior GP samples rather than sampling everything jointly, achieved the best balance between predictive accuracy and coefficient interpretability on the diabetes dataset.

These results imply that competition between the Gaussian process and linear components may not be the reason the linear mean contributes insignificantly. Rather, insignificance may be the result, where heavy regularization improves the poor convergence of linear coefficients, even as it trivializes the linear component of the model.

---

## Repository Structure

```
bayesian-lasso-gp/
│
├── Research_Paper_Livia_Fingerson_June_12.pdf   ← Full research paper
│
├── phase5-final/                                ← Production code (start here)
│   ├── model0_ZeroMeanGP.py                     ← Model 0: zero-mean GP baseline
│   ├── model1_BL_baseline.py                    ← Model 1: Bayesian Lasso linear
│   ├── model2_BL_GP_ARD.py                      ← Model 2a: BL + GP with ARD
│   ├── model2_BL_GP_noARD.py                    ← Model 2b: BL + GP without ARD
│   ├── model3_OGP_AnBeta.py                     ← Model 3: Orthogonal GP, analytical β
│   ├── model4_OGP_SampBeta.py                   ← Model 4: Orthogonal GP, sampled β
│   ├── helper_funcs/                            ← Shared utilities (data, prediction, plotting)
│   ├── make_plots.ipynb                         ← Visualization notebook
│   └── results/                                 ← Saved posterior summaries and plots
│
├── data/                                        ← Datasets
│   ├── synthetic_data/                          ← Small synthetic datasets (phase 1)
│   └── synthetic_data_large_coefficients/       ← Larger synthetic datasets (phase 5)
│
└── initial_project_phases/                      ← Research history (exploratory work)
    ├── phase1/                                  ← Feature selection study (complete)
    ├── phase2-MCMC/                             ← MCMC inference experiments
    ├── phase3-refining/                         ← Model refinement
    └── phase4-OGP/                              ← Orthogonal GP development
```

---

## Research Progression

This repository captures the full arc of the research, from early exploration to the final production models. Each phase built on the lessons of the previous one.

**Phase 1 — Feature Selection Benchmark**  
The project began as a study of sparse feature selection for GP regression. Five methods were compared (standard GP, ARD GP, Lasso-filtered GP, and variations) across 105 synthetic datasets with controlled sparsity and noise levels. The central finding was that existing feature selection approaches don't cleanly separate which variables are important from how to model their relationships. This pointed toward the need for a more principled Bayesian approach to the linear component.  
📄 [Phase 1 report](initial_project_phases/phase1/IEMS399_Final_Report.pdf)

**Phase 2 — MCMC Inference Framework**  
Explored Markov Chain Monte Carlo sampling as a route to full Bayesian inference for GP models with linear means. Several sampling strategies were tested, including the Bilby framework. This phase established the core inference machinery and identified practical challenges with convergence.

**Phase 3 — Model Refinement**  
Iteratively refined the model formulation and inference procedure. Bayesian Lasso priors were introduced and tested. The competition problem between the linear and GP components became the central focus.

**Phase 4 — Orthogonal GP Development**  
Implemented the orthogonal GP kernel (C\*) from Plumlee & Roshan (2016). This phase produced the core mathematical machinery that became Models 3 and 4 in the final comparison.

**Phase 5 — Final Models & Analysis**  
All six competing models were implemented cleanly, run on the diabetes dataset, and analyzed. The results form the basis of the final paper.

---

## Additional Experiments: Synthetic Data

The final paper focuses on the diabetes dataset as a real-world benchmark. In addition to those experiments, a suite of **controlled synthetic data experiments** was developed to validate the methodology under known ground-truth conditions.

The synthetic datasets are stored in [`data/synthetic_data_large_coefficients/`](data/synthetic_data_large_coefficients/). Each dataset was generated from the model:

```
y = Xβ + ε(X) + η
```

where `β` is sparse (only 10%, 20%, or 50% of 30 features are active), `ε(X)` is structured GP noise, and `η` is independent Gaussian noise. Because the true active features and coefficients are known exactly, these datasets allow for precision/recall evaluation of feature recovery — a more direct measure of interpretability than is possible on real data.

These experiments were conducted as part of the research process but did not make it into the final paper.

---

## Acknowledgements

This research was conducted as an independent study (IEMS 399) at Northwestern University under the supervision of **Professor Moses Y.-H. Chan**, whose guidance shaped both the research direction and methodology throughout all phases of the project.
