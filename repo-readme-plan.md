# Plan: Repository Polish for Job & Grad School Applications

## Top-Level Overview

**Goal**: Make this repository immediately legible and impressive to non-technical (recruiters, admissions officers) and technical (researchers, professors) readers who arrive via a resume or application link.

**Recent structural changes that caused breakage**:
- `phase1/`, `phase2-MCMC/`, `phase3-refining/`, `phase4-OGP/` were moved inside a new `initial_project_phases/` folder (all files that assumed they were at the repo root are now one level deeper)
- `synthetic_data/` and `synthetic_data_large_coefficients/` were moved inside a new `data/` folder
- `phase5-writing/` was renamed to `phase5-final/`

**Scope**:
1. Fix all runtime-breaking path and import errors across Python files and Jupyter notebooks
2. Fix comments/documentation that reference the old folder structure
3. Write a polished root-level `README.md`

**Non-goals**: Unused import cleanup, run instructions, result summaries, changes to archive notebooks with old Dengue data (those are truly archived and not related to this research).

---

## Sub-Task 1 — Fix Broken Import in `model0_ZeroMeanGP.py`

**Intent**: `model0_ZeroMeanGP.py` imports a non-existent function `predictions` — the real function is `predictions_gp`. This causes an `ImportError` on import and prevents Model 0 from running.

**Expected Outcomes**:
- Line 32 of `phase5-final/model0_ZeroMeanGP.py` imports `predictions_gp` instead of `predictions`
- No `ImportError` when the file is imported

**Todo List**:
- [ ] In `phase5-final/model0_ZeroMeanGP.py` line 32, change `predictions` to `predictions_gp` in the import statement

**Relevant Context**:
- Broken file: `phase5-final/model0_ZeroMeanGP.py`, line 32
- Source of truth: `phase5-final/helper_funcs/predicting.py` (defines `predictions_gp`, not `predictions`)

**Status**: `[x] done`

---

## Sub-Task 2 — Fix All Hardcoded Absolute Paths in Python Files

**Intent**: Multiple `.py` files across phases 3, 4, and 5 contain hardcoded absolute paths tied to the researcher's local machine (`/Users/liviafingerson/...`). These also point to the old pre-reorganization folder structure (`synthetic_data_large/` instead of `data/synthetic_data_large_coefficients/`). All must be replaced with portable relative paths anchored to `__file__`.

**Expected Outcomes**:
- All 7 Python files use `Path(__file__).parent...` to navigate to the data folder
- The paths correctly resolve to `data/synthetic_data_large_coefficients/simulated_datasets_large_coef/` from the repo root
- Code works on any machine regardless of where the repo is cloned

**Todo List**:
- [ ] `phase5-final/helper_funcs/data_setup.py` line 144: Replace hardcoded path with `Path(__file__).parent.parent.parent / "data" / "synthetic_data_large_coefficients" / "simulated_datasets_large_coef"` (file is 3 levels deep: `helper_funcs/` → `phase5-final/` → repo root)
- [ ] `phase5-final/helper_funcs/data_setup.py` line 145 (commented): Update the commented-out `synthetic_data` path to `Path(__file__).parent.parent.parent / "data" / "synthetic_data"` for consistency
- [ ] `initial_project_phases/phase4-OGP/machinery/helper_funcs/data_setup.py` line 110: Replace hardcoded path with `Path(__file__).parent.parent.parent.parent / "data" / "synthetic_data_large_coefficients" / "simulated_datasets_large_coef"` (4 levels deep: `helper_funcs/` → `machinery/` → `phase4-OGP/` → `initial_project_phases/` → repo root)
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_matern.py` line 210: Replace hardcoded path with `Path(__file__).parent.parent.parent / "data" / "synthetic_data_large_coefficients" / "simulated_datasets_large_coef"` (3 levels deep: `machinery/` → `phase3-refining/` → `initial_project_phases/` → repo root)
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_justGP.py` line 194: Same fix as above
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_main.py` line 204: Same fix as above
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_mega.py` line 273: Same fix as above
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v5_ols.py` line 194: Same fix as above

**Relevant Context**:
- All broken paths reference old name `synthetic_data_large` (without `_coefficients`)
- All paths also reference old pre-`data/` location
- `from pathlib import Path` is already imported in all these files

**Status**: `[x] done`

---

## Sub-Task 3 — Fix Broken Paths in Jupyter Notebooks

**Intent**: Several Jupyter notebooks contain paths that are now wrong — either hardcoded absolute paths or relative paths that assumed the notebooks were at the repo root (before the folders were moved into `initial_project_phases/`). Active notebooks (phase 1 analysis, phase 2 MCMC, lambda testing) need fixed paths; archive notebooks with Dengue data can be skipped.

**Expected Outcomes**:
- All active notebooks in `initial_project_phases/phase1/notebooks/` and `initial_project_phases/phase2-MCMC/` load data from the correct locations
- `phase5-final/make_plots.ipynb` no longer references the old `phase5-writing` folder name

**Todo List**:
- [ ] `initial_project_phases/phase1/notebooks/simulation_on_synthetic_datasets.ipynb` cell with `meta_path`: Replace absolute path (`/Users/liviafingerson/Documents/GitHub/IEMS399-GP/Simulation Datasets/...`) with relative path `../../data/synthetic_data_large_coefficients/...`
- [ ] `initial_project_phases/phase1/notebooks/analysis_of_results.ipynb` cell with `root_dir`: Replace absolute path (`/Users/liviafingerson/Documents/GitHub/IEMS399-GP/Simulation Datasets`) with relative path `../../data/synthetic_data_large_coefficients`
- [ ] `initial_project_phases/phase2-MCMC/MCMC/lambda_testing/lambda_test_analysis.ipynb` cell with `filepath`: Replace absolute path (`/Users/liviafingerson/Desktop/GitHub/IEMS399-GP/MCMC/lambda_testing/50walkers/...`) with relative path `../50walkers/emcee_lbda{lbda}_result.json`
- [ ] `initial_project_phases/phase2-MCMC/MCMC/Archive/bilby_gp_test.ipynb`: Fix `sample_dat_path` and `val_dat_path` — change `../synthetic_data/` to `../../../data/synthetic_data_large_coefficients/`
- [ ] `initial_project_phases/phase2-MCMC/MCMC/Archive/bilby_linear_test.ipynb`: Same fix — `../synthetic_data/` to `../../../data/synthetic_data_large_coefficients/`
- [ ] `initial_project_phases/phase2-MCMC/MCMC/Archive/BLxBilby.ipynb`: Same fix — `../synthetic_data/` to `../../../data/synthetic_data_large_coefficients/`
- [ ] `initial_project_phases/phase2-MCMC/MCMC/Archive/penalized_gp.ipynb`: Same fix in the two commented-out path lines
- [ ] `phase5-final/make_plots.ipynb`: Remove the `phase5-writing/` prefix from the hardcoded path string on line 54 (and remove the `path.replace('phase5-writing/', '')` workaround on line 61 if it exists)

**Relevant Context**:
- Phase 1 notebooks are 2 levels deep inside `initial_project_phases/phase1/notebooks/`, so `../../` reaches repo root
- Phase 2 archive notebooks are 3 levels deep inside `initial_project_phases/phase2-MCMC/MCMC/Archive/`, so `../../../` reaches repo root
- Lambda testing notebook is 3 levels deep but references a sibling folder, so `../50walkers/` is the correct relative path
- Archive notebooks with Dengue data (`MaxIncidence Models.ipynb`, `7_Feature_Selection_GP_with_Dengue_Data.ipynb`, `5. GridSearch.ipynb`) should be skipped — they reference a different dataset entirely unrelated to this research

**Status**: `[x] done`

---

## Sub-Task 4 — Fix Stale `cd` Commands in Code Comments

**Intent**: Several Python files contain `cd` commands in their header comments (instructions for how to run the script from terminal). These reference old folder paths — either `phase5-writing` or the old repo-root locations of phase3/phase4 (before they moved into `initial_project_phases/`). Fixing these keeps the repo honest and prevents confusion for any reader who tries to follow the instructions.

**Expected Outcomes**:
- All `cd` commands in comments across phase5-final and initial_project_phases reference the current correct paths

**Todo List**:
- [ ] `phase5-final/model0_ZeroMeanGP.py` line 7: Change `cd Desktop/GitHub/IEMS399-GP/phase5-writing` → `cd Desktop/GitHub/IEMS399-GP/phase5-final`
- [ ] `phase5-final/model1_BL_baseline.py` line 6: Same fix
- [ ] `phase5-final/model2_BL_GP_ARD.py` line 5: Same fix
- [ ] `phase5-final/model2_BL_GP_noARD.py` line 5: Same fix
- [ ] `phase5-final/model4_OGP_SampBeta.py` line 12: Change `cd Desktop/GitHub/IEMS399-GP/phase4-OGP` → `cd Desktop/GitHub/IEMS399-GP/phase5-final`
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_matern.py` line 20: Change `cd Desktop/GitHub/IEMS399-GP/phase3-refining` → `cd Desktop/GitHub/IEMS399-GP/initial_project_phases/phase3-refining`
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_justGP.py` line 11: Same fix
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v4_main.py` line 14: Same fix
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_mega.py` line 25: Same fix
- [ ] `initial_project_phases/phase3-refining/machinery/mechanism_v5_ols.py` line 16: Same fix
- [ ] `initial_project_phases/phase4-OGP/machinery/mechanism_OGP_betanosample.py` line 7: Change `cd Desktop/GitHub/IEMS399-GP/phase4-OGP` → `cd Desktop/GitHub/IEMS399-GP/initial_project_phases/phase4-OGP`
- [ ] `initial_project_phases/phase4-OGP/machinery/mechanism_OGP_betasample_v2.py` line 16: Same fix
- [ ] `initial_project_phases/phase4-OGP/machinery/mechanism_OGP_betasample.py` line 9: Same fix

**Relevant Context**:
- These are all comments/docstrings — no runtime impact, but they mislead readers who follow the instructions

**Status**: `[x] done`

---

## Sub-Task 5 — Update Phase 1 README

**Intent**: The `initial_project_phases/phase1/README.md` references a `synthetic_data/` folder as if it lives inside phase1. That folder no longer exists there — it was moved to `data/` at the repo root. The README should point readers to the correct location.

**Expected Outcomes**:
- `initial_project_phases/phase1/README.md` correctly describes where the synthetic data now lives

**Todo List**:
- [ ] Read `initial_project_phases/phase1/README.md` to see its current content around the `synthetic_data/` reference (line 48)
- [ ] Update the reference to note the data has moved to `data/synthetic_data_large_coefficients/` at the repo root

**Relevant Context**:
- File: `initial_project_phases/phase1/README.md`, line 48

**Status**: `[x] done`

---

## Sub-Task 6 — Write the Root-Level `README.md`

**Intent**: Replace the current sparse README with a polished, two-tier document. The first tier is a narrative introduction accessible to any reader. The second tier is a technical breakdown for researchers or graduate program evaluators. The README must surface: the research question, the final paper, the phase progression, the final models, and the synthetic data experiments that did not appear in the final paper.

**Expected Outcomes**:
- `README.md` at the repo root is fully rewritten
- Non-technical readers understand what the research is about and why it matters within the first few paragraphs
- Technical readers can navigate directly to the relevant code for each model and phase
- The final paper PDF is linked prominently at the top
- The phase 1 report is mentioned in the research phases section
- A dedicated section flags the synthetic data experiments as additional work not in the paper, pointing to `data/`
- No run instructions, no specific result summaries

**Todo List**:
- [ ] Write a header section: title, one-sentence summary, link to final paper PDF
- [ ] Write a "Background & Research Question" section — what problem is being solved, why it matters, non-technical framing
- [ ] Write a "Methodology" section — brief technical explanation of GP regression, Bayesian Lasso, orthogonal GP, the 6 models compared
- [ ] Write a "Repository Structure" section — annotated tree explaining each top-level folder, with `phase5-final/` called out as the production code
- [ ] Write a "Research Progression" section — describe phases 1–5 as a narrative arc; link to `initial_project_phases/phase1/IEMS399_Final_Report.pdf`
- [ ] Write an "Additional Experiments: Synthetic Data" section — note that controlled synthetic experiments exist in `data/` and were used to validate methodology but did not appear in the final paper; point to `data/` folder
- [ ] Write an "Acknowledgements" section crediting Professor Moses Y.-H. Chan

**Relevant Context**:
- Existing README: `README.md` (root) — currently sparse
- Final paper: `Research_Paper_Livia_Fingerson_June_12.pdf`
- Phase 1 report: `initial_project_phases/phase1/IEMS399_Final_Report.pdf`
- Production code: `phase5-final/` (6 model files + `helper_funcs/`)
- Synthetic data: `data/synthetic_data_large_coefficients/`
- Research question: Can Bayesian Lasso priors recover interpretability of linear coefficients in GP regression, where the GP and linear components compete to explain the same variance?
- Models: Zero-mean GP (baseline), Bayesian Lasso linear (baseline), BL+GP with ARD, BL+GP without ARD, Orthogonal GP with analytical β, Orthogonal GP with sampled β
- Key finding: Two-stage orthogonal GP with analytical β outperforms joint inference on the diabetes dataset
- Tech stack: Python, PyMC, ArviZ, PyTensor, scikit-learn, NumPy, Pandas, Matplotlib
- Advisor: Professor Moses Y.-H. Chan, Northwestern University

**Status**: `[x] done`
