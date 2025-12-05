# CT-based Analysis and Machine-Learning Prediction of Lumbar Spondylolisthesis

This project provides a fully reproducible Python pipeline for analyzing CT-derived body composition metrics and predicting lumbar spondylolisthesis with classical statistical methods and machine-learning models. The code covers data loading, preprocessing, descriptive statistics, correlation and partial correlation analyses, quartile-based stratification, model training and evaluation, and automatic generation of tables and figures.

---

## Key Functions

- **Data handling**
  - Load a single CSV file containing all patient-level CT and clinical variables
  - Harmonize variable types (e.g., sex, binary outcomes)
  - Split data into training and test sets with stratification

- **Statistical analysis**
  - Prevalence of spondylolisthesis and mean slip grade, stratified by:
    - Sex
    - Age groups
    - Presence of pars defects
  - Spearman and Kendall rank correlations between slip grade and CT-derived metrics
  - Partial correlations between slip grade and key imaging variables adjusted for age and sex
  - Quartile-based stratification of selected variables (e.g., L3 IMATI, paravertebral fat area, PMFI) and comparison of prevalence and slip grade across quartiles

- **Machine-learning models**
  - End-to-end training and evaluation of six classifiers:
    - Logistic Regression  
    - Random Forest  
    - Gradient Boosting  
    - Linear SVM  
    - RBF SVM  
    - K-Nearest Neighbors  
  - Evaluation on a held-out test set with:
    - Accuracy  
    - Area under the ROC curve (AUC)  
    - Sensitivity (recall for the positive class)  
    - Specificity  

- **Model interpretation and visualization**
  - Random Forest feature importance for all input variables
  - ROC curves for all models in a single figure
  - Bar plots for model performance metrics
  - Figures summarizing correlation patterns and quartile-based stratification results

- **Single entry point**
  - Running one script executes the entire pipeline, saving all numerical results to `results/` and all figures to `figures/`.

---

## Project Structure

```text
.
├─ figures/                                  # Auto-generated plots
├─ results/                                  # Auto-generated tables and metrics
├─ config.py               # Global configuration (paths, feature names, seeds)
├─ data_loader.py          # Data loading and basic cleaning
├─ preprocessing.py        # Train/test split and preprocessing pipeline
├─ stats_analysis.py       # Descriptive statistics and correlation analyses
├─ ml_models.py            # Machine-learning models and evaluation
├─ plotting.py             # Plotting utilities for all main figures
├─ main.py                 # Entry script that runs the entire workflow
├─ requirements.txt        # Python dependencies
````

---

## Analysis Workflow

The workflow is organized into modular scripts so that each step can be reused or extended.

### 1. Configuration

All global settings are defined in `config.py`, including:

* Paths to the data, results, and figure directories
* Names of the outcome variables (e.g., binary spondylolisthesis indicator, slip grade)
* Lists of numeric and categorical features used in the analyses
* The subset of variables used in partial correlation analyses
* Random seeds and test set fraction

Editing `config.py` is usually sufficient to adapt the code to another dataset with similar structure.

### 2. Data Loading and Preprocessing

`data_loader.py`:

* Reads the CSV file from the `data/` directory
* Converts sex and other binary indicators to numeric form
* Ensures consistent data types for the outcome variables

`preprocessing.py`:

* Splits the dataset into training and test sets using stratified sampling on the binary outcome
* Applies a preprocessing pipeline (median imputation + standardization) to all features

### 3. Statistical Analyses

`stats_analysis.py` implements the statistical components of the study:

* `describe_prevalence_by_group`
  Computes group-wise sample size, prevalence of spondylolisthesis, and mean slip grade by sex, age categories, and presence of pars defects. The resulting table is saved as `results/table1_prevalence_by_group.csv`.

* `correlation_with_slip_grade`
  Calculates Spearman's rho and Kendall's tau between slip grade and each selected variable, saving the full table as `results/correlations_spearman_kendall.csv`.

* `partial_correlations_key_vars`
  Uses a residual-based method to estimate partial correlations between slip grade and key imaging variables after adjusting for age and sex. Results are stored in `results/partial_correlations.csv`.

* `quartile_stratification`
  Divides selected variables into quartiles and, for each quartile, computes sample size, prevalence of spondylolisthesis, and mean slip grade. The summary is saved as `results/quartile_summary.csv`.

The helper function `run_all_stats` executes all of these analyses in sequence.

### 4. Machine-Learning Models

`ml_models.py`:

* Defines six standard classifiers with reasonable default hyperparameters
* Trains each model on the preprocessed training data
* Evaluates performance on the test data using accuracy, AUC, sensitivity, and specificity
* Computes ROC curves for all models
* Extracts feature importance values from the Random Forest model

All metrics are written to `results/ml_metrics_and_importance.csv`. Feature importances are returned as a sorted DataFrame for plotting.

### 5. Visualization

`plotting.py` generates publication-ready figures:

* `plot_correlations_bar`
  Horizontal bar plot of Spearman's rho between slip grade and the most relevant variables.

* `plot_quartile_results`
  For each variable used in the quartile analysis, plots spondylolisthesis prevalence (bar chart) and mean slip grade (line plot) across quartiles.

* `plot_roc_curves`
  ROC curves of all machine-learning models in a single figure.

* `plot_ml_metrics`
  Bar plots comparing accuracy, sensitivity, and specificity across models.

* `plot_rf_feature_importance`
  Horizontal bar plot showing the top features ranked by Random Forest importance.

All figures are saved automatically in the `figures/` directory.

### 6. End-to-End Execution

`main.py` orchestrates the full workflow:

1. Load and clean the dataset
2. Run all statistical analyses and save tables
3. Train and evaluate all machine-learning models
4. Generate all figures

Running `python main.py` executes all steps in order and prints a message when the pipeline is complete.

---

## Installation

```bash
pip install -r requirements.txt
```

The `requirements.txt` file lists the main dependencies:

* pandas
* numpy
* scipy
* statsmodels
* scikit-learn
* matplotlib

Any recent versions that satisfy these minimum requirements should be sufficient.

---

## Usage

1. Place a CSV file containing the study data in the `data/` directory.

2. Ensure that column names match those referenced in `config.py`, or update `config.py` accordingly.

3. From the project root, run:

   ```bash
   python main.py
   ```

4. Inspect the generated tables in `results/` and figures in `figures/`.

---

## License

The code can be distributed and modified under the terms of the MIT License (see the `LICENSE` file for details).

```
```
