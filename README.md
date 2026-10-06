# Belyev UIRS

This project analyzes vehicle motion measurements across terrain types and estimates the normalized kinetic-energy coefficient (`Ke`). The pipeline filters raw measurements, recalculates `Ke` for each filtering window, removes outliers, aggregates values on a symmetric angular-velocity grid, fits discrete cosine transform (DCT) models, and classifies terrain surfaces.

## Run

Install the project dependencies and run the main pipeline:

```powershell
python -m pip install -e .
python Major.py
```

The input dataset is expected at `data/borealtc/merged_interpolation.csv`. Reports, model artifacts, motion splits, and plots are written to `data/borealtc/`.

## Python scripts

| File | Purpose |
| --- | --- |
| `Major.py` | Main data-processing and classification pipeline. It filters raw motion signals, calculates `Ke`, removes outliers, trains DCT surface models, classifies experiments, and writes reports and plots. |
| `optimize_hyperparameters.py` | Runs Optuna hyperparameter optimization. For every trial it filters the raw signals, recalculates `Ke`, trains the DCT models, evaluates the classifier, and saves the best parameters and trial history. |
| `analyze_best_dct_models.py` | Rebuilds the DCT models using the best Optuna trial, generates analytical formulas, exports cosine coefficients, and compares the selected DCT positions between surfaces. |
| `print_dct_models.py` | Reads saved DCT coefficients and renders the analytical surface formulas as HTML with MathJax or prints the formula source to the console. |

## Formula and optimization commands

Generate the analytical formulas for the best trial:

```powershell
python analyze_best_dct_models.py
```

Open `formulas.html` to view the baseline and best-trial formulas in a browser. Run a fresh 100-trial Optuna study with:

```powershell
python optimize_hyperparameters.py --trials 100 --study-name classifier_normal_likelihood_clean
```

## Included results

The repository includes representative median `Ke` plots for ASPHALT, FLOORING, ICE, and SNOW, aggregate binned-energy visualizations, classification reports, confusion matrices, analytical DCT formulas, cosine coefficient tables, and formula similarity summaries.

## Repository files

### Project and configuration files

| File | Description |
| --- | --- |
| `.gitignore` | Lists local environments, Python caches, and Optuna runtime files that should not be committed. |
| `README.md` | Project documentation, setup instructions, script descriptions, and result-file documentation. |
| `pyproject.toml` | Project metadata, Python version requirement, and runtime dependencies, including Optuna. |
| `uv.lock` | Locked dependency versions for reproducible installation with `uv`. |
| `formulas.html` | Browser-renderable MathJax page containing the baseline and best-trial analytical DCT formulas. |

### Python source files

| File | Description |
| --- | --- |
| `Major.py` | Defines the complete feature, filtering, DCT-model, classifier, reporting, and plotting pipeline. It filters raw signals, recalculates `Ke`, trains surface models, classifies experiments, and writes artifacts. |
| `optimize_hyperparameters.py` | Runs Optuna trials over filtering window size, angular-velocity bin count, DCT coefficient count, and classifier memory. Each trial recalculates `Ke` after filtering and evaluates the classifier. |
| `analyze_best_dct_models.py` | Rebuilds models from the best Optuna configuration, exports analytical formulas, writes cosine coefficients, and compares selected DCT positions across surfaces. |
| `print_dct_models.py` | Reads saved DCT coefficient tables and prints analytical formulas or renders them as an HTML page. It does not create LaTeX files. |

### Classification and model reports

| File | Description |
| --- | --- |
| `data/borealtc/classification_report.txt` | Human-readable precision, recall, F1-score, and accuracy report. |
| `data/borealtc/classification_report.csv` | Tabular version of the classification metrics. |
| `data/borealtc/dct_confusion_matrix.png` | Visual row-normalized confusion matrix for terrain classification. |
| `data/borealtc/dct_confusion_matrix.csv` | Row-normalized confusion matrix values. |
| `data/borealtc/dct_confusion_matrix_counts.csv` | Raw confusion-matrix counts. |

### DCT formulas and coefficient analysis

| File | Description |
| --- | --- |
| `data/borealtc/best_dct_cosine_coefficients.txt` | Human-readable nonzero coefficients multiplying cosine terms for the best trial. |
| `data/borealtc/best_dct_formula_similarity.txt` | Comparison of selected DCT positions and overlap between ASPHALT, FLOORING, ICE, and SNOW formulas. |

### Visual results

| File | Description |
| --- | --- |
| `data/borealtc/all_terrains_binned_ke_no_dct.png` | Aggregated binned `Ke` curves across all terrains without DCT curves. |
| `data/borealtc/all_terrains_binned_ke_no_std.png` | Aggregated binned `Ke` curves across all terrains without standard-deviation bands. |
| `data/borealtc/median_ke_plots/ASPHALT_median_ke.png` | Median `Ke` curve and DCT model for ASPHALT. |
| `data/borealtc/median_ke_plots/FLOORING_median_ke.png` | Median `Ke` curve and DCT model for FLOORING. |
| `data/borealtc/median_ke_plots/ICE_median_ke.png` | Median `Ke` curve and DCT model for ICE. |
| `data/borealtc/median_ke_plots/SNOW_median_ke.png` | Median `Ke` curve and DCT model for SNOW. |

The raw input file `data/borealtc/merged_interpolation.csv` is required for a local pipeline run but is not stored in this repository.