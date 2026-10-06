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