# Belyev UIRS

This project analyzes vehicle motion measurements across terrain types and estimates the normalized kinetic-energy coefficient (`Ke`). The pipeline filters raw measurements, recalculates `Ke` for each filtering window, removes outliers, aggregates values on a symmetric angular-velocity grid, fits discrete cosine transform (DCT) models, and classifies terrain surfaces.

## Project stage

Earlier stages prepared the filtered motion data, DCT surface models, analytical formulas, and Optuna search. The current stage evaluates the likelihood-based classifier with probability memory on filtered experiments and documents the resulting metrics and model formulas.

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
| `plot_errors.py` | Standalone plotting module (numpy, pandas, plotly only): `plot_scatter_with_errors` shades classifier-error intervals; `plot_run` plots one run of the predictions dataset, one panel per signal. |
| `export_predictions.py` | Classifies every cleaned measurement with the saved DCT models and writes `data/borealtc/dataset_with_predictions.csv.gz` (all original columns plus `predicted_terrain`, `raw_predicted_terrain`, `wrong`, `raw_wrong`). |
| `data_split.py` | Builds the wheel-only feature set (no IMU): wheel currents, wheel speeds, overall speed and turn rate with 1 s / 3 s trailing-window means, deviations and amplitudes. `make_splits` cuts every run into 30 s blocks (the first 3 s of each block are dropped so trailing windows never cross blocks), splits whole blocks 0.7 / 0.1 / 0.2 into train / val / test with `train_test_split` stratified by surface (seed 42), and fits `StandardScaler` and `LabelEncoder` on train only. Every model, sklearn or pytorch, must use it. |
| `train_classifiers.py` | Tunes a decision tree, random forest, logistic regression and CatBoost with Optuna (fit on train, macro-F1 on val), scores the chosen models once on test, and writes split shares, metrics, confusion matrices, feature importances and best parameters to `data/borealtc/ml_classifiers/`. Earlier results are kept for comparison: run-grouped 5-fold CV in `data/borealtc/ml_classifiers_grouped_cv/` and the leaky row-level split in `data/borealtc/ml_classifiers_row_split/`. |
| `train_mlp.py` | MLP classifier in pure PyTorch on the same split and features: AdamW, ReduceLROnPlateau, cross-entropy, best-epoch selection by val macro-F1, Optuna over layer count, units per layer, dropout, activation, learning rate, weight decay and class weights. The batch is the largest that fits in video memory. Logs to TensorBoard (`uv run tensorboard --logdir data/borealtc/mlp/tensorboard`) and writes results to `data/borealtc/mlp/`. PyTorch is installed from the CUDA 12.8 index (needed for RTX 50xx GPUs). |
| `compare_dct_positions.py` | Lists nonzero DCT coefficient indices for every surface in the baseline and best models, groups identical index sets, and generates green/red pairwise comparison matrices. |

## Interactive scatter plots

`plot_errors.plot_scatter_with_errors` wraps `plotly.express.scatter` and shades
precomputed classifier-error intervals in red. Pass ordinary Plotly Express
options through keyword arguments (the template defaults to `plotly_white`);
`y` can be one column or a list of columns:

```python
from plot_errors import plot_scatter_with_errors

figure = plot_scatter_with_errors(
    data,
    x="time",
    y=["robot_omega", "robot_vel"],
    wrong_cls="wrong_cls",
)
figure.show()
```

The error column must contain booleans or 0/1 values; missing values are
treated as correct. The function does not calculate classification results.
The x column must be numeric or datetime-like so error intervals can be
located. Error bands are drawn as filled traces in one legend group, so one
legend click hides or shows them on every facet panel. With `share_y=False`
facet panels get independent y scales (used by `plot_run`). Margins are kept
small and the legend sits above the plot, so the chart uses the full width.

### Full dataset with predictions

After `python Major.py` has written the cleaned data and `dct_models.pkl`, build
the per-measurement predictions dataset (about 520k rows, ~55 MB, git-ignored):

```powershell
python export_predictions.py                 # memory carried between runs, as in Major.py
python export_predictions.py --reset-memory  # memory reset at the start of every run
```

Then plot any run with its errors shaded:

```python
import pandas as pd
from plot_errors import plot_run

data = pd.read_csv("data/borealtc/dataset_with_predictions.csv.gz")
plot_run(data, "SNOW", 2, y=["omega", "Ke", "speed"]).show()
```

## Classifier error analysis

`Major.py` also contains helpers that explain where and why the DCT classifier
makes mistakes:

| Function | Purpose |
| --- | --- |
| `classify_samples_by_dct` | Classifies every measurement of every experiment and adds predicted terrains, raw and memory probabilities, and the `wrong` / `raw_wrong` error flags. `reset_memory=True` resets the probability memory at the start of each run. |
| `add_error_features` | Adds per-run features used to explain errors: `abs_omega`, turn `direction`, mean and differential motor current, `abs_speed`, rolling `omega` `amplitude`, and relative position inside the run (`run_pos`). |
| `error_by_feature` | Error rate inside equal-count bins of each feature, overall and per true terrain. |
| `model_separation` | Distance between every pair of surface models, in units of their pooled spread, along the common `omega` range. |
| `pick_example_runs` | Selects the worst run and runs with a mix of correct and wrong predictions for every terrain. |
| `save_example_plots` | Saves interactive HTML plots of the selected runs (one panel per signal) with classifier-error intervals shaded in red. |
| `draw_confusion_matrix` | Draws a row-normalized confusion matrix; every cell shows the percent of its true terrain and the count. |
| `run_error_analysis` | Runs the whole analysis and writes the CSV summaries (`1_error_by_terrain.csv` … `7_model_separation.csv`, `samples_with_errors.csv.gz`), the per-measurement confusion matrix `2_confusion_per_sample.png`, and example plots to `data/borealtc/error_analysis/`. |

The example plots are committed in `data/borealtc/error_analysis/example_plots/`
(`worst_<TERRAIN>_run<N>.html` and `mixed_<TERRAIN>_run<N>.html`).

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
| `data/borealtc/dct_confusion_matrix.png` | Row-normalized confusion matrix for terrain classification; cells show percents of the true terrain and counts. |
| `data/borealtc/dct_confusion_matrix.csv` | Row-normalized confusion matrix values. |
| `data/borealtc/dct_confusion_matrix_counts.csv` | Raw confusion-matrix counts. |

### DCT formulas and coefficient analysis

| File | Description |
| --- | --- |
| `data/borealtc/best_dct_cosine_coefficients.txt` | Human-readable nonzero coefficients multiplying cosine terms for the best trial. |
| `data/borealtc/best_dct_formula_similarity.txt` | Comparison of selected DCT positions and overlap between ASPHALT, FLOORING, ICE, and SNOW formulas. |
| `data/borealtc/dct_position_comparison_by_surface.txt` | Sorted nonzero coefficient indices per surface and groups of surfaces with identical index sets for the baseline and best models. |
| `data/borealtc/dct_position_matrix_baseline.html` | Plotly pairwise surface matrix for baseline coefficient-index sets (green = identical, red = different); hover shows the XOR count and differing coefficient indices. |
| `data/borealtc/dct_position_matrix_best.html` | Plotly pairwise surface matrix for best-model coefficient-index sets (green = identical, red = different); hover shows the XOR count and differing coefficient indices. |
| `data/borealtc/dct_position_matrix_baseline_vs_best.html` | Plotly cross-model matrix: baseline surfaces by row, best-model surfaces by column; hover shows the XOR count and differing coefficient indices. |
| `data/borealtc/dct_coefficient_position_matrix.html` | Per-surface baseline-versus-best matrix across coefficient indices (green = same selection state, red = different). |

Regenerate the per-surface index lists and comparison matrices with:

```powershell
python compare_dct_positions.py
```

The pairwise Plotly matrices are saved as standalone HTML files that load Plotly from its CDN. Each cell shows the size of the symmetric difference (XOR) between the two surfaces' coefficient-index sets; hover over a cell to see the count, the XOR indices, and which indices occur only in each surface.

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