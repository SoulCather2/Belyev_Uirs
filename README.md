# Belyev UIRS

This project analyzes vehicle motion measurements across terrain types and estimates the normalized kinetic-energy coefficient (`Ke`). The pipeline filters raw measurements, recalculates `Ke` for each filtering window, removes outliers, aggregates values on a symmetric angular-velocity grid, fits discrete cosine transform (DCT) models, and classifies terrain surfaces.

## Project stage

Earlier stages prepared the filtered motion data, DCT surface models, analytical formulas, Optuna search, and the likelihood-based classifier with probability memory. The current stage classifies terrain from wheel signals only (no IMU) with tuned sklearn models, CatBoost, an MLP and a GRU with MLP encoder and head on windows of consecutive measurements. See [Wheel-only classifiers](#wheel-only-classifiers).

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
| `train_mlp.py` | MLP classifier in pure PyTorch on the same split and features: AdamW, ReduceLROnPlateau, cross-entropy, best-epoch selection by val macro-F1, Optuna over layer count, units per layer, dropout, activation, learning rate, weight decay and class weights. The batch is the largest that fits in 70% of free video memory; it is any integer, found by doubling and a binary search, not a power of two. A trial that trains longer than `--trial-minutes` (default 10) is pruned. Logs to TensorBoard (`uv run tensorboard --logdir data/borealtc/mlp/tensorboard`) and writes results to `data/borealtc/mlp/`. PyTorch is installed from the CUDA 12.8 index (needed for RTX 50xx GPUs). `--features instant` uses only the 10 instant features and writes to `data/borealtc/mlp_instant/`. |
| `train_gru.py` | GRU classifier on windows of consecutive instant and derived features (no window statistics, no IMU): an MLP encoder lifts every time step to a higher dimension, one GRU layer reads the window, and an MLP head turns its final state into class logits. Windows are cut inside the 30 s blocks and end at the same labelled rows as the other models. Optuna tunes the encoder and head layers and units, the window length (up to 2 s), the reading step, the GRU size and direction and the optimizer; the GRU always has one layer. Writes to `data/borealtc/gru_mlp/`. |
| `train_flat_windows.py` | Decision tree, random forest, CatBoost and a plain MLP on the best window of the first GRU run (`data/borealtc/gru/gru_best_params.json`) stretched into one vector. Writes to `data/borealtc/flat_windows/`. |
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

## Wheel-only classifiers

Install the dependencies with `uv sync`, then run:

```powershell
uv run python train_classifiers.py             # tree, forest, logistic regression, CatBoost
uv run python train_mlp.py                     # MLP on all 38 features
uv run python train_mlp.py --features instant  # MLP on instant features only
uv run python train_gru.py --trials 80 --timeout 10800  # GRU with MLP encoder and head, about 30 minutes on an RTX 5060
uv run python train_flat_windows.py            # trees and MLP on the first GRU's stretched window
uv run tensorboard --logdir data/borealtc/gru_mlp/tensorboard
```

Results on the shared test split (6056 rows from 48 blocks). The first GRU (no MLP head, up to 3 GRU layers) chose a 0.8 s window read every 5th sample, so 16 steps of 16 features; the stretched-window models use the same window.

| Model | Input | Test macro-F1 | Test accuracy |
| --- | --- | --- | --- |
| MLP | 38 features with 1 s / 3 s window statistics | 0.871 | 0.914 |
| CatBoost | 38 features | 0.863 | 0.904 |
| GRU + MLP, 9 trials | 2.0 s window, 40 steps | 0.857 | 0.900 |
| MLP | stretched 0.8 s window | 0.846 | 0.889 |
| Random forest | 38 features | 0.837 | 0.883 |
| CatBoost | stretched 0.8 s window | 0.832 | 0.883 |
| GRU, first run | 0.8 s window, 16 steps | 0.831 | 0.893 |
| GRU + MLP, 80 trials | 1.8 s window, 36 steps | 0.831 | 0.881 |
| Decision tree | 38 features | 0.815 | 0.862 |
| Random forest | stretched 0.8 s window | 0.807 | 0.856 |
| MLP | 10 instant features | 0.805 | 0.852 |
| Decision tree | stretched 0.8 s window | 0.798 | 0.857 |
| Logistic regression | 38 features | 0.707 | 0.754 |

GRU + MLP was tuned twice. The first search stopped after 9 trials: one trial filled 90% of video memory, Windows moved it to shared RAM and that trial took 2 hours. After lowering the memory share to 70% and adding the per-trial time limit, the full search ran all 80 trials in 26 minutes. It reached a higher val macro-F1 (0.973 against 0.953) but a lower test macro-F1 (0.831 against 0.857): the val split has only 24 blocks, so with many trials the search fits the val blocks. A block bootstrap puts the difference at 0.026 with a 95% interval from -0.001 to 0.053, so neither GRU + MLP run clearly beats the plain MLP on the stretched window.

Each result folder holds the classification report (`*_classification_report.txt` / `.csv`), the confusion matrix with percents (`*_confusion.png`), best parameters (`*_best_params.json`), Optuna trials (`*_trials.csv`) and the summary. Networks also have weights (`*_best.pt`), per-epoch history (`*_history.csv`) and TensorBoard logs.

| Folder | Contents |
| --- | --- |
| `data/borealtc/ml_classifiers/` | Tree, forest, logistic regression and CatBoost on 38 features |
| `data/borealtc/mlp/` | MLP on 38 features |
| `data/borealtc/mlp_instant/` | MLP on instant features |
| `data/borealtc/gru/` | First GRU run: optional encoder, 1–3 GRU layers, linear output |
| `data/borealtc/gru_mlp/` | GRU with MLP encoder and head, full 80-trial search |
| `data/borealtc/gru_mlp_9trials/` | GRU with MLP encoder and head, first search stopped after 9 trials |
| `data/borealtc/flat_windows/` | Tree, forest, CatBoost and MLP on the stretched window |

`data/borealtc/error_analysis/samples_with_errors.csv.gz` (111 MB, written by `Major.py`) and the `catboost_info/` training cache are not stored in the repository.

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