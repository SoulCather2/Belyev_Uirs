"""Optimize the classifier hyperparameters with Optuna."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score

from Major import (
    PipelineConfig,
    build_symmetric_grid,
    calculate_features,
    calculate_dct_coefficients,
    classify_experiments_by_dct,
    filter_by_experiment,
    remove_ke_outliers,
    split_by_motion,
)


DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "borealtc"
STUDY_NAME = "dct_hyperparameter_search_v1"


def nearest_odd(value: float, minimum: int, maximum: int) -> int:
    """Map a continuous trial value to a valid odd integer."""
    integer = int(round(value))
    integer = min(max(integer, minimum), maximum)
    if integer % 2 == 0:
        integer += 1 if integer < maximum else -1
    return integer


def effective_parameters(trial: optuna.Trial) -> dict[str, int | float]:
    """Convert continuous Optuna suggestions into pipeline values."""
    window_size = max(3, int(round(trial.suggest_float("window_size", 3.0, 41.0))))
    n_bins = nearest_odd(trial.suggest_float("n_bins", 11.0, 101.0), 11, 101)
    n_dct_coefficients = max(
        2,
        int(round(trial.suggest_float("n_dct_coefficients", 2.0, 30.0))),
    )
    memory_margin = trial.suggest_float("memory_margin", 0.0, 0.5)
    n_dct_coefficients = min(n_dct_coefficients, n_bins)
    return {
        "window_size": window_size,
        "n_bins": n_bins,
        "n_dct_coefficients": n_dct_coefficients,
        "memory_margin": memory_margin,
    }


def build_objective(raw_data: pd.DataFrame) -> callable:
    """Build an objective with feature calculation shared by every trial."""
    base_config = PipelineConfig()
    features = calculate_features(raw_data, base_config)

    def objective(trial: optuna.Trial) -> float:
        parameters = effective_parameters(trial)
        config = replace(base_config, **parameters)
        grid, omega_limit = build_symmetric_grid(features, config)
        filtered = filter_by_experiment(features, config)
        cleaned = remove_ke_outliers(filtered, config)

        split_data = split_by_motion(cleaned, grid, omega_limit)
        training_data = pd.concat(split_data.values(), ignore_index=True)
        _, _, models, models_std = calculate_dct_coefficients(
            training_data, config, grid, omega_limit
        )
        classification = classify_experiments_by_dct(
            cleaned, models, models_std, config
        )
        if classification.empty:
            return 0.0

        accuracy = float(
            accuracy_score(
                classification["terrain"], classification["predicted_terrain"]
            )
        )
        for name, value in parameters.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("classified_experiments", len(classification))
        return accuracy

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize DCT classifier hyperparameters with Optuna."
    )
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None, help="Maximum runtime in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--study-name", default=STUDY_NAME)
    args = parser.parse_args()

    input_path = args.data_dir / "merged_interpolation.csv"
    raw_data = pd.read_csv(input_path)
    raw_data = raw_data[raw_data["terrain"] != "SANDY_LOAM"].copy()

    storage_path = args.data_dir / "optuna_dct.db"
    storage = f"sqlite:///{storage_path.as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(
        build_objective(raw_data),
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
    )

    best_trial = study.best_trial
    best_result = {
        "accuracy": best_trial.value,
        "trial": best_trial.number,
        "parameters": best_trial.user_attrs,
        "suggestions": best_trial.params,
    }
    (args.data_dir / "best_dct_hyperparameters.json").write_text(
        json.dumps(best_result, indent=2), encoding="utf-8"
    )
    study.trials_dataframe().to_csv(
        args.data_dir / "dct_hyperparameter_trials.csv", index=False
    )
    print(json.dumps(best_result, indent=2))


if __name__ == "__main__":
    main()
