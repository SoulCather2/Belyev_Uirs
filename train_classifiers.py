"""Terrain classification from wheel signals only (no IMU) with Optuna-tuned models.

Features use wheel currents and wheel speeds plus the motion they imply:
overall speed, turn rate (direction) and their amplitudes over trailing windows.
Runs are never split between folds, so a model cannot memorize a run. Every
model is tuned and scored with the same 5-fold run-grouped cross-validation and
its out-of-fold predictions give the confusion matrix (each run is predicted by
a model that never saw it).
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from Major import PipelineConfig, draw_confusion_matrix

DATA_DIR = Path(__file__).parent / "data" / "borealtc"
OUT_DIR = DATA_DIR / "ml_classifiers"
SAMPLE_RATE = 100  # Hz, time step is 0.01 s
WINDOWS = {"1s": 1 * SAMPLE_RATE, "3s": 3 * SAMPLE_RATE}
VEL_LIMIT = 1.0  # m/s, |vel| above this are logging glitches (99.9% quantile is 0.79)
CUR_LIMIT = 40.0  # A, 99.9% quantile is ~24, the rest are spikes up to 1900


def build_features(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Per-sample wheel features plus trailing-window statistics computed inside every run."""
    d = data[["terrain", "run_idx", "time", "velL", "velR", "curL", "curR"]].copy()
    d[["velL", "velR"]] = d[["velL", "velR"]].clip(-VEL_LIMIT, VEL_LIMIT)
    d[["curL", "curR"]] = d[["curL", "curR"]].clip(-CUR_LIMIT, CUR_LIMIT)
    d["speed"] = (d["velL"] + d["velR"]) / 2
    d["omega"] = (d["velR"] - d["velL"]) / config.track_width
    d["abs_speed"] = d["speed"].abs()
    d["abs_omega"] = d["omega"].abs()
    d["cur_sum"] = d["curL"] + d["curR"]
    d["cur_diff"] = d["curL"] - d["curR"]

    groups = d.groupby(["terrain", "run_idx"], sort=False)
    for name, size in WINDOWS.items():
        for column in ["curL", "curR", "cur_sum", "cur_diff", "speed", "omega"]:
            rolling = groups[column].rolling(size, min_periods=1)
            d[f"{column}_mean_{name}"] = rolling.mean().reset_index(level=[0, 1], drop=True)
            d[f"{column}_std_{name}"] = rolling.std().reset_index(level=[0, 1], drop=True).fillna(0)
        for column in ["speed", "omega"]:
            high = groups[column].rolling(size, min_periods=1).max().reset_index(level=[0, 1], drop=True)
            low = groups[column].rolling(size, min_periods=1).min().reset_index(level=[0, 1], drop=True)
            d[f"{column}_amp_{name}"] = high - low
    d["run"] = d["terrain"] + "_" + d["run_idx"].astype(str)
    return d


def feature_columns(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c not in {"terrain", "run_idx", "time", "run"}]


# ---------------------------------------------------------------- models

def make_tree(trial: optuna.Trial):
    return DecisionTreeClassifier(
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 1000, log=True),
        max_features=trial.suggest_categorical("max_features", [None, "sqrt", 0.5]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=0,
    )


def make_forest(trial: optuna.Trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 4, 24),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 500, log=True),
        max_features=trial.suggest_float("max_features", 0.1, 0.8),
        max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        n_jobs=-1,
        random_state=0,
    )


def make_logreg(trial: optuna.Trial):
    scaler = trial.suggest_categorical("scaler", ["standard", "quantile"])
    transform = (
        StandardScaler() if scaler == "standard"
        else QuantileTransformer(output_distribution="normal", n_quantiles=500, random_state=0)
    )
    model = LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        max_iter=3000,
    )
    return make_pipeline(transform, model)


def make_catboost(trial: optuna.Trial):
    return CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
        bootstrap_type="Bernoulli",
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        auto_class_weights=trial.suggest_categorical("auto_class_weights", ["None", "Balanced"]),
        loss_function="MultiClass",
        thread_count=-1,
        random_seed=0,
        verbose=False,
    )


MODELS = {
    "decision_tree": make_tree,
    "random_forest": make_forest,
    "logistic_regression": make_logreg,
    "catboost": make_catboost,
}


def predict(model, x) -> np.ndarray:
    return np.asarray(model.predict(x)).ravel()


# ---------------------------------------------------------------- tuning

def cross_validate(name, trial, x, y, folds, report: bool = False) -> tuple[list[float], np.ndarray]:
    """Macro-F1 per fold and out-of-fold predictions; prunes hopeless trials after each fold."""
    scores, oof = [], np.empty(len(y), dtype=object)
    for step, (fit_idx, valid_idx) in enumerate(folds):
        model = MODELS[name](trial)
        model.fit(x.iloc[fit_idx], y[fit_idx])
        oof[valid_idx] = predict(model, x.iloc[valid_idx])
        scores.append(f1_score(y[valid_idx], oof[valid_idx], average="macro"))
        if report:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores, oof


def tune(name, x, y, folds, n_trials, seed=0) -> optuna.Study:
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(
        lambda trial: float(np.mean(cross_validate(name, trial, x, y, folds, report=True)[0])),
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[lambda study, trial: print(
            f"  {name} trial {trial.number + 1}/{n_trials}: {trial.state.name.lower()}, best {study.best_value:.3f}",
            flush=True,
        )],
    )
    return study


def importances(name: str, model, features: list[str]) -> pd.Series:
    if name == "logistic_regression":
        values = np.abs(model[-1].coef_).mean(axis=0)
    elif name == "catboost":
        values = model.get_feature_importance()
    else:
        values = model.feature_importances_
    return pd.Series(values / values.sum(), index=features)


TRIALS = {"decision_tree": 60, "logistic_regression": 40, "random_forest": 30, "catboost": 30}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20, help="keep every n-th sample (100 Hz is highly autocorrelated)")
    parser.add_argument("--trials", type=int, default=None, help="trials per model (default: TRIALS)")
    parser.add_argument("--models", nargs="+", default=list(TRIALS), choices=list(MODELS))
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA_DIR / "merged_interpolation.csv")
    raw = raw[raw["terrain"] != "SANDY_LOAM"]
    data = build_features(raw, PipelineConfig())
    data = data.groupby("run", sort=False).nth(slice(None, None, args.stride)).reset_index(drop=True)
    features = feature_columns(data)
    x, y = data[features], data["terrain"].to_numpy()

    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(x, y, groups=data["run"]))
    split = pd.DataFrame(
        [data.iloc[valid].groupby("terrain")["run"].nunique().rename(f"fold{i}") for i, (_, valid) in enumerate(folds)]
    ).fillna(0).astype(int)
    split.to_csv(OUT_DIR / "fold_runs.csv")
    print(f"{len(features)} features, stride {args.stride}, {len(data)} rows\nValidation runs per fold:\n{split}\n")

    summary, per_terrain = [], {}
    for name in args.models:
        started = time.time()
        study = tune(name, x, y, folds, args.trials or TRIALS[name])
        study.trials_dataframe().to_csv(OUT_DIR / f"{name}_trials.csv", index=False)
        scores, oof = cross_validate(name, optuna.trial.FixedTrial(study.best_params), x, y, folds)

        report = pd.DataFrame(classification_report(y, oof, output_dict=True, zero_division=0)).T
        per_terrain[name] = report.loc[sorted(set(y)), "recall"]
        confusion = pd.crosstab(pd.Series(y, name="terrain"), pd.Series(oof, name="predicted"))
        figure = draw_confusion_matrix(confusion, f"{name}: out-of-fold, per measurement")
        figure.savefig(OUT_DIR / f"{name}_confusion.png", dpi=150)
        plt.close(figure)
        model = MODELS[name](optuna.trial.FixedTrial(study.best_params)).fit(x, y)
        importances(name, model, features).sort_values(ascending=False).to_csv(
            OUT_DIR / f"{name}_importance.csv", header=["importance"]
        )
        (OUT_DIR / f"{name}_best_params.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
        summary.append({
            "model": name,
            "macro_f1": f1_score(y, oof, average="macro"),
            "fold_f1_mean": np.mean(scores),
            "fold_f1_std": np.std(scores),
            "accuracy": accuracy_score(y, oof),
            "trials": len(study.trials),
            "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
            "minutes": (time.time() - started) / 60,
        })
        print(pd.DataFrame(summary).set_index("model").round(3).tail(1), "\n", study.best_params, "\n", flush=True)

    summary = pd.DataFrame(summary).set_index("model")
    recall = pd.DataFrame(per_terrain).T
    summary.to_csv(OUT_DIR / "summary.csv")
    recall.to_csv(OUT_DIR / "recall_by_terrain.csv")
    print(summary.round(3))
    print("\nRecall by terrain (out-of-fold):")
    print(recall.round(3))


if __name__ == "__main__":
    main()
