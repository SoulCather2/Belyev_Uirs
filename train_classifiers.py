"""Terrain classification from wheel signals only (no IMU) with Optuna-tuned models.

Features use wheel currents and wheel speeds plus the motion they imply:
overall speed, turn rate (direction) and their amplitudes over trailing windows
(see data_split.py). Every model trains on the train split, Optuna picks its
hyperparameters by macro-F1 on the val split, and the chosen model is scored
once on the test split. The split and the scaling are shared with all models.
"""

import argparse
import json
import time

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
from sklearn.tree import DecisionTreeClassifier

from data_split import DATA_DIR, SEED, Splits, load_dataset, make_splits
from Major import draw_confusion_matrix

OUT_DIR = DATA_DIR / "ml_classifiers"


# ---------------------------------------------------------------- models

def make_tree(trial: optuna.Trial):
    return DecisionTreeClassifier(
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]),
        max_depth=trial.suggest_int("max_depth", 3, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 1000, log=True),
        max_features=trial.suggest_categorical("max_features", [None, "sqrt", 0.5]),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        random_state=SEED,
    )


def make_forest(trial: optuna.Trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 400, step=50),
        max_depth=trial.suggest_int("max_depth", 4, 30),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 500, log=True),
        max_features=trial.suggest_float("max_features", 0.1, 0.8),
        max_samples=trial.suggest_float("max_samples", 0.3, 1.0),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"]),
        n_jobs=-1,
        random_state=SEED,
    )


def make_logreg(trial: optuna.Trial):
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        class_weight=trial.suggest_categorical("class_weight", [None, "balanced"]),
        max_iter=3000,
    )


def make_catboost(trial: optuna.Trial):
    return CatBoostClassifier(
        iterations=2000,
        early_stopping_rounds=100,
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        depth=trial.suggest_int("depth", 4, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
        random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
        bootstrap_type="Bernoulli",
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        auto_class_weights=trial.suggest_categorical("auto_class_weights", ["None", "Balanced"]),
        loss_function="MultiClass",
        thread_count=-1,
        random_seed=SEED,
        verbose=False,
    )


MODELS = {
    "decision_tree": make_tree,
    "random_forest": make_forest,
    "logistic_regression": make_logreg,
    "catboost": make_catboost,
}
TRIALS = {"decision_tree": 100, "logistic_regression": 50, "random_forest": 50, "catboost": 30}


def fit(name: str, trial, s: Splits):
    model = MODELS[name](trial)
    if name == "catboost":
        model.fit(s.x_train, s.y_train, eval_set=(s.x_val, s.y_val))
    else:
        model.fit(s.x_train, s.y_train)
    return model


def predict(model, x) -> np.ndarray:
    return np.asarray(model.predict(x)).ravel().astype(int)


def tune(name: str, s: Splits, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        model = fit(name, trial, s)
        if name == "catboost":
            trial.set_user_attr("best_iteration", model.get_best_iteration() + 1)
        return f1_score(s.y_val, predict(model, s.x_val), average="macro")

    study = optuna.create_study(study_name=name, direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
        callbacks=[lambda study, trial: print(
            f"  {name} trial {trial.number + 1}/{n_trials}: val macro-F1 {trial.value:.3f}, best {study.best_value:.3f}",
            flush=True,
        )],
    )
    return study


def importances(name: str, model, features: list[str]) -> pd.Series:
    if name == "logistic_regression":
        values = np.abs(model.coef_).mean(axis=0)
    elif name == "catboost":
        values = model.get_feature_importance()
    else:
        values = model.feature_importances_
    return pd.Series(values / values.sum(), index=features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20, help="keep every n-th sample (100 Hz is highly autocorrelated)")
    parser.add_argument("--trials", type=int, default=None, help="trials per model (default: TRIALS)")
    parser.add_argument("--models", nargs="+", default=list(TRIALS), choices=list(MODELS))
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    out = type(OUT_DIR)(args.out)
    out.mkdir(parents=True, exist_ok=True)

    s = make_splits(load_dataset(args.stride))
    shares = s.shares()
    shares.to_csv(out / "split_shares.csv")
    sizes = pd.Series({"train": len(s.train), "val": len(s.val), "test": len(s.test)}, name="rows")
    print(f"{len(s.features)} features, stride {args.stride}, seed {SEED}")
    print(pd.DataFrame({"rows": sizes, "share_%": sizes / sizes.sum() * 100}).round(1))
    print("\nSurface share inside each split, %:")
    print(shares.round(2), "\n")

    labels = s.encoder.classes_
    summary, per_terrain = [], {}
    for name in args.models:
        started = time.time()
        study = tune(name, s, args.trials or TRIALS[name])
        study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)
        model = fit(name, optuna.trial.FixedTrial(study.best_params), s)
        val_pred, test_pred = predict(model, s.x_val), predict(model, s.x_test)

        report = pd.DataFrame(classification_report(
            s.y_test, test_pred, labels=range(len(labels)), target_names=labels, output_dict=True, zero_division=0
        )).T
        per_terrain[name] = report.loc[labels, "recall"]
        report.to_csv(out / f"{name}_classification_report.csv")
        (out / f"{name}_classification_report.txt").write_text(classification_report(
            s.y_test, test_pred, labels=range(len(labels)), target_names=labels, digits=3, zero_division=0
        ), encoding="utf-8")
        confusion = pd.crosstab(
            pd.Series(s.encoder.inverse_transform(s.y_test), name="terrain"),
            pd.Series(s.encoder.inverse_transform(test_pred), name="predicted"),
        ).reindex(index=labels, columns=labels, fill_value=0)
        figure = draw_confusion_matrix(confusion, f"{name}: test split, per measurement")
        figure.savefig(out / f"{name}_confusion.png", dpi=150)
        plt.close(figure)
        importances(name, model, s.features).sort_values(ascending=False).to_csv(
            out / f"{name}_importance.csv", header=["importance"]
        )
        params = dict(study.best_params)
        if name == "catboost":
            params["iterations"] = study.best_trial.user_attrs["best_iteration"]
        (out / f"{name}_best_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
        summary.append({
            "model": name,
            "val_macro_f1": f1_score(s.y_val, val_pred, average="macro"),
            "test_macro_f1": f1_score(s.y_test, test_pred, average="macro"),
            "test_accuracy": accuracy_score(s.y_test, test_pred),
            "trials": len(study.trials),
            "minutes": (time.time() - started) / 60,
        })
        print(pd.DataFrame(summary).set_index("model").round(3).tail(1), "\n", params, "\n", flush=True)

    summary = pd.DataFrame(summary).set_index("model")
    recall = pd.DataFrame(per_terrain).T
    summary.to_csv(out / "summary.csv")
    recall.to_csv(out / "recall_by_terrain.csv")
    print(summary.round(3))
    print("\nRecall by terrain on the test split:")
    print(recall.round(3))


if __name__ == "__main__":
    main()
