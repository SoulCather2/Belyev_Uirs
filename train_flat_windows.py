"""Trees and a plain MLP on the GRU's windows, stretched into one vector per sample.

Uses the window length, reading step, features, rows and split of the tuned GRU
(data/borealtc/gru/gru_best_params.json), so the only difference from the GRU is
that the order of measurements is not built into the model.
"""

import argparse
import json
from dataclasses import replace

import numpy as np
import optuna
import pandas as pd
import torch

from data_split import DATA_DIR, SAMPLE_RATE, load_sequences, window_offsets
from train_classifiers import run_models
GRU_DIR = DATA_DIR / "gru"  # window of the first GRU run
from train_gru import Windows
from train_mlp import MLP, add_training_args, run_study, suggest_params

OUT_DIR = DATA_DIR / "flat_windows"
TRIALS = {"decision_tree": 100, "random_forest": 50, "catboost": 30}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--models", nargs="+", default=[*TRIALS, "mlp"], choices=[*TRIALS, "mlp"])
    add_training_args(parser, trials=60, max_epochs=1000, patience=60, warmup=100)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gru = json.loads((GRU_DIR / "gru_best_params.json").read_text(encoding="utf-8"))
    offsets = window_offsets(round(gru["window_s"] * SAMPLE_RATE), gru["step"])
    s, values = load_sequences(args.stride)
    print(f"window {gru['window_s']} s read every {gru['step']} samples: {len(offsets)} steps x {len(s.features)} features")

    trees = [name for name in args.models if name in TRIALS]
    if trees:
        columns = [f"{feature}_t{offset * 10}ms" for offset in offsets for feature in s.features]
        x_train, x_val, x_test = (
            values[part["row"].to_numpy()[:, None] + offsets].reshape(len(part), -1) for part in (s.train, s.val, s.test)
        )
        run_models(replace(s, x_train=x_train, x_val=x_val, x_test=x_test, features=columns), trees, TRIALS, OUT_DIR)

    if "mlp" in args.models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(values, device=device)
        inputs = [Windows(tensor, part["row"].to_numpy(), offsets, flatten=True) for part in (s.train, s.val, s.test)]

        def prepare(params: dict) -> tuple:
            def make_model() -> MLP:
                return MLP(inputs[0].shape[1], len(s.encoder.classes_), params["hidden"], params["dropout"], params["activation"])

            return make_model, *inputs

        run_study("mlp", OUT_DIR, s, device, suggest_params, prepare, args)


if __name__ == "__main__":
    main()
