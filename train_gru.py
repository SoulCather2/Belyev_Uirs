"""GRU terrain classifier on windows of consecutive instant wheel measurements.

No window statistics and no IMU: a sample is the sequence of instant and derived
features (data_split.SEQUENCE_FEATURES) ending at a labelled row. Splits come from
whole 30 s blocks, so rows are never shuffled before windows are cut and a window
stays inside its block. Labelled rows are the same as for the other models, so test
scores are comparable for every window size.

Optuna tunes the window length (up to 2 s), the reading step inside the window, the
optional MLP encoder, the GRU and the optimizer. Training is shared with train_mlp.py.

    uv run tensorboard --logdir data/borealtc/gru/tensorboard
"""

import argparse

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn

from data_split import DATA_DIR, SAMPLE_RATE, load_sequences, window_offsets
from train_mlp import ACTIVATIONS, add_training_args, run_study

OUT_DIR = DATA_DIR / "gru_mlp"  # data/borealtc/gru holds the earlier run without the head MLP


class Windows:
    """Windows ending at ``rows``, cut from ``values`` on the GPU when indexed like a tensor."""

    def __init__(self, values: torch.Tensor, rows: np.ndarray, offsets: np.ndarray, flatten: bool = False):
        self.values = values
        self.rows = torch.as_tensor(rows, device=values.device)
        self.offsets = torch.as_tensor(offsets.copy(), device=values.device)
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index) -> torch.Tensor:
        windows = self.values[self.rows[index, None] + self.offsets]
        return windows.flatten(1) if self.flatten else windows

    @property
    def shape(self) -> tuple:
        n, width = len(self.offsets), self.values.shape[1]
        return (len(self), n * width) if self.flatten else (len(self), n, width)


def mlp_layers(width: int, hidden: list[int], params: dict) -> tuple[list[nn.Module], int]:
    layers = []
    for units in hidden:
        layers += [nn.Linear(width, units), ACTIVATIONS[params["activation"]](), nn.Dropout(params["dropout"])]
        width = units
    return layers, width


class GRUClassifier(nn.Module):
    """MLP encoder on every time step -> one GRU layer -> MLP head on its final state -> class logits."""

    def __init__(self, n_inputs: int, n_classes: int, params: dict):
        super().__init__()
        layers, width = mlp_layers(n_inputs, params["encoder"], params)
        self.encoder = nn.Sequential(*layers)
        self.gru = nn.GRU(width, params["gru_units"], num_layers=1, bidirectional=params["bidirectional"], batch_first=True)
        self.directions = 2 if params["bidirectional"] else 1
        layers, width = mlp_layers(params["gru_units"] * self.directions, params["head"], params)
        self.head = nn.Sequential(*layers, nn.Linear(width, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(self.encoder(x))
        return self.head(torch.cat(list(hidden[-self.directions:]), dim=1))


def suggest_params(trial: optuna.Trial) -> dict:
    encoder_layers = trial.suggest_int("encoder_layers", 1, 3)
    head_layers = trial.suggest_int("head_layers", 1, 3)
    return {
        "window_s": trial.suggest_float("window_s", 0.2, 2.0, step=0.2),
        "step": trial.suggest_categorical("step", [2, 5, 10]),  # signals are interpolated to 100 Hz and very smooth
        # every encoder layer is wider than the 16 input features, so each step is lifted to a higher dimension
        "encoder": [trial.suggest_int(f"encoder_units_{i}", 32, 512, log=True) for i in range(encoder_layers)],
        "gru_units": trial.suggest_int("gru_units", 16, 256, log=True),
        "bidirectional": trial.suggest_categorical("bidirectional", [False, True]),
        "head": [trial.suggest_int(f"head_units_{i}", 16, 512, log=True) for i in range(head_layers)],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", list(ACTIVATIONS)),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20)
    add_training_args(parser, trials=60, max_epochs=400, patience=40, warmup=60)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    s, values = load_sequences(args.stride)
    print(f"{len(s.features)} features per step: {s.features}")
    print("Surface share inside each split, %:")
    print(s.shares().round(2), "\n")
    values = torch.tensor(values, device=device)

    def prepare(params: dict) -> tuple:
        offsets = window_offsets(round(params["window_s"] * SAMPLE_RATE), params["step"])

        def make_model() -> GRUClassifier:
            return GRUClassifier(values.shape[1], len(s.encoder.classes_), params)

        return make_model, *(Windows(values, part["row"].to_numpy(), offsets) for part in (s.train, s.val, s.test))

    run_study("gru", OUT_DIR, s, device, suggest_params, prepare, args)


if __name__ == "__main__":
    main()
