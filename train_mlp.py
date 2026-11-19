"""MLP terrain classifier on wheel-only features (pure PyTorch) tuned with Optuna.

Uses the same split, scaler and label encoder as the sklearn models (data_split.py).
Training: AdamW, ReduceLROnPlateau on the val loss, cross-entropy loss. The weights
with the best val macro-F1 are kept (best-model selection), and Optuna maximizes
that same val macro-F1. The test split is used once, for the final model.
Every trial and the final run are logged to TensorBoard:

    uv run tensorboard --logdir data/borealtc/mlp/tensorboard

``--features instant`` drops the window statistics and writes to data/borealtc/mlp_instant.
"""

import argparse
import copy
import json
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from data_split import DATA_DIR, INSTANT_FEATURES, SEED, Splits, load_dataset, make_splits
from Major import draw_confusion_matrix

OUT_DIR = DATA_DIR / "mlp"
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
MEMORY_SHARE = 0.7  # part of free video memory a training step may take; more spills into shared RAM on Windows and slows training ~40x
BATCH_TOLERANCE = 0.02  # the chosen batch is within 2% of the largest that fits


class MLP(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int, hidden: list[int], dropout: float, activation: str):
        super().__init__()
        layers, width = [], n_inputs
        for units in hidden:
            layers += [nn.Linear(width, units), ACTIVATIONS[activation](), nn.Dropout(dropout)]
            width = units
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def suggest_params(trial: optuna.Trial) -> dict:
    n_layers = trial.suggest_int("n_layers", 1, 5)
    return {
        "hidden": [trial.suggest_int(f"units_{i}", 16, 1024, log=True) for i in range(n_layers)],
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", list(ACTIVATIONS)),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
        "class_weight": trial.suggest_categorical("class_weight", ["none", "balanced"]),
    }


def largest_batch(model: nn.Module, sample_shape: tuple, n_rows: int, device: torch.device) -> int:
    """Biggest batch (any size, at most the whole train split) whose training step fits in video memory.

    The batch doubles until a step no longer fits, then a binary search narrows it to within BATCH_TOLERANCE.
    """
    if device.type != "cuda":
        return min(n_rows, 4096)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)

    def fits(size: int) -> bool:
        try:
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            x = torch.zeros(size, *sample_shape, device=device)
            loss = model(x).sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
            used = torch.cuda.max_memory_allocated(device) - before
            del x, loss
            return used <= free * MEMORY_SHARE
        except RuntimeError:  # out of memory, from torch or from cuDNN
            model.zero_grad(set_to_none=True)
            return False
        finally:
            torch.cuda.empty_cache()

    low = min(256, n_rows)
    if not fits(low):
        return low
    high = low
    while high < n_rows:
        high = min(2 * high, n_rows)
        if not fits(high):
            break
        low = high
    while high - low > max(1, low * BATCH_TOLERANCE):
        middle = (low + high) // 2
        if fits(middle):
            low = middle
        else:
            high = middle
    return low


def predict_logits(model, x, batch: int) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[start:start + batch]) for start in range(0, len(x), batch)])


def evaluate(model, x, y, loss_fn, batch: int) -> tuple[float, np.ndarray]:
    logits = predict_logits(model, x, batch)
    return float(loss_fn(logits, y)), logits.argmax(1).cpu().numpy()


def train(make_model, x_train, x_val, params: dict, s: Splits, device, max_epochs: int, patience: int, writer=None, trial=None, time_limit=None):
    """Train one model; return the best-val-F1 weights, its score and the per-epoch history.

    ``x_train`` and ``x_val`` are tensors or anything indexed like them (see train_gru.Windows).
    A trial running longer than ``time_limit`` seconds is pruned.
    """
    started = time.time()
    torch.manual_seed(SEED)
    n_classes = len(s.encoder.classes_)
    model = make_model().to(device)
    y_train = torch.tensor(s.y_train, dtype=torch.long, device=device)
    y_val = torch.tensor(s.y_val, dtype=torch.long, device=device)

    weight = None
    if params["class_weight"] == "balanced":
        counts = np.bincount(s.y_train, minlength=n_classes)
        weight = torch.tensor(len(s.y_train) / (n_classes * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    batch = largest_batch(model, tuple(x_train.shape[1:]), len(x_train), device)
    generator = torch.Generator(device=device).manual_seed(SEED)

    best = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device, generator=generator)
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / len(x_train)
        val_loss, val_pred = evaluate(model, x_val, y_val, loss_fn, batch)
        val_f1 = f1_score(s.y_val, val_pred, average="macro")
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": val_f1, "lr": lr})
        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalar("val/macro_f1", val_f1, epoch)
            writer.add_scalar("lr", lr, epoch)
        if val_f1 > best["f1"]:
            best = {"f1": val_f1, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if trial is not None:
            trial.report(best["f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if time_limit is not None and time.time() - started > time_limit:
                trial.set_user_attr("timed_out", True)
                raise optuna.TrialPruned(f"longer than {time_limit / 60:.0f} min")
        if epoch - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    return model, best, pd.DataFrame(history), batch


def run_study(
    name: str,
    out: Path,
    s: Splits,
    device: torch.device,
    suggest: Callable[[optuna.Trial], dict],
    prepare: Callable[[dict], tuple],
    args: argparse.Namespace,
) -> None:
    """Tune with Optuna on val, retrain the best parameters, score once on test and save the results.

    ``prepare(params)`` returns the model factory and the train, val and test inputs for these parameters.
    """
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "tensorboard"

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        make_model, x_train, x_val, _ = prepare(params)
        with SummaryWriter(logdir / f"trial_{trial.number:03d}") as writer:
            _, best, history, batch = train(
                make_model, x_train, x_val, params, s, device, args.max_epochs, args.patience, writer, trial, args.trial_minutes * 60
            )
            writer.add_hparams(
                {k: str(v) if isinstance(v, list) else v for k, v in params.items()} | {"batch": batch},
                {"hparam/val_macro_f1": best["f1"]},
                run_name=".",
            )
        trial.set_user_attr("best_epoch", best["epoch"])
        trial.set_user_attr("epochs", len(history))
        trial.set_user_attr("batch", batch)
        return best["f1"]

    started = time.time()
    study = optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=args.warmup),
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[lambda study, trial: print(
            f"  trial {trial.number + 1}/{args.trials}: {trial.state.name.lower()}"
            + (f", val macro-F1 {trial.value:.3f}" if trial.value is not None else "")
            + f", best {study.best_value:.3f}, {(time.time() - started) / 60:.1f} min",
            flush=True,
        )],
    )
    study.trials_dataframe().to_csv(out / f"{name}_trials.csv", index=False)

    params = suggest(optuna.trial.FixedTrial(study.best_params))
    make_model, x_train, x_val, x_test = prepare(params)
    with SummaryWriter(logdir / "final") as writer:
        model, best, history, batch = train(make_model, x_train, x_val, params, s, device, args.max_epochs, args.patience, writer)
    history.to_csv(out / f"{name}_history.csv", index=False)
    torch.save(model.state_dict(), out / f"{name}_best.pt")

    test_pred = predict_logits(model, x_test, batch).argmax(1).cpu().numpy()
    labels = s.encoder.classes_
    report_text = classification_report(s.y_test, test_pred, target_names=labels, digits=3, zero_division=0)
    report = pd.DataFrame(classification_report(
        s.y_test, test_pred, target_names=labels, output_dict=True, zero_division=0
    )).T
    report.to_csv(out / f"{name}_classification_report.csv")
    (out / f"{name}_classification_report.txt").write_text(report_text, encoding="utf-8")
    confusion = pd.crosstab(
        pd.Series(s.encoder.inverse_transform(s.y_test), name="terrain"),
        pd.Series(s.encoder.inverse_transform(test_pred), name="predicted"),
    ).reindex(index=labels, columns=labels, fill_value=0)
    figure = draw_confusion_matrix(confusion, f"{name}: test split, per measurement")
    figure.savefig(out / f"{name}_confusion.png", dpi=150)
    plt.close(figure)

    saved = dict(params) | {"batch": batch, "best_epoch": best["epoch"], "epochs": len(history)}
    (out / f"{name}_best_params.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    summary = pd.DataFrame([{
        "model": name,
        "val_macro_f1": best["f1"],
        "test_macro_f1": f1_score(s.y_test, test_pred, average="macro"),
        "test_accuracy": accuracy_score(s.y_test, test_pred),
        "trials": len(study.trials),
        "pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
        "minutes": (time.time() - started) / 60,
    }]).set_index("model")
    summary.to_csv(out / f"{name}_summary.csv")
    print("\n", saved)
    print(summary.round(3))
    print("\nClassification report on the test split:\n" + report_text)


def add_training_args(parser: argparse.ArgumentParser, trials: int, max_epochs: int, patience: int, warmup: int) -> None:
    parser.add_argument("--trials", type=int, default=trials)
    parser.add_argument("--timeout", type=float, default=None, help="stop starting new trials after this many seconds")
    parser.add_argument("--max-epochs", type=int, default=max_epochs)
    parser.add_argument("--patience", type=int, default=patience, help="stop when val macro-F1 has not improved for this many epochs")
    parser.add_argument("--warmup", type=int, default=warmup, help="epochs of a trial that are never pruned")
    parser.add_argument("--trial-minutes", type=float, default=10, help="prune a trial that trains longer than this")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--features", choices=["all", "instant"], default="all")
    add_training_args(parser, trials=60, max_epochs=1000, patience=60, warmup=100)
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    s = make_splits(load_dataset(args.stride), INSTANT_FEATURES if args.features == "instant" else None)
    print(f"{len(s.features)} features")
    print("Surface share inside each split, %:")
    print(s.shares().round(2), "\n")
    x_train, x_val, x_test = (torch.tensor(x, dtype=torch.float32, device=device) for x in (s.x_train, s.x_val, s.x_test))

    def prepare(params: dict) -> tuple:
        def make_model() -> MLP:
            return MLP(x_train.shape[1], len(s.encoder.classes_), params["hidden"], params["dropout"], params["activation"])
        return make_model, x_train, x_val, x_test

    out = OUT_DIR if args.features == "all" else DATA_DIR / "mlp_instant"
    run_study("mlp", out, s, device, suggest_params, prepare, args)


if __name__ == "__main__":
    main()
