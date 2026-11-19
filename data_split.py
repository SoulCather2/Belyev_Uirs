"""Wheel-only (no IMU) feature dataset and the one train/val/test split every model uses.

All models, sklearn and pytorch, must call ``make_splits`` with the default ``SEED``
so they train, validate and test on exactly the same rows.

Neighbouring samples of a run are nearly identical, so splitting single rows
leaks the test set into train. Every run is therefore cut into 30 s blocks and
whole blocks are split. The first 3 s of every block are dropped: the longest
trailing window is 3 s, so no feature of a block looks into the previous one. StandardScaler and
LabelEncoder are fitted on the train split only and applied to val and test.

Sequence models use the same blocks and the same labelled rows: a sample of a split
is the window of consecutive measurements ending at one of its rows. The longest
window (2 s) is shorter than the dropped 3 s, so a window never leaves its block.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from Major import PipelineConfig

DATA_DIR = Path(__file__).parent / "data" / "borealtc"
SEED = 42
TRAIN_SIZE, TEST_SIZE, VAL_SIZE = 0.7, 0.2, 0.1
SAMPLE_RATE = 100  # Hz, time step is 0.01 s
WINDOWS = {"1s": 1 * SAMPLE_RATE, "3s": 3 * SAMPLE_RATE}
VEL_LIMIT = 1.0  # m/s, |vel| above this are logging glitches (99.9% quantile is 0.79)
CUR_LIMIT = 40.0  # A, 99.9% quantile is ~24, the rest are spikes up to 1900
BLOCK = 30 * SAMPLE_RATE  # samples per split block
GAP = max(WINDOWS.values())  # samples dropped at the start of every block
MAX_SEQUENCE = 2 * SAMPLE_RATE  # samples in the longest sequence window
INSTANT_FEATURES = ["velL", "velR", "curL", "curR", "speed", "omega", "abs_speed", "abs_omega", "cur_sum", "cur_diff"]
DERIVED_FEATURES = ["power_L", "power_R", "d_speed", "d_omega", "d_cur_sum", "d_cur_diff"]
SEQUENCE_FEATURES = INSTANT_FEATURES + DERIVED_FEATURES


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
    d["power_L"] = d["curL"] * d["velL"]
    d["power_R"] = d["curR"] * d["velR"]
    for column in ["speed", "omega", "cur_sum", "cur_diff"]:
        d[f"d_{column}"] = groups[column].diff().fillna(0) * SAMPLE_RATE
    d["run"] = d["terrain"] + "_" + d["run_idx"].astype(str)
    return d


def feature_columns(d: pd.DataFrame) -> list[str]:
    service = {"terrain", "run_idx", "time", "run", "block", "row"}
    return [c for c in d.columns if c not in service and c not in DERIVED_FEATURES]


def load_all_samples() -> pd.DataFrame:
    """Features of every 100 Hz sample of every surface except SANDY_LOAM, in time order inside each run."""
    raw = pd.read_csv(DATA_DIR / "merged_interpolation.csv")
    raw = raw[raw["terrain"] != "SANDY_LOAM"]
    data = build_features(raw, PipelineConfig()).reset_index(drop=True)
    position = data.groupby("run", sort=False).cumcount()
    data["block"] = data["run"] + "_b" + (position // BLOCK).astype(str)
    data["row"] = np.arange(len(data))
    return data


def pick_rows(data: pd.DataFrame, stride: int = 20) -> pd.DataFrame:
    """Labelled rows: every ``stride``-th sample of each block after its first GAP samples."""
    position = data.groupby("run", sort=False).cumcount()
    data = data[position % BLOCK >= GAP]
    return data.groupby("block", sort=False).nth(slice(None, None, stride)).reset_index(drop=True)


def load_dataset(stride: int = 20) -> pd.DataFrame:
    """Labelled rows with all their features, every ``stride``-th sample of each block."""
    return pick_rows(load_all_samples(), stride)


@dataclass
class Splits:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    scaler: StandardScaler
    encoder: LabelEncoder

    def shares(self) -> pd.DataFrame:
        """Percent of every surface inside each split."""
        return pd.DataFrame({
            name: part["terrain"].value_counts(normalize=True).sort_index() * 100
            for name, part in (("train", self.train), ("val", self.val), ("test", self.test))
        }).rename_axis("terrain")


def make_splits(data: pd.DataFrame, features: list[str] | None = None, seed: int = SEED) -> Splits:
    """Stratified 0.7 / 0.1 / 0.2 train / val / test split of whole blocks, scaled and encoded on train only."""
    features = features or feature_columns(data)
    blocks = data.groupby("block", sort=True)["terrain"].first()
    rest, test = train_test_split(blocks, test_size=TEST_SIZE, stratify=blocks, random_state=seed)
    train, val = train_test_split(
        rest, test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE), stratify=rest, random_state=seed
    )
    train, val, test = (data[data["block"].isin(part.index)] for part in (train, val, test))
    scaler = StandardScaler().fit(train[features])
    encoder = LabelEncoder().fit(train["terrain"])
    return Splits(
        x_train=scaler.transform(train[features]),
        x_val=scaler.transform(val[features]),
        x_test=scaler.transform(test[features]),
        y_train=encoder.transform(train["terrain"]),
        y_val=encoder.transform(val["terrain"]),
        y_test=encoder.transform(test["terrain"]),
        train=train,
        val=val,
        test=test,
        features=features,
        scaler=scaler,
        encoder=encoder,
    )


def window_offsets(samples: int, step: int) -> np.ndarray:
    """Row offsets of a window of ``samples`` consecutive measurements read every ``step``-th, ending at offset 0."""
    if samples > MAX_SEQUENCE:
        raise ValueError(f"window of {samples} samples is longer than {MAX_SEQUENCE}")
    return -np.arange(0, samples, step)[::-1]


def load_sequences(stride: int = 20) -> tuple[Splits, np.ndarray]:
    """The shared split on SEQUENCE_FEATURES and every sample scaled by its train-only scaler.

    ``values[row + window_offsets(...)]`` is the window ending at a labelled row.
    """
    data = load_all_samples()
    s = make_splits(pick_rows(data, stride), SEQUENCE_FEATURES)
    return s, s.scaler.transform(data[SEQUENCE_FEATURES]).astype(np.float32)
