from itertools import combinations
from pathlib import Path
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from scipy.fft import dct
from scipy.signal import savgol_filter
from scipy.stats import norm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

_EPOCH = pd.Timestamp("1970-01-01")

@dataclass(frozen=True)
class PipelineConfig:
    window_size: int = 30
    track_width: float = 0.6
    wheel_radius: float = 0.13
    motor_resistance: float = 0.46
    motor_back_emf: float = 0.141 / np.pi
    no_load_current: float = 1.35
    n_bins: int = 71
    n_dct_coefficients: int = 28
    model_smoothing_window: int = 9
    model_smoothing_polyorder: int = 2
    std_score_weight: float = 0.5
    std_floor: float = 0.25
    memory_margin: float = 0.32952224820710885
    omega_limit: float = 0.5
    terrain_omega_limits: dict[str, float] = field(default_factory=dict)
    motion_columns: tuple[str, ...] = ("wx", "wy", "wz", "ax", "ay", "az", "velL", "velR", "curL", "curR")
    derived_columns: tuple[str, ...] = ("omega", "speed", "Ke")
    group_columns: tuple[str, ...] = ("terrain", "run_idx")


def build_symmetric_grid(data: pd.DataFrame, config: PipelineConfig) -> tuple[np.ndarray, float]:
    """Build the default grid used for shared preprocessing outputs."""
    if config.n_bins < 3 or config.n_bins % 2 == 0:
        raise ValueError("N_BINS must be an odd integer greater than one")
    limits = []
    for _, terrain_data in data.groupby("terrain"):
        minimum = float(terrain_data["omega"].min())
        maximum = float(terrain_data["omega"].max())
        limits.append(min(abs(minimum), abs(maximum)))
    if not limits or min(limits) <= 0:
        raise ValueError("Each terrain must contain both sides of zero")
    omega_limit = config.omega_limit
    if omega_limit <= 0 or omega_limit > max(limits):
        raise ValueError(f"omega_limit must be in (0, {max(limits):.3f}]")
    
    bin_edges = np.linspace(-omega_limit, omega_limit, config.n_bins + 1)
    grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid[config.n_bins // 2] = 0.0
    return grid, omega_limit


def terrain_grid(terrain: str, config: PipelineConfig) -> tuple[np.ndarray, float]:
    """Return the configured symmetric grid for one terrain model."""
    omega_limit = config.terrain_omega_limits.get(terrain, config.omega_limit)
    bin_edges = np.linspace(-omega_limit, omega_limit, config.n_bins + 1)
    grid = (bin_edges[:-1] + bin_edges[1:]) / 2
    grid[config.n_bins // 2] = 0.0
    return grid, omega_limit


def terrain_omega_limit(data: pd.DataFrame, terrain: str, config: PipelineConfig) -> float:
    """Return the symmetric range used to bin one terrain."""
    terrain_data = data[data["terrain"] == terrain]
    if terrain_data.empty:
        raise ValueError(f"Terrain {terrain!r} is absent from data")
    if terrain == "ICE":
        reference_data = data[data["terrain"] == "ASPHALT"]
        if reference_data.empty:
            raise ValueError("ASPHALT is required as the ICE range reference")
        terrain_data = reference_data
    limit = min(
        abs(float(terrain_data["omega"].min())),
        abs(float(terrain_data["omega"].max())),
    )
    configured_limit = config.terrain_omega_limits.get(terrain)
    if configured_limit is not None:
        limit = min(limit, configured_limit)
    if limit <= 1e-6:
        raise ValueError(f"Terrain {terrain!r} does not span both omega directions")
    return limit


def assign_omega_bins(data: pd.DataFrame, omega_grid: np.ndarray, omega_limit: float) -> pd.DataFrame:
    """Assign measurements to equal-width bins centered symmetrically around zero."""
    result = data[data["omega"].between(-omega_limit, omega_limit)].copy()
    width = 2 * omega_limit / len(omega_grid)
    result["omega_bin"] = (
        (np.floor((result["omega"] + omega_limit) / width)).clip(0, len(omega_grid) - 1)
    )
    result["omega_bin"] = result["omega_bin"].astype(int).map(dict(enumerate(omega_grid)))
    return result


def robust_std(values: pd.Series) -> float:
    """Estimate spread without letting one extreme value dominate."""
    values = values.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return float(1.4826 * mad)


class DCTModel:
    """Continuous DCT model compatible with the project's classifier."""

    def __init__(
        self,
        data: np.ndarray,
        cutoff_amount: int | None = None,
        default_cutoff: int = 10,
        range_min: float = -1,
        range_max: float = 1,
    ) -> None:
        self.range_min = range_min
        self.range_max = range_max
        self.coefficients = dct(np.asarray(data, dtype=float))
        if cutoff_amount is None:
            cutoff_amount = default_cutoff
        if cutoff_amount is not None:
            largest = np.argsort(np.abs(self.coefficients))[::-1]
            self.coefficients[largest[cutoff_amount:]] = 0

    def numpy_func(self, x: float | np.ndarray, scaled: bool = False) -> np.ndarray:
        """Evaluate the analytical cosine sum at one or more directions."""
        x = np.asarray(x, dtype=float)
        if scaled:
            x = (x - self.range_min) / (self.range_max - self.range_min)
            x = x * (len(self.coefficients) - 1)
        result = np.full_like(x, self.coefficients[0] / 2, dtype=float)
        for index in range(1, len(self.coefficients)):
            result += self.coefficients[index] * np.cos(
                index / len(self.coefficients) * np.pi * (x + 0.5)
            )
        return result / len(self.coefficients)


class MedianModel:
    """Adapter exposing a DCT model through the classifier interface."""

    def __init__(self, name: str, feature_name: str, model: DCTModel) -> None:
        self.name = name
        self.feature_name = feature_name
        self.model = model

    def predict(self, direction: float) -> float:
        # DCT-модель возвращает ожидаемое значение Ke для данного направления.
        return float(self.model.numpy_func(direction, scaled=True))

    def predict_many(self, directions: np.ndarray) -> np.ndarray:
        return self.model.numpy_func(directions, scaled=True)


class MetricNormal:
    """Convert model deviation to a normal-distribution likelihood."""

    def __init__(self, name: str, feature_name: str, std: float) -> None:
        self.name = name
        self.feature_name = feature_name
        self.std = max(float(std), 1e-6)

    def calc_metric(self, value: float) -> float:
        # Чем меньше отклонение от модели, тем выше правдоподобие поверхности.
        return float(norm(0, self.std).pdf(value) + 1e-3)


class Classifier:
    """Classify sequential measurements using likelihood and probability memory."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.models: dict[str, MedianModel] = {}
        self.metrics: dict[str, MetricNormal] = {}
        self.alpha = alpha
        self.prev_results: np.ndarray | None = None

    def add_surf(self, model: MedianModel, metric: MetricNormal) -> None:
        if model.name != metric.name or model.feature_name != metric.feature_name:
            raise ValueError("Model and metric must describe the same surface and feature")
        self.models[model.name] = model
        self.metrics[model.name] = metric

    def classify(self, direction: float, feature: float) -> dict[str, float]:
        return {
            surface: self.metrics[surface].calc_metric(
                feature - model.predict(direction)
            )
            for surface, model in self.models.items()
        }

    def classify_many(
        self, directions: np.ndarray, features: np.ndarray
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Classify a whole experiment with vectorized likelihood calculations."""
        names = list(self.models)
        predictions = np.column_stack(
            [self.models[name].predict_many(directions) for name in names]
        )
        standard_deviations = np.array(
            [self.metrics[name].std for name in names], dtype=float
        )
        deviations = features[:, None] - predictions
        likelihoods = np.exp(
            -0.5 * (deviations / standard_deviations) ** 2
        ) / (standard_deviations * np.sqrt(2 * np.pi)) + 1e-3
        raw_values = likelihoods / likelihoods.sum(axis=1, keepdims=True)

        # Последовательная память остается поэлементной, но дорогие расчеты выше векторизованы.
        memory_values = np.empty_like(raw_values)
        previous = self.prev_results
        for index, raw_value in enumerate(raw_values):
            if previous is None:
                memory_value = raw_value
            else:
                memory_value = self.alpha * previous + (1 - self.alpha) * raw_value
                memory_value = memory_value / memory_value.sum()
            memory_values[index] = memory_value
            previous = memory_value
        self.prev_results = previous
        return names, raw_values, memory_values

    def classify_type_and_prob(
        self, direction: float, feature: float
    ) -> tuple[str, np.ndarray, str, np.ndarray]:
        names = list(self.models)
        # Нормируем likelihood каждой поверхности в набор вероятностей.
        raw_values = np.array(
            list(self.classify(direction, feature).values()), dtype=float
        )
        raw_values /= raw_values.sum()
        raw_name = names[int(np.argmax(raw_values))]

        memory_values = raw_values
        if self.prev_results is not None:
            # alpha удерживает предыдущий прогноз и сглаживает резкие переключения.
            memory_values = self.alpha * self.prev_results + (1 - self.alpha) * raw_values
            memory_values /= memory_values.sum()
        self.prev_results = memory_values
        memory_name = names[int(np.argmax(memory_values))]
        return raw_name, raw_values, memory_name, memory_values


def calculate_features(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Calculate motion and energy features for every measurement."""
    missing_columns = set(config.motion_columns + config.group_columns) - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    features = data.copy()
    features["omega"] = (features["velR"] - features["velL"]) / config.track_width
    features["speed"] = (features["velR"] + features["velL"]) / 2
    left_wheel_omega = features["velL"] / config.wheel_radius
    right_wheel_omega = features["velR"] / config.wheel_radius
    left_voltage = features["curL"] * config.motor_resistance + config.motor_back_emf * left_wheel_omega
    right_voltage = features["curR"] * config.motor_resistance + config.motor_back_emf * right_wheel_omega
    actual_power = (left_voltage * features["curL"]).abs() + (right_voltage * features["curR"]).abs()
    no_load_power = (left_voltage * config.no_load_current).abs() + (right_voltage * config.no_load_current).abs()
    features["Ke"] = actual_power / no_load_power.where(no_load_power.ne(0))
    return features


def filter_by_experiment(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Filter raw motion measurements without crossing experiment boundaries."""
    filter_columns = list(config.motion_columns)
    filtered = data.copy()
    filtered[filter_columns] = (
        data.groupby(list(config.group_columns), sort=False, group_keys=False)[filter_columns]
        .rolling(window=config.window_size, center=True, min_periods=1)
        .mean()
        .reset_index(level=list(config.group_columns), drop=True)
    )
    return filtered


def split_by_motion(
    data: pd.DataFrame, 
    grid: np.ndarray, 
    omega_limit: float, 
    output_dir: Path | None = None
) -> dict[Path, pd.DataFrame]:
    """
    Split records by turn direction and discretized angular velocity.
    Returns a dictionary of DataFrames. Saves to disk only if output_dir is provided.
    """
    data = data.copy()
    data["motion_direction"] = data["omega"].map(
        lambda value: "left" if value > 0 else "right" if value < 0 else "straight"
    )
    data = assign_omega_bins(data, grid, omega_limit)

    files = {}
    for (direction, omega_bin), group in data.groupby(["motion_direction", "omega_bin"], sort=True):
        filename = f"{direction}_omega_{omega_bin:+.2f}.csv"
        path = (output_dir / filename) if output_dir is not None else Path(filename)
        files[path] = group.copy()
        
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            group.to_csv(path, index=False)
            
    return files


def remove_ke_outliers(data: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Remove Ke outliers independently for each terrain, run, and per-terrain symmetric omega bin."""
    rows = []
    for terrain, t_data in data.groupby("terrain"):
        terrain_limit = terrain_omega_limit(data, terrain, config)
        
        t_data = t_data[t_data["omega"].between(-terrain_limit, terrain_limit)].copy()
        
        bin_edges = np.linspace(-terrain_limit, terrain_limit, config.n_bins + 1)
        t_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        t_grid[config.n_bins // 2] = 0.0
        
        width = 2 * terrain_limit / len(t_grid)
        t_data["omega_bin"] = (
            (np.floor((t_data["omega"] + terrain_limit) / width))
            .clip(0, len(t_grid) - 1)
            .astype(int)
        ).map(dict(enumerate(t_grid)))
        
        grouped_ke = t_data.groupby(["run_idx", "omega_bin"])["Ke"]
        q1 = grouped_ke.transform("quantile", 0.25)
        q3 = grouped_ke.transform("quantile", 0.75)
        iqr = q3 - q1
        valid = t_data["Ke"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        rows.append(t_data.loc[valid])
        
    return pd.concat(rows, ignore_index=True).drop(columns="omega_bin")


def calculate_dct_coefficients(
    data: pd.DataFrame,
    config: PipelineConfig,
    grid: np.ndarray,
    omega_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, DCTModel], dict[str, np.ndarray]]:
    """Calculate DCT models with per-terrain symmetric range trimming."""
    coefficient_rows = []
    grid_rows = []
    dct_models = {}
    dct_models_std = {}

    for terrain, terrain_data in data.groupby("terrain", sort=True):
        terrain_limit = terrain_omega_limit(data, terrain, config)

        limited_terrain = terrain_data[terrain_data["omega"].between(-terrain_limit, terrain_limit)].copy()
        
        bin_edges = np.linspace(-terrain_limit, terrain_limit, config.n_bins + 1)
        terrain_grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        terrain_grid[config.n_bins // 2] = 0.0
        
        width = 2 * terrain_limit / len(terrain_grid)
        limited_terrain["omega_bin_idx"] = (
            (np.floor((limited_terrain["omega"] + terrain_limit) / width))
            .clip(0, len(terrain_grid) - 1)
            .astype(int)
        )
        limited_terrain["omega_bin"] = limited_terrain["omega_bin_idx"].map(dict(enumerate(terrain_grid)))
        
        terrain_medians = limited_terrain.groupby("omega_bin", as_index=False)["Ke"].agg(
            Ke_median="median", Ke_std=robust_std
        )
        
        real_series = terrain_medians.set_index("omega_bin")["Ke_median"].reindex(terrain_grid)
        real_std_series = terrain_medians.set_index("omega_bin")["Ke_std"].reindex(terrain_grid)
        
        series = real_series.interpolate(method="linear").ffill().bfill()
        if len(series) >= config.model_smoothing_window:
            series = pd.Series(
                savgol_filter(series.to_numpy(), config.model_smoothing_window, config.model_smoothing_polyorder),
                index=terrain_grid,
            )
        
        model = DCTModel(
            series.to_numpy(),
            cutoff_amount=config.n_dct_coefficients,
            range_min=terrain_grid.min(),
            range_max=terrain_grid.max(),
        )
        model.omega_grid = terrain_grid
        dct_models[terrain] = model
        
        std_series = real_std_series.interpolate(method="linear").ffill().bfill()
        std_series = std_series.fillna(terrain_medians["Ke_std"].mean()).fillna(config.std_floor)
        dct_models_std[terrain] = std_series.to_numpy()
        
        coefficients = model.coefficients
        coefficient_rows.extend(
            {"terrain": terrain, "coefficient_index": index, "coefficient": value}
            for index, value in enumerate(coefficients)
        )
        grid_rows.extend(
            {
                "terrain": terrain,
                "omega_bin": omega,
                "Ke_median": real_series.loc[omega],
                "Ke_std": real_std_series.loc[omega]
            }
            for omega in terrain_grid
        )

    return pd.DataFrame(grid_rows), pd.DataFrame(coefficient_rows), dct_models, dct_models_std


def plot_median_ke(
    data: pd.DataFrame, 
    config: PipelineConfig, 
    output_dir: Path | None = None,
    source_data: pd.DataFrame | None = None,
    display_limit: float | None = None,
) -> dict[str, plt.Figure]:
    """
    Generate median Ke and standard deviation plots for each terrain.
    Returns a dict of figures. Saves to disk only if output_dir is provided.
    """
    figures = {}
    for terrain in sorted(data["terrain"].unique()):
        terrain_data = data[data["terrain"] == terrain].sort_values("omega_bin")
        figure, axis = plt.subplots(figsize=(10, 5))

        axis.plot(
            terrain_data["omega_bin"],
            terrain_data["Ke_median"],
            color="0.45",
            marker="o",
            linestyle="--",
            linewidth=1.5,
            label="Median Ke",
        )
        model = DCTModel(
            terrain_data["Ke_median"].to_numpy(),
            default_cutoff=config.n_dct_coefficients,
            range_min=terrain_data["omega_bin"].min(),
            range_max=terrain_data["omega_bin"].max(),
        )
        dct_x = np.linspace(
            float(terrain_data["omega_bin"].min()),
            float(terrain_data["omega_bin"].max()),
            400,
        )
        dct_curve = model.numpy_func(dct_x, scaled=True)
        axis.plot(
            dct_x,
            dct_curve,
            color="crimson",
            linewidth=2.5,
            zorder=5,
            label=f"DCT-II model ({config.n_dct_coefficients} coeffs)",
        )
        axis.fill_between(
            terrain_data["omega_bin"],
            terrain_data["Ke_median"] - terrain_data["Ke_std"],
            terrain_data["Ke_median"] + terrain_data["Ke_std"],
            alpha=0.25,
            label="Ke +/- std",
        )
        axis.set_title(str(terrain))
        axis.set_xlabel("Angular velocity omega, rad/s")
        axis.set_ylabel("Ke")
        if source_data is not None and display_limit is not None:
            observed_limit = max(
                abs(float(source_data.loc[source_data["terrain"] == terrain, "omega"].min())),
                abs(float(source_data.loc[source_data["terrain"] == terrain, "omega"].max())),
            )
            display_limit = max(display_limit, observed_limit)
            axis.set_xlim(-display_limit, display_limit)
        else:
            axis.set_xlim(
                float(terrain_data["omega_bin"].min()),
                float(terrain_data["omega_bin"].max()),
            )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_dir / f"{terrain}_median_ke.png", dpi=150)
            
        figures[terrain] = figure
    return figures


def plot_terrain_ranges(
    source_data: pd.DataFrame,
    output_path: Path | None = None,
    display_limit: float = 1.0,
) -> plt.Figure:
    """Show each terrain's observed symmetric omega range without plotting data."""
    limits = {}
    for terrain, terrain_data in source_data.groupby("terrain", sort=True):
        negative_limit = abs(float(terrain_data["omega"].min()))
        positive_limit = abs(float(terrain_data["omega"].max()))
        limits[terrain] = min(negative_limit, positive_limit)

    display_limit = max(display_limit, max(limits.values()))
    figure, axis = plt.subplots(figsize=(11, 4.5))
    y_positions = np.arange(len(limits))
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(limits)))

    for y_position, (terrain, limit), color in zip(y_positions, limits.items(), colors):
        axis.plot([-limit, limit], [y_position, y_position], color=color, linewidth=8, solid_capstyle="round")
        axis.plot([-limit, limit], [y_position, y_position], color="black", linewidth=1)
        axis.text(
            0,
            y_position + 0.16,
            f"- {limit:.3f} ... + {limit:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.axvline(-display_limit, color="0.35", linestyle=":", linewidth=1.5)
    axis.axvline(display_limit, color="0.35", linestyle=":", linewidth=1.5)
    axis.set(
        xlim=(-display_limit, display_limit),
        ylim=(-0.6, len(limits) - 0.4),
        yticks=y_positions,
        yticklabels=list(limits),
        xlabel="Angular velocity omega, rad/s",
        title="Symmetric omega range retained for each terrain",
    )
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    return figure


def plot_all_terrains(
    median_ke_data: pd.DataFrame,
    config: PipelineConfig,
    output_path: Path | None = None,
    x_limit: float = 0.7,
    show_std: bool = True,
) -> plt.Figure:
    """
    Plot ONLY the real aggregated binned data for all terrains.
    Returns the figure. Saves to disk only if output_path is provided.
    """
    colors = {
        "ASPHALT": "#1f77b4",
        "FLOORING": "#ff7f0e",
        "ICE": "#2ca02c",
        "SNOW": "#9467bd",
    }
    figure, axis = plt.subplots(figsize=(13, 7))
    
    for terrain in sorted(median_ke_data["terrain"].unique()):
        terrain_data = median_ke_data[median_ke_data["terrain"] == terrain].sort_values("omega_bin")
        color = colors.get(terrain, "#000000")
        
        axis.plot(
            terrain_data["omega_bin"],
            terrain_data["Ke_median"],
            color=color,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=terrain,
        )
        if show_std:
            axis.fill_between(
                terrain_data["omega_bin"],
                terrain_data["Ke_median"] - terrain_data["Ke_std"].fillna(0),
                terrain_data["Ke_median"] + terrain_data["Ke_std"].fillna(0),
                color=color,
                alpha=0.15,
            )
        
    axis.set_title("Real aggregated Ke(omega) data across all terrains")
    axis.set_xlabel("Angular velocity omega, rad/s")
    axis.set_ylabel("Ke")
    if x_limit <= 0:
        raise ValueError("x_limit must be positive")
    axis.set_xlim(-x_limit, x_limit)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    figure.tight_layout()
    
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    return figure


def plot_ke_by_discrete_directions(
    data: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot the real aggregated Ke data for each terrain and direction bin.
    Returns the figure. Saves to disk only if output_path is provided.
    """
    colors = {
        "ASPHALT": "#1f77b4",
        "FLOORING": "#ff7f0e",
        "ICE": "#2ca02c",
        "SANDY_LOAM": "#d62728",
        "SNOW": "#9467bd",
    }
    figure, axis = plt.subplots(figsize=(13, 7))
    for terrain in sorted(data["terrain"].unique()):
        terrain_data = data[data["terrain"] == terrain].sort_values("omega_bin")
        color = colors.get(terrain)
        axis.fill_between(
            terrain_data["omega_bin"],
            terrain_data["Ke_median"] - terrain_data["Ke_std"].fillna(0),
            terrain_data["Ke_median"] + terrain_data["Ke_std"].fillna(0),
            color=color,
            alpha=0.16,
        )
        axis.plot(
            terrain_data["omega_bin"],
            terrain_data["Ke_median"],
            linewidth=1.8,
            marker="o",
            markersize=3,
            color=color,
            label=terrain,
        )
    axis.set_title("Ke depending on discrete motion direction")
    axis.set_xlabel("Discrete angular direction, omega bin (rad/s)")
    axis.set_ylabel("Ke")
    axis.set_xlim(-0.5, 0.5)
    axis.set_xticks(np.linspace(-0.5, 0.5, 11))
    axis.set_ylim(0, 10)
    axis.grid(True, alpha=0.3)
    axis.legend(title="Surface")
    figure.tight_layout()
    
    if output_path is not None:
        figure.savefig(output_path, dpi=150)
    return figure


def classify_experiments_by_dct(
    data: pd.DataFrame,
    dct_models: dict[str, DCTModel],
    dct_models_std: dict[str, np.ndarray],
    config: PipelineConfig,
    initial_prediction: str | None = None,
) -> pd.DataFrame:
    """Classify sequential measurements with likelihood-based probability memory."""
    # Классификатор получает DCT-модели и их разброс по каждой поверхности.
    classifier = Classifier(alpha=config.memory_margin)
    for terrain, model in dct_models.items():
        median_std = float(np.nanmedian(dct_models_std[terrain]))
        median_model = MedianModel(terrain, "Ke", model)
        metric = MetricNormal(terrain, "Ke", median_std)
        classifier.add_surf(median_model, metric)

    rows = []
    previous_prediction = initial_prediction

    for (true_terrain, run_idx), experiment in data.groupby(["terrain", "run_idx"], sort=True):
        raw_prediction = None
        memory_prediction = None
        raw_values = None
        memory_values = None
        directions = experiment["omega"].to_numpy(dtype=float)
        features = experiment["Ke"].to_numpy(dtype=float)
        names, raw_probabilities, memory_probabilities = classifier.classify_many(
            directions, features
        )
        raw_values = raw_probabilities[-1]
        memory_values = memory_probabilities[-1]
        raw_prediction = names[int(np.argmax(raw_values))]
        memory_prediction = names[int(np.argmax(memory_values))]

        if memory_prediction is None or raw_values is None or memory_values is None:
            continue

        rows.append(
            {
                "terrain": true_terrain,
                "run_idx": run_idx,
                "predicted_terrain": memory_prediction,
                "raw_predicted_terrain": raw_prediction,
                "previous_prediction": previous_prediction,
                **{
                    f"raw_probability_{terrain}": value
                    for terrain, value in zip(classifier.models, raw_values)
                },
                **{
                    f"memory_probability_{terrain}": value
                    for terrain, value in zip(classifier.models, memory_values)
                },
            }
        )
        previous_prediction = memory_prediction

    return pd.DataFrame(rows)


def generate_classification_report(
    classification: pd.DataFrame, 
    output_dir: Path | None = None
) -> tuple[float, str, pd.DataFrame]:
    """
    Generate sklearn classification metrics.
    Returns (accuracy, report_text, report_df). Saves to disk only if output_dir is provided.
    """
    labels = sorted(set(classification["terrain"]) | set(classification["predicted_terrain"]))
    y_true = classification["terrain"]
    y_pred = classification["predicted_terrain"]
    
    accuracy = accuracy_score(y_true, y_pred)
    report_text = classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)
    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0)
    ).transpose()
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_intro = (
            "Project stage summary\n"
            "=====================\n"
            "Earlier stages prepared filtered motion data, DCT surface models, "
            "and Optuna hyperparameter search.\n"
            "This stage evaluates the likelihood-based surface classifier with "
            "probability memory on the filtered experiments.\n\n"
        )
        (output_dir / "classification_report.txt").write_text(
            report_intro + f"Accuracy: {accuracy:.6f}\n\n{report_text}",
            encoding="utf-8",
        )
        report_df.to_csv(output_dir / "classification_report.csv")
        (output_dir / "classification_accuracy.txt").write_text(f"{accuracy:.6f}\n", encoding="utf-8")
        
    return float(accuracy), report_text, report_df


def plot_scatter_with_errors(
    data: pd.DataFrame,
    x: str,
    y: str | Sequence[str],
    wrong_cls: str,
    *,
    error_name: str | None = None,
    error_color: str = "red",
    error_opacity: float = 0.3,
    **kwargs: Any,
) -> go.Figure:
    """Scatter plot (plotly.express.scatter) with shaded regions where the classifier is wrong.
 
    Parameters
    ----------
    data : DataFrame with the already computed data. Correctness of the
        classification must be computed beforehand; nothing is computed here.
    x : name of the x column (numeric or datetime).
    y : name or list of names of y columns (as in px.scatter).
    wrong_cls : name of a boolean (or 0/1) column; True means "classifier is wrong".
        NaN is treated as "no error".
    error_name : legend label of the shading (defaults to ``wrong_cls``).
    error_color, error_opacity : shading appearance.
    **kwargs : passed as is to ``plotly.express.scatter``
        (color, symbol, facet_col, template, title, labels, ...).
 
    Returns
    -------
    plotly.graph_objects.Figure. The shading is one legend entry; clicking it
    hides/shows all shaded regions (in all facets as well).
 
    Notes
    -----
    A shaded region spans from the midpoint between a wrong point and its
    correct left neighbour to the midpoint with its correct right neighbour,
    so consecutive wrong points merge into one band. Requires plotly >= 5.15
    (shapes in the legend).
    """
    # ---- validation (before any plotting) ----
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    for name in (x, wrong_cls):
        if name not in data.columns:
            raise ValueError(f"Column {name!r} is not in the dataset")
    if "row" in kwargs or "col" in kwargs:
        raise TypeError("Do not pass row/col; use facet_row/facet_col")
 
    error_values = data[wrong_cls]
    if not is_bool_dtype(error_values.dtype):
        if not is_numeric_dtype(error_values.dtype) or not error_values.dropna().isin([0, 1]).all():
            raise TypeError(f"{wrong_cls!r} must contain boolean or 0/1 values")
    errors = error_values.fillna(False).astype(bool).to_numpy()
 
    x_values = data[x]
    if is_datetime64_any_dtype(x_values.dtype):
        dt = pd.to_datetime(x_values)
        if dt.dt.tz is not None:
            dt = dt.dt.tz_localize(None)  # wall-clock time, as shown on the axis
        coordinates = ((dt - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        datetime_axis = True
    elif is_numeric_dtype(x_values.dtype) and not is_bool_dtype(x_values.dtype):
        coordinates = x_values.to_numpy(dtype=float, na_value=np.nan)
        datetime_axis = False
    else:
        raise TypeError(f"{x!r} must be numeric or datetime to shade intervals")
 
    figure = px.scatter(data_frame=data, x=x, y=y, **kwargs)
 
    # ---- build shaded intervals ----
    valid = np.isfinite(coordinates)
    points = pd.DataFrame({"c": coordinates[valid], "err": errors[valid]})
    if points.empty:
        return figure
    # One x can hold several series/rows: shade it if any of them is wrong.
    points = points.groupby("c", sort=True, as_index=False)["err"].any()
    c = points["c"].to_numpy()
    err = points["err"].to_numpy()
    if not err.any():
        return figure
 
    n = len(c)
    if n == 1:
        edges = np.array([c[0] - 0.5, c[0] + 0.5])
    else:
        mid = (c[:-1] + c[1:]) / 2
        edges = np.concatenate(([c[0] - (c[1] - c[0]) / 2], mid, [c[-1] + (c[-1] - c[-2]) / 2]))
 
    padded = np.concatenate(([False], err, [False])).astype(int)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)  # exclusive
    intervals = [(edges[s], edges[e]) for s, e in zip(starts, ends)]
 
    def axis_value(value: float) -> pd.Timestamp | float:
        return _EPOCH + pd.to_timedelta(value, unit="s") if datetime_axis else float(value)
 
    # ---- add shapes; single legend entry controlling all of them ----
    label = error_name or wrong_cls
    first = len(figure.layout.shapes)
    for left, right in intervals:
        figure.add_vrect(
            x0=axis_value(left),
            x1=axis_value(right),
            fillcolor=error_color,
            opacity=error_opacity,
            line_width=0,
            layer="below",
            name=label,
            legendgroup="classification_errors",
            showlegend=False,
            row="all",
            col="all",
        )
    new_shapes = figure.layout.shapes[first:]
    if new_shapes:
        new_shapes[0].showlegend = True
    return figure

def plot_confusion_matrix(
    classification: pd.DataFrame, 
    output_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, plt.Figure]:
    """
    Generate raw and row-normalized DCT classification error maps.
    Returns (normalized_df, raw_df, figure). Saves to disk only if output_dir is provided.
    """
    labels = sorted(set(classification["terrain"]) | set(classification["predicted_terrain"]))
    matrix = confusion_matrix(classification["terrain"], classification["predicted_terrain"], labels=labels).astype(float)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)

    normalized_df = pd.DataFrame(normalized, index=labels, columns=labels)
    raw_df = pd.DataFrame(matrix, index=labels, columns=labels)

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Recall")
    axis.set(
        xlabel="Predicted terrain",
        ylabel="True terrain",
        title="DCT classifier confusion matrix",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center")
    figure.tight_layout()
    
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_df.to_csv(output_dir / "dct_confusion_matrix.csv")
        raw_df.to_csv(output_dir / "dct_confusion_matrix_counts.csv")
        figure.savefig(output_dir / "dct_confusion_matrix.png", dpi=150)
        
    return normalized_df, raw_df, figure
def classify_samples_by_dct(
    data: pd.DataFrame,
    dct_models: dict,
    dct_models_std: dict,
    config,
    reset_memory: bool = False,
) -> pd.DataFrame:
    """Classify every measurement; add ``wrong`` / ``raw_wrong`` columns.
 
    reset_memory=False reproduces classify_experiments_by_dct (probability memory
    is carried from one experiment to the next). reset_memory=True starts every
    experiment from scratch, which is the honest per-run estimate.
    Rows inside an experiment must already be in time order.
    """
    missing = {"terrain", "run_idx", "omega", "Ke"} - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if data[["omega", "Ke"]].isna().any().any():
        raise ValueError("omega and Ke must not contain NaN (it would poison the probability memory)")
 
    classifier = Classifier(alpha=config.memory_margin)
    for terrain, model in dct_models.items():
        median_std = float(np.nanmedian(dct_models_std[terrain]))
        classifier.add_surf(MedianModel(terrain, "Ke", model), MetricNormal(terrain, "Ke", median_std))
 
    parts = []
    for _, experiment in data.groupby(["terrain", "run_idx"], sort=True):
        if reset_memory:
            classifier.prev_results = None
        names, raw_values, memory_values = classifier.classify_many(
            experiment["omega"].to_numpy(dtype=float),
            experiment["Ke"].to_numpy(dtype=float),
        )
        part = experiment.copy()
        part["sample_idx"] = np.arange(len(part))
        part["raw_predicted_terrain"] = np.asarray(names)[raw_values.argmax(axis=1)]
        part["predicted_terrain"] = np.asarray(names)[memory_values.argmax(axis=1)]
        for column, name in enumerate(names):
            part[f"raw_probability_{name}"] = raw_values[:, column]
            part[f"memory_probability_{name}"] = memory_values[:, column]
        parts.append(part)
 
    result = pd.concat(parts, ignore_index=True)
    result["wrong"] = result["predicted_terrain"] != result["terrain"]
    result["raw_wrong"] = result["raw_predicted_terrain"] != result["terrain"]
    return result

def add_error_features(
    samples: pd.DataFrame, amplitude_window: int = 60, straight_band: float = 0.02
) -> pd.DataFrame:
    """Add features used to explain errors (all computed inside one run)."""
    s = samples.copy()
    s["abs_omega"] = s["omega"].abs()
    s["direction"] = np.where(
        s["omega"] > straight_band, "left", np.where(s["omega"] < -straight_band, "right", "straight")
    )
    if {"curL", "curR"} <= set(s.columns):
        s["cur_mean"] = (s["curL"].abs() + s["curR"].abs()) / 2
        s["cur_diff"] = (s["curL"] - s["curR"]).abs()
    if "speed" in s.columns:
        s["abs_speed"] = s["speed"].abs()
    keys = ["terrain", "run_idx"]
    roll = s.groupby(keys, sort=False)["omega"]
    s["amplitude"] = (
        roll.transform(lambda v: v.rolling(amplitude_window, center=True, min_periods=1).max())
        - roll.transform(lambda v: v.rolling(amplitude_window, center=True, min_periods=1).min())
    )
    size = s.groupby(keys, sort=False)["omega"].transform("size")
    s["run_pos"] = s["sample_idx"] / (size - 1).clip(lower=1)
    return s
 
 
def error_by_feature(s: pd.DataFrame, features: list[str], n_bins: int = 8, error_col: str = "wrong") -> pd.DataFrame:
    """Error rate inside equal-count bins of every feature, overall and per true terrain."""
    rows = []
    groups = [("ALL", s)] + [(str(t), g) for t, g in s.groupby("terrain", sort=True)]
    for terrain, g in groups:
        for feature in features:
            valid = g[np.isfinite(g[feature])]
            if valid[feature].nunique() < 2:
                continue
            bins = pd.qcut(valid[feature], q=n_bins, duplicates="drop")
            for interval, part in valid.groupby(bins, observed=True):
                rows.append(
                    {
                        "terrain": terrain,
                        "feature": feature,
                        "bin_left": float(interval.left),
                        "bin_right": float(interval.right),
                        "n": len(part),
                        "error_rate": float(part[error_col].mean()),
                    }
                )
    return pd.DataFrame(rows)
 
 
def model_separation(dct_models: dict, dct_models_std: dict, points: int = 200) -> pd.DataFrame:
    """How far apart two surface models are, in units of their spread, at each omega."""
    names = list(dct_models)
    lo = max(m.range_min for m in dct_models.values())
    hi = min(m.range_max for m in dct_models.values())
    omegas = np.linspace(lo, hi, points)
    predictions = {n: MedianModel(n, "Ke", dct_models[n]).predict_many(omegas) for n in names}
    stds = {n: float(np.nanmedian(dct_models_std[n])) for n in names}
    rows = []
    for a, b in combinations(names, 2):
        pooled = np.sqrt((stds[a] ** 2 + stds[b] ** 2) / 2)
        separation = np.abs(predictions[a] - predictions[b]) / pooled
        rows.extend({"pair": f"{a} vs {b}", "omega": w, "separation": v} for w, v in zip(omegas, separation))
    return pd.DataFrame(rows)
 
 
def pick_example_runs(by_run: pd.DataFrame, worst: int = 1, mixed: int = 2, mixed_target: float = 0.4) -> pd.DataFrame:
    """Per terrain: the worst runs, plus runs with a mix of right and wrong (most informative)."""
    picks = []
    for terrain, runs in by_run.groupby("terrain", sort=True):
        top = runs.sort_values("error_rate", ascending=False).head(worst)
        rest = runs.drop(top.index)
        mid = rest.assign(d=(rest["error_rate"] - mixed_target).abs()).sort_values("d").head(mixed)
        picks.append(top.assign(kind="worst"))
        picks.append(mid.drop(columns="d").assign(kind="mixed"))
    return pd.concat(picks, ignore_index=True)
 
 
def save_example_plots(s: pd.DataFrame, by_run: pd.DataFrame, out_dir: Path, y=None) -> list[Path]:
    """Save html plots (one panel per signal, own y scale) with red error bands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x = "time" if "time" in s.columns else "sample_idx"
    signals = [c for c in (y or ["omega", "Ke", "speed", "cur_mean"]) if c in s.columns]
    saved = []
    for _, row in pick_example_runs(by_run).iterrows():
        part = s[(s["terrain"] == row["terrain"]) & (s["run_idx"] == row["run_idx"])]
        long = part.melt(id_vars=[x, "wrong"], value_vars=signals, var_name="signal", value_name="value")
        fig = plot_scatter_with_errors(
            long, x=x, y="value", wrong_cls="wrong", facet_row="signal", color="signal",
            error_name="Ошибка классификатора",
            title=f"Поверхность: {row['terrain']}, прогон {row['run_idx']} "
                  f"(ошибок {row['error_rate']:.0%}, {row['kind']})",
        )
        fig.update_yaxes(matches=None)
        fig.update_traces(marker_size=4)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        path = out_dir / f"{row['kind']}_{row['terrain']}_run{row['run_idx']}.html"
        fig.write_html(path)
        saved.append(path)
    return saved
 
 
def run_error_analysis(cleaned, dct_models, dct_models_std, config, out_dir: Path,
                       n_bins: int = 8, make_plots: bool = True) -> pd.DataFrame:
    """Compute everything needed for the error analysis and save small CSV summaries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    carry = classify_samples_by_dct(cleaned, dct_models, dct_models_std, config, reset_memory=False)
    reset = classify_samples_by_dct(cleaned, dct_models, dct_models_std, config, reset_memory=True)
    s = add_error_features(reset)
 
    # 1. Overall, and the effect of carrying memory between runs.
    by_terrain = pd.DataFrame(
        {
            "n": s.groupby("terrain").size(),
            "error_reset_memory": s.groupby("terrain")["wrong"].mean(),
            "error_carry_memory": carry.groupby("terrain")["wrong"].mean(),
            "error_no_memory": s.groupby("terrain")["raw_wrong"].mean(),
        }
    ).reset_index()
    by_terrain.to_csv(out_dir / "1_error_by_terrain.csv", index=False)
 
    # 2. Which surface is mistaken for which.
    pd.crosstab(s["terrain"], s["predicted_terrain"]).to_csv(out_dir / "2_confusion_per_sample.csv")
 
    # 3. Errors against features.
    features = [c for c in ["abs_omega", "omega", "abs_speed", "cur_mean", "cur_diff", "Ke", "amplitude"] if c in s.columns]
    error_by_feature(s, features, n_bins).to_csv(out_dir / "3_error_by_feature.csv", index=False)
    s.groupby(["terrain", "direction"])["wrong"].agg(n="size", error_rate="mean").reset_index().to_csv(
        out_dir / "4_error_by_direction.csv", index=False
    )
 
    # 5. Position inside the run (start effect).
    s["run_pos_bin"] = pd.cut(s["run_pos"], [-0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0])
    s.groupby(["terrain", "run_pos_bin"], observed=True)["wrong"].agg(n="size", error_rate="mean").reset_index().assign(
        run_pos_bin=lambda d: d["run_pos_bin"].astype(str)
    ).to_csv(out_dir / "5_error_by_run_position.csv", index=False)
 
    # 6. Per run (to pick example plots), and 7. how close the models are.
    by_run = s.groupby(["terrain", "run_idx"]).agg(
        n=("wrong", "size"), error_rate=("wrong", "mean"),
        mean_abs_omega=("abs_omega", "mean"), mean_amplitude=("amplitude", "mean"),
        mean_Ke=("Ke", "mean"), Ke_std=("Ke", "std"),
        omega_std=("omega", "std"), omega_range=("omega", lambda v: v.max() - v.min()),
    ).reset_index()
    by_run.to_csv(out_dir / "6_error_by_run.csv", index=False)
    model_separation(dct_models, dct_models_std).to_csv(out_dir / "7_model_separation.csv", index=False)
 
    s.drop(columns="run_pos_bin").to_csv(out_dir / "samples_with_errors.csv.gz", index=False)
    if make_plots:
        for path in save_example_plots(s, by_run, out_dir / "example_plots"):
            print("saved", path)
    print(by_terrain.round(3).to_string(index=False))
    print(f"\nSaved analysis tables to {out_dir}")
    return s


def main() -> None:
    """Main execution pipeline. All state is local to this function."""
    data_dir = Path(__file__).parent / "data" / "borealtc"
    input_path = data_dir / "merged_interpolation.csv"
    output_path = data_dir / "merged_interpolation_features_filtered.csv"

    # 1. Load and filter the input data.
    data = pd.read_csv(input_path)
    initial_count = len(data)
    data = data[data["terrain"] != "SANDY_LOAM"].copy()
    print(f"Removed SANDY_LOAM. Remaining samples: {len(data)} (was {initial_count})")
    print(f"Remaining terrains: {data['terrain'].unique().tolist()}")
    
    # 2. Configure the processing pipeline.
    config = PipelineConfig()
    
    # 3. Filter raw measurements, then calculate Ke from the filtered signals.
    filtered_motion = filter_by_experiment(data, config)
    features = calculate_features(filtered_motion, config)
    grid, omega_limit = build_symmetric_grid(features, config)
    
    # 4. Remove outliers from the window-specific features.
    cleaned = remove_ke_outliers(features, config)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved filtered dataframe to {output_path}")

    # Обучение идет на разбитых motion-данных, а проверка ниже - на cleaned.
    split_data = split_by_motion(cleaned, grid, omega_limit)
    training_data = pd.concat(split_data.values(), ignore_index=True)
    
    # 5. Calculate final DCT models and standard deviations from split data.
    print(f"Using {config.n_dct_coefficients} DCT coefficients")
    median_ke, coefficients, dct_models, dct_models_std = calculate_dct_coefficients(
        training_data, config, grid, omega_limit
    )
    
    # 7. Define output paths.
    median_path = data_dir / "median_ke_by_omega.csv"
    coefficients_path = data_dir / "dct2_coefficients.csv"
    dct_models_path = data_dir / "dct_models.pkl"
    plot_dir = data_dir / "median_ke_plots"
    terrain_ranges_path = data_dir / "terrain_omega_ranges.png"
    all_terrains_path = data_dir / "all_terrains_binned_ke.png"
    all_terrains_no_std_path = data_dir / "all_terrains_binned_ke_no_std.png"
    classification_path = data_dir / "dct_experiment_classification.csv"
    
    # 8. Save model artifacts.
    median_ke.to_csv(median_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    with dct_models_path.open("wb") as file:
        pickle.dump((dct_models, dct_models_std), file)
        
    # 9. Generate and save plots.
    plot_median_ke(
        median_ke,
        config,
        output_dir=plot_dir,
        source_data=features,
    )
    plot_all_terrains(median_ke, config, output_path=all_terrains_path)
    plot_all_terrains(
        median_ke,
        config,
        output_path=all_terrains_no_std_path,
        show_std=False,
    )
    plot_terrain_ranges(features, output_path=terrain_ranges_path)
    
    # 10. Classify real cleaned experiments with models trained on split data.
    classification = classify_experiments_by_dct(
        cleaned, dct_models, dct_models_std, config
    )
    classification.to_csv(classification_path, index=False)
    
    # 11. Calculate metrics and reports.
    accuracy, report_text, report_df = generate_classification_report(classification, output_dir=data_dir)
    norm_matrix, raw_matrix, conf_figure = plot_confusion_matrix(classification, output_dir=data_dir)
    
    # 12. Print results and split records by motion.
    print(f"Classification accuracy: {accuracy:.4f}")
    print(f"Saved DCT classification to {classification_path}")
    print(f"Saved classification report to {data_dir / 'classification_report.txt'}")
    
    motion_splits = split_by_motion(cleaned, grid, omega_limit, output_dir=data_dir / "motion_splits")
    print(f"Generated {len(motion_splits)} motion splits, saved to {data_dir / 'motion_splits'}")
    # в main(), после classification = classify_experiments_by_dct(...)
    classification = classification.reset_index(drop=True)
    classification["experiment"] = classification.index
    classification["wrong"] = classification["terrain"] != classification["predicted_terrain"]

    fig = plot_scatter_with_errors(
        classification,
        x="experiment",
        y=[f"memory_probability_{t}" for t in dct_models],
        wrong_cls="wrong",
        error_name="Ошибка классификатора",
    )
    fig.write_html(data_dir / "classification_errors.html")
    samples = classify_samples_by_dct(cleaned, dct_models, dct_models_std, config)

    one = samples[(samples["terrain"] == "FLOORING") & (samples["run_idx"] == 0)]
    fig = plot_scatter_with_errors(
        one, x="time", y=["omega", "Ke"], wrong_cls="wrong",
        error_name="Ошибка классификатора",
    )
    fig.write_html(data_dir / "flooring_run0_errors.html")

    run_error_analysis(cleaned, dct_models, dct_models_std, config, out_dir=data_dir / "error_analysis")
    print({t: float(np.nanmedian(s)) for t, s in dct_models_std.items()})
    x = samples[(samples.terrain == "SNOW") & (samples.omega.abs() > 0.276)]
    print(x.predicted_terrain.value_counts(normalize=True))

if __name__ == "__main__":
    main()