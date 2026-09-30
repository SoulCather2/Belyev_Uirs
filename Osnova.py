from pathlib import Path
import pickle
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import dct
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class PipelineConfig:
    window_size: int = 10
    track_width: float = 0.6
    wheel_radius: float = 0.13
    motor_resistance: float = 0.46
    motor_back_emf: float = 0.141 / np.pi
    no_load_current: float = 1.35
    n_bins: int = 51
    n_dct_coefficients: int = 10
    model_smoothing_window: int = 9
    model_smoothing_polyorder: int = 2
    std_score_weight: float = 0.5
    std_floor: float = 0.25
    memory_margin: float = 0.05
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
    """Apply a centered mean filter without crossing experiment boundaries."""
    filter_columns = [*config.motion_columns, *config.derived_columns]
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
    """
    Classify experiments with std-normalized distances and class memory.
    Evaluated POINT-BY-POINT on original averaged data (NO binning).
    """
    rows = []
    previous_prediction = initial_prediction
    
    for (true_terrain, run_idx), experiment in data.groupby(["terrain", "run_idx"], sort=True):
        rmse_scores = {}
        std_scores = {}
        
        omega_vals = experiment["omega"].to_numpy(dtype=float)
        ke_vals = experiment["Ke"].to_numpy(dtype=float)
        
        for terrain, model in dct_models.items():
            model_grid = model.omega_grid
            model_limit = max(abs(model_grid.min()), abs(model_grid.max()))
            
            mask = (omega_vals >= -model_limit) & (omega_vals <= model_limit)
            if not np.any(mask):
                continue
                
            omega_valid = omega_vals[mask]
            ke_valid = ke_vals[mask]
            
            curve_valid = model.numpy_func(omega_valid, scaled=True)
            
            median_std = float(np.nanmedian(dct_models_std[terrain]))
            fill_val = median_std if not np.isnan(median_std) else config.std_floor
            std_interp = interp1d(model_grid, dct_models_std[terrain], bounds_error=False, fill_value=fill_val)
            std_valid = np.maximum(std_interp(omega_valid), config.std_floor)
            
            rmse_scores[terrain] = float(np.sqrt(np.mean((ke_valid - curve_valid) ** 2)))
            std_scores[terrain] = float(np.sqrt(np.mean(((ke_valid - curve_valid) / std_valid) ** 2)))
        
        if not rmse_scores:
            continue
            
        rmse_scale = max(float(np.median(list(rmse_scores.values()))), 1e-6)
        std_scale = max(float(np.median(list(std_scores.values()))), 1e-6)
        
        scores = {
            terrain: rmse_scores[terrain] / rmse_scale
            + config.std_score_weight * std_scores[terrain] / std_scale
            for terrain in rmse_scores
        }
        
        prediction = min(scores, key=scores.get)
        
        # Hysteresis: keep the previous prediction when the new one is only slightly better.
        if previous_prediction in scores and prediction != previous_prediction:
            if scores[previous_prediction] <= scores[prediction] * (1 + config.memory_margin):
                prediction = previous_prediction

        rows.append(
            {
                "terrain": true_terrain,
                "run_idx": run_idx,
                "predicted_terrain": prediction,
                "previous_prediction": previous_prediction,
                **{f"rmse_{terrain}": score for terrain, score in rmse_scores.items()},
                **{f"std_score_{terrain}": score for terrain, score in std_scores.items()},
            }
        )
        previous_prediction = prediction

    return pd.DataFrame(rows)


def classify_experiments_leave_one_out(
    data: pd.DataFrame,
    config: PipelineConfig,
    grid: np.ndarray,
    omega_limit: float,
) -> pd.DataFrame:
    """Evaluate each original experiment against models trained without it."""
    rows = []
    ordered_groups = data[list(config.group_columns)].drop_duplicates().sort_values("run_idx")
    previous_prediction = None
    
    for group in ordered_groups.itertuples(index=False):
        group_mask = (data["terrain"] == group.terrain) & (data["run_idx"] == group.run_idx)
        training_data = data.loc[~group_mask]
        
        _, _, models, models_std = calculate_dct_coefficients(training_data, config, grid, omega_limit)
        
        prediction = classify_experiments_by_dct(
            data.loc[group_mask], models, models_std, config,
            initial_prediction=previous_prediction
        )
        if prediction.empty:
            continue
            
        prediction.loc[:, "previous_prediction"] = previous_prediction
        rows.append(prediction.iloc[0])
        previous_prediction = prediction.iloc[0]["predicted_terrain"]
        
    return pd.DataFrame(rows)


def select_dct_coefficients(
    data: pd.DataFrame,
    candidates: list[int],
    config: PipelineConfig,
    grid: np.ndarray,
    omega_limit: float,
    n_splits: int = 5,
) -> tuple[int, pd.DataFrame]:
    """Select the DCT cutoff using grouped cross-validation by experiment."""
    groups = data["terrain"].astype(str) + "::" + data["run_idx"].astype(str)
    n_splits = min(n_splits, groups.nunique())
    if n_splits < 2:
        raise ValueError("At least two experiments are required for cross-validation")

    folds = list(GroupKFold(n_splits=n_splits).split(data, groups=groups))
    rows = []
    for cutoff in candidates:
        candidate_config = replace(config, n_dct_coefficients=cutoff)
        fold_scores = []
        for train_indices, test_indices in folds:
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
            _, _, models, models_std = calculate_dct_coefficients(
                train_data, candidate_config, grid, omega_limit
            )
            predictions = classify_experiments_by_dct(
                test_data, models, models_std, candidate_config
            )
            if predictions.empty:
                continue
            report = classification_report(
                predictions["terrain"], predictions["predicted_terrain"],
                output_dict=True, zero_division=0,
            )
            fold_scores.append(
                {
                    "accuracy": accuracy_score(predictions["terrain"], predictions["predicted_terrain"]),
                    "macro_f1": report["macro avg"]["f1-score"],
                }
            )
        average = pd.DataFrame(fold_scores).mean()
        rows.append(
            {
                "n_coefficients": cutoff,
                "cv_accuracy": average["accuracy"],
                "cv_macro_f1": average["macro_f1"],
            }
        )
    results = pd.DataFrame(rows).sort_values(["cv_macro_f1", "cv_accuracy"], ascending=False)
    return int(results.iloc[0]["n_coefficients"]), results


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
        (output_dir / "classification_report.txt").write_text(
            f"Accuracy: {accuracy:.6f}\n\n{report_text}", encoding="utf-8"
        )
        report_df.to_csv(output_dir / "classification_report.csv")
        (output_dir / "classification_accuracy.txt").write_text(f"{accuracy:.6f}\n", encoding="utf-8")
        
    return float(accuracy), report_text, report_df


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
    
    # 3. Calculate features and the shared angular-velocity grid.
    features = calculate_features(data, config)
    grid, omega_limit = build_symmetric_grid(features, config)
    
    # 4. Filter measurements and remove outliers.
    filtered = filter_by_experiment(features, config)
    cleaned = remove_ke_outliers(filtered, config)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved filtered dataframe to {output_path}")
    
    # 5. Calculate final DCT models and standard deviations.
    print(f"Using {config.n_dct_coefficients} DCT coefficients")
    median_ke, coefficients, dct_models, dct_models_std = calculate_dct_coefficients(
        cleaned, config, grid, omega_limit
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
    
    # 10. Run leave-one-experiment-out classification.
    classification = classify_experiments_leave_one_out(cleaned, config, grid, omega_limit)
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


if __name__ == "__main__":
    main()