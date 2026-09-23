from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import dct
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold


WINDOW_SIZE = 5
TRACK_WIDTH = 0.6
WHEEL_RADIUS = 0.13
MOTOR_RESISTANCE = 0.46
MOTOR_BACK_EMF = 0.141 / 3.141592653589793
NO_LOAD_CURRENT = 1.35
OMEGA_BIN_WIDTH = 0.02
N_DCT_COEFFICIENTS = 10
MOTION_COLUMNS = ["wx", "wy", "wz", "ax", "ay", "az", "velL", "velR", "curL", "curR"]
DERIVED_COLUMNS = ["omega", "speed", "Ke"]
GROUP_COLUMNS = ["terrain", "run_idx"]
OMEGA_GRID = np.round(np.arange(-1, 1 + OMEGA_BIN_WIDTH, OMEGA_BIN_WIDTH), 2)


class DCTModel:
	"""Continuous DCT model compatible with the project's classifier."""

	def __init__(
		self,
		data: np.ndarray,
		cutoff_amount: int | None = None,
		range_min: float = -1,
		range_max: float = 1,
	) -> None:
		self.range_min = range_min
		self.range_max = range_max
		self.coefficients = dct(np.asarray(data, dtype=float))
		if cutoff_amount is None:
			cutoff_amount = N_DCT_COEFFICIENTS
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


if __name__ == "__main__":
	sys.modules.setdefault("Osnova", sys.modules[__name__])
	DCTModel.__module__ = "Osnova"


def calculate_features(data: pd.DataFrame) -> pd.DataFrame:
	"""Calculate motion and energy features for every measurement."""
	missing_columns = set(MOTION_COLUMNS + GROUP_COLUMNS) - set(data.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Missing required columns: {missing}")

	features = data.copy()
	features["omega"] = (features["velR"] - features["velL"]) / TRACK_WIDTH
	features["speed"] = (features["velR"] + features["velL"]) / 2
	left_wheel_omega = features["velL"] / WHEEL_RADIUS
	right_wheel_omega = features["velR"] / WHEEL_RADIUS
	left_voltage = features["curL"] * MOTOR_RESISTANCE + MOTOR_BACK_EMF * left_wheel_omega
	right_voltage = features["curR"] * MOTOR_RESISTANCE + MOTOR_BACK_EMF * right_wheel_omega
	actual_power = (left_voltage * features["curL"]).abs() + (
		right_voltage * features["curR"]
	).abs()
	no_load_power = (left_voltage * NO_LOAD_CURRENT).abs() + (
		right_voltage * NO_LOAD_CURRENT
	).abs()
	features["Ke"] = actual_power / no_load_power.where(no_load_power.ne(0))
	return features


def filter_by_experiment(data: pd.DataFrame) -> pd.DataFrame:
	"""Apply a centered mean filter without crossing experiment boundaries."""
	filter_columns = [*MOTION_COLUMNS, *DERIVED_COLUMNS]
	filtered = data.copy()
	filtered[filter_columns] = (
		data.groupby(GROUP_COLUMNS, sort=False, group_keys=False)[filter_columns]
		.rolling(window=WINDOW_SIZE, center=True, min_periods=1)
		.mean()
		.reset_index(level=GROUP_COLUMNS, drop=True)
	)
	return filtered


def split_by_motion(data: pd.DataFrame, output_dir: Path) -> None:
	"""Save records split by turn direction and discretized angular velocity."""
	data = data.copy()
	data["motion_direction"] = data["omega"].map(
		lambda value: "left" if value > 0 else "right" if value < 0 else "straight"
	)
	data["omega_bin"] = (data["omega"] / OMEGA_BIN_WIDTH).round() * OMEGA_BIN_WIDTH
	data = data[data["omega"].between(-1, 1)].copy()

	output_dir.mkdir(exist_ok=True)
	for (direction, omega_bin), group in data.groupby(
		["motion_direction", "omega_bin"], sort=True
	):
		filename = f"{direction}_omega_{omega_bin:+.2f}.csv"
		group.to_csv(output_dir / filename, index=False)


def remove_ke_outliers(data: pd.DataFrame) -> pd.DataFrame:
	"""Remove Ke outliers independently for each terrain and omega bin."""
	cleaned = data.copy()
	cleaned["omega_bin"] = (cleaned["omega"] / OMEGA_BIN_WIDTH).round() * OMEGA_BIN_WIDTH
	grouped_ke = cleaned.groupby(["terrain", "omega_bin"])["Ke"]
	q1 = grouped_ke.transform("quantile", 0.25)
	q3 = grouped_ke.transform("quantile", 0.75)
	iqr = q3 - q1
	valid = cleaned["Ke"].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
	return cleaned.loc[valid].drop(columns="omega_bin")


def calculate_dct_coefficients(
	data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, DCTModel], dict[str, float]]:
	"""Calculate DCT models, coefficients, and standard deviations per terrain."""
	limited = data[data["omega"].between(-1, 1)].copy()
	limited["omega_bin"] = (
		limited["omega"] / OMEGA_BIN_WIDTH
	).round() * OMEGA_BIN_WIDTH
	median_ke = (
		limited.groupby(["terrain", "omega_bin"], as_index=False)["Ke"]
		.agg(Ke_median="median", Ke_std="std")
	)

	coefficient_rows = []
	grid_rows = []
	dct_models = {}
	dct_models_std = {}
	for terrain, terrain_medians in median_ke.groupby("terrain", sort=True):
		series = terrain_medians.set_index("omega_bin")["Ke_median"].reindex(OMEGA_GRID)
		series = series.interpolate().ffill().bfill()
		model = DCTModel(
			series.to_numpy(),
			cutoff_amount=N_DCT_COEFFICIENTS,
			range_min=OMEGA_GRID.min(),
			range_max=OMEGA_GRID.max(),
		)
		dct_models[terrain] = model
		dct_models_std[terrain] = float(terrain_medians["Ke_std"].mean())
		coefficients = model.coefficients
		coefficient_rows.extend(
			{"terrain": terrain, "coefficient_index": index, "coefficient": value}
			for index, value in enumerate(coefficients)
		)
		grid_rows.extend(
			{"terrain": terrain, "omega_bin": omega, "Ke_median": value, "Ke_std": terrain_medians.set_index("omega_bin")["Ke_std"].reindex(OMEGA_GRID).interpolate().ffill().bfill().loc[omega]}
			for omega, value in series.items()
		)

	return (
		pd.DataFrame(grid_rows),
		pd.DataFrame(coefficient_rows),
		dct_models,
		dct_models_std,
	)


def plot_median_ke(data: pd.DataFrame, output_dir: Path) -> None:
	"""Save one median Ke and standard deviation plot for each terrain."""
	output_dir.mkdir(exist_ok=True)
	for terrain in sorted(data["terrain"].unique()):
		terrain_data = data[data["terrain"] == terrain].sort_values("omega_bin")
		figure, axis = plt.subplots(figsize=(10, 5))
		axis.plot(terrain_data["omega_bin"], terrain_data["Ke_median"], label="Median Ke")
		model = DCTModel(
			terrain_data["Ke_median"].to_numpy(),
			range_min=terrain_data["omega_bin"].min(),
			range_max=terrain_data["omega_bin"].max(),
		)
		dct_curve = model.numpy_func(terrain_data["omega_bin"], scaled=True)
		axis.plot(
			terrain_data["omega_bin"],
			dct_curve,
			color="crimson",
			linestyle="--",
			linewidth=2.5,
			zorder=5,
			label=f"DCT-II approximation ({N_DCT_COEFFICIENTS} coeffs)",
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
		axis.grid(True, alpha=0.3)
		axis.legend()
		figure.tight_layout()
		figure.savefig(output_dir / f"{terrain}_median_ke.png", dpi=150)
		plt.close(figure)


def plot_all_terrains(data: pd.DataFrame, output_path: Path) -> None:
	"""Plot all terrain median curves, uncertainty bands, and DCT models."""
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
		color = colors.get(terrain, None)
		axis.plot(
			terrain_data["omega_bin"],
			terrain_data["Ke_median"],
			color=color,
			linewidth=1.5,
			label=f"{terrain} median",
		)
		axis.fill_between(
			terrain_data["omega_bin"],
			terrain_data["Ke_median"] - terrain_data["Ke_std"],
			terrain_data["Ke_median"] + terrain_data["Ke_std"],
			color=color,
			alpha=0.12,
		)
		model = DCTModel(
			terrain_data["Ke_median"].to_numpy(),
			range_min=terrain_data["omega_bin"].min(),
			range_max=terrain_data["omega_bin"].max(),
		)
		dct_curve = model.numpy_func(terrain_data["omega_bin"], scaled=True)
		axis.plot(
			terrain_data["omega_bin"],
			dct_curve,
			color=color,
			linestyle="--",
			linewidth=2,
			label=f"{terrain} DCT-II",
		)
	axis.set_title("Ke by angular velocity for all terrains")
	axis.set_xlabel("Angular velocity omega, rad/s")
	axis.set_ylabel("Ke")
	axis.set_xlim(-0.5, 0.5)
	axis.grid(True, alpha=0.3)
	axis.legend(ncol=2)
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


def plot_ke_by_discrete_directions(
	data: pd.DataFrame,
	dct_models: dict[str, DCTModel],
	output_path: Path,
) -> None:
	"""Plot the continuous DCT model for each terrain."""
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
		smooth_omega = np.linspace(-0.5, 0.5, 400)
		smooth_ke = dct_models[terrain].numpy_func(smooth_omega, scaled=True)
		smooth_std = np.interp(
			smooth_omega,
			terrain_data["omega_bin"],
			terrain_data["Ke_std"].fillna(0),
		)
		color = colors.get(terrain)
		axis.fill_between(
			smooth_omega,
			smooth_ke - smooth_std,
			smooth_ke + smooth_std,
			color=color,
			alpha=0.16,
		)
		axis.plot(
			smooth_omega,
			smooth_ke,
			linewidth=1.8,
			color=color,
			label=terrain,
		)
	axis.set_title("Ke depending on discrete motion direction")
	axis.set_xlabel("Discrete angular direction, omega bin (rad/s)")
	axis.set_ylabel("Ke")
	axis.set_xlim(-0.5, 0.5)
	axis.set_ylim(0, 15)
	axis.set_xticks(np.arange(-0.5, 0.51, 0.1))
	axis.grid(True, alpha=0.3)
	axis.legend(title="Surface")
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


def classify_experiments_by_dct(
	data: pd.DataFrame,
	dct_models: dict[str, DCTModel],
) -> pd.DataFrame:
	"""Classify each experiment by its mean Ke curve against DCT terrain models."""
	model_curves = {
		terrain: model.numpy_func(OMEGA_GRID, scaled=True)
		for terrain, model in dct_models.items()
	}

	experiment_means = (
		data[data["omega"].between(-1, 1)]
		.assign(omega_bin=lambda frame: (frame["omega"] / OMEGA_BIN_WIDTH).round() * OMEGA_BIN_WIDTH)
		.groupby(["terrain", "run_idx", "omega_bin"], as_index=False)["Ke"]
		.mean()
	)
	rows = []
	for (true_terrain, run_idx), experiment in experiment_means.groupby(
		["terrain", "run_idx"], sort=True
	):
		observed = experiment.set_index("omega_bin")["Ke"].reindex(OMEGA_GRID)
		observed = observed.interpolate().ffill().bfill().to_numpy()
		scores = {
			terrain: float(np.sqrt(np.mean((observed - curve) ** 2)))
			for terrain, curve in model_curves.items()
		}
		rows.append(
			{
				"terrain": true_terrain,
				"run_idx": run_idx,
				"predicted_terrain": min(scores, key=scores.get),
				**{f"rmse_{terrain}": score for terrain, score in scores.items()},
			}
		)
	return pd.DataFrame(rows)


def select_dct_coefficients(
	data: pd.DataFrame,
	candidates: list[int],
	n_splits: int = 5,
) -> tuple[int, pd.DataFrame]:
	"""Select the DCT cutoff using grouped cross-validation by experiment."""
	groups = data["terrain"].astype(str) + "::" + data["run_idx"].astype(str)
	n_splits = min(n_splits, groups.nunique())
	if n_splits < 2:
		raise ValueError("At least two experiments are required for cross-validation")

	folds = list(GroupKFold(n_splits=n_splits).split(data, groups=groups))
	original_cutoff = N_DCT_COEFFICIENTS
	rows = []
	try:
		for cutoff in candidates:
			globals()["N_DCT_COEFFICIENTS"] = cutoff
			fold_scores = []
			for train_indices, test_indices in folds:
				train_data = data.iloc[train_indices]
				test_data = data.iloc[test_indices]
				_, _, models, _ = calculate_dct_coefficients(train_data)
				predictions = classify_experiments_by_dct(test_data, models)
				if predictions.empty:
					continue
				report = classification_report(
					predictions["terrain"],
					predictions["predicted_terrain"],
					output_dict=True,
					zero_division=0,
				)
				fold_scores.append(
					{
						"accuracy": accuracy_score(
							predictions["terrain"], predictions["predicted_terrain"]
						),
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
	finally:
		globals()["N_DCT_COEFFICIENTS"] = original_cutoff

	results = pd.DataFrame(rows).sort_values(
		["cv_macro_f1", "cv_accuracy"], ascending=False
	)
	return int(results.iloc[0]["n_coefficients"]), results


def save_classification_report(classification: pd.DataFrame, output_dir: Path) -> float:
	"""Save sklearn classification metrics and return classification accuracy."""
	labels = sorted(
		set(classification["terrain"]) | set(classification["predicted_terrain"])
	)
	y_true = classification["terrain"]
	y_pred = classification["predicted_terrain"]
	accuracy = accuracy_score(y_true, y_pred)
	report = classification_report(
		y_true,
		y_pred,
		labels=labels,
		target_names=labels,
		zero_division=0,
	)
	report_dict = classification_report(
		y_true,
		y_pred,
		labels=labels,
		target_names=labels,
		output_dict=True,
		zero_division=0,
	)
	(output_dir / "classification_report.txt").write_text(
		f"Accuracy: {accuracy:.6f}\n\n{report}", encoding="utf-8"
	)
	pd.DataFrame(report_dict).transpose().to_csv(
		output_dir / "classification_report.csv"
	)
	(output_dir / "classification_accuracy.txt").write_text(
		f"{accuracy:.6f}\n", encoding="utf-8"
	)
	return float(accuracy)


def plot_confusion_matrix(classification: pd.DataFrame, output_dir: Path) -> None:
	"""Save raw and row-normalized DCT classification error maps."""
	labels = sorted(
		set(classification["terrain"]) | set(classification["predicted_terrain"])
	)
	matrix = confusion_matrix(
		classification["terrain"], classification["predicted_terrain"], labels=labels
	).astype(float)
	row_totals = matrix.sum(axis=1, keepdims=True)
	normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)

	matrix_path = output_dir / "dct_confusion_matrix.csv"
	pd.DataFrame(normalized, index=labels, columns=labels).to_csv(matrix_path)
	pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
		output_dir / "dct_confusion_matrix_counts.csv"
	)

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
	figure.savefig(output_dir / "dct_confusion_matrix.png", dpi=150)
	plt.close(figure)


def main() -> None:
	data_dir = Path(__file__).parent / "data" / "borealtc"
	input_path = data_dir / "merged_interpolation.csv"
	output_path = data_dir / "merged_interpolation_features_filtered.csv"

	data = pd.read_csv(input_path)
	filtered = filter_by_experiment(calculate_features(data))
	cleaned = remove_ke_outliers(filtered)
	cleaned.to_csv(output_path, index=False)
	print(f"Saved filtered dataframe to {output_path}")
	best_cutoff, cv_results = select_dct_coefficients(cleaned, list(range(3, 21)))
	globals()["N_DCT_COEFFICIENTS"] = best_cutoff
	cv_results_path = data_dir / "dct_coefficient_cv_results.csv"
	cv_results.to_csv(cv_results_path, index=False)
	print(f"Selected {best_cutoff} DCT coefficients by grouped CV")
	median_ke, coefficients, dct_models, dct_models_std = calculate_dct_coefficients(cleaned)
	median_path = data_dir / "median_ke_by_omega.csv"
	coefficients_path = data_dir / "dct2_coefficients.csv"
	dct_models_path = data_dir / "dct_models.pkl"
	plot_dir = data_dir / "median_ke_plots"
	all_plot_path = data_dir / "all_terrains_ke_dct.png"
	discrete_directions_plot_path = data_dir / "ke_by_discrete_directions.png"
	median_ke.to_csv(median_path, index=False)
	coefficients.to_csv(coefficients_path, index=False)
	with dct_models_path.open("wb") as file:
		pickle.dump((dct_models, dct_models_std), file)
	plot_median_ke(median_ke, plot_dir)
	plot_all_terrains(median_ke, all_plot_path)
	plot_ke_by_discrete_directions(
		median_ke, dct_models, discrete_directions_plot_path
	)
	classification = classify_experiments_by_dct(cleaned, dct_models)
	classification_path = data_dir / "dct_experiment_classification.csv"
	classification.to_csv(classification_path, index=False)
	accuracy = save_classification_report(classification, data_dir)
	plot_confusion_matrix(classification, data_dir)
	print(f"Saved median Ke values to {median_path}")
	print(f"Saved DCT-II coefficients to {coefficients_path}")
	print(f"Saved DCT models to {dct_models_path}")
	print(f"Saved DCT coefficient CV results to {cv_results_path}")
	print(f"Saved median Ke plots to {plot_dir}")
	print(f"Saved all-terrain Ke plot to {all_plot_path}")
	print(f"Saved discrete-direction Ke plot to {discrete_directions_plot_path}")
	print(f"Saved DCT classification to {classification_path}")
	print(f"Saved classification report to {data_dir / 'classification_report.txt'}")
	print(f"Classification accuracy: {accuracy:.4f}")
	split_by_motion(cleaned, data_dir / "motion_splits")
	print(f"Saved motion splits to {data_dir / 'motion_splits'}")


if __name__ == "__main__":
	main()
