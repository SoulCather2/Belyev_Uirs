"""Build and compare analytic DCT formulas for the best Optuna trial."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from Major import (
    PipelineConfig,
    build_symmetric_grid,
    calculate_dct_coefficients,
    calculate_features,
    filter_by_experiment,
    remove_ke_outliers,
    split_by_motion,
)
from print_dct_models import load_formulas, model_formula, render_html


DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "borealtc"


def load_best_config(data_dir: Path) -> tuple[PipelineConfig, dict]:
    """Load effective integer/float parameters saved by Optuna."""
    result_path = data_dir / "best_dct_hyperparameters.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    parameters = result["parameters"]
    config = replace(
        PipelineConfig(),
        window_size=int(parameters["window_size"]),
        n_bins=int(parameters["n_bins"]),
        n_dct_coefficients=int(parameters["n_dct_coefficients"]),
        memory_margin=float(parameters["memory_margin"]),
    )
    return config, result


def train_best_models(data_dir: Path) -> tuple[dict, dict, PipelineConfig, dict]:
    """Rebuild models using exactly the saved best-trial configuration."""
    config, result = load_best_config(data_dir)
    raw_data = pd.read_csv(data_dir / "merged_interpolation.csv")
    raw_data = raw_data[raw_data["terrain"] != "SANDY_LOAM"].copy()
    features = calculate_features(raw_data, config)
    grid, omega_limit = build_symmetric_grid(features, config)
    filtered = filter_by_experiment(features, config)
    cleaned = remove_ke_outliers(filtered, config)
    split_data = split_by_motion(cleaned, grid, omega_limit)
    training_data = pd.concat(split_data.values(), ignore_index=True)
    _, _, models, _ = calculate_dct_coefficients(
        training_data, config, grid, omega_limit
    )
    return models, cleaned, config, result


def coefficient_table(models: dict) -> pd.DataFrame:
    """Return all DCT coefficients and their selected/nonzero status."""
    rows = []
    for surface, model in sorted(models.items()):
        for index, coefficient in enumerate(model.coefficients):
            rows.append(
                {
                    "terrain": surface,
                    "coefficient_index": index,
                    "coefficient": float(coefficient),
                    "is_nonzero": bool(coefficient != 0.0),
                }
            )
    return pd.DataFrame(rows)


def summarize_similarity(coefficients: pd.DataFrame) -> str:
    """Describe common DCT positions and pairwise overlap."""
    positions = {
        surface: set(
            group.loc[group["is_nonzero"], "coefficient_index"].astype(int)
        )
        for surface, group in coefficients.groupby("terrain", sort=True)
    }
    surfaces = list(positions)
    common = set.intersection(*positions.values()) if positions else set()
    union = set.union(*positions.values()) if positions else set()

    lines = [
        "DCT basis-position similarity",
        "==============================",
        f"Surfaces: {', '.join(surfaces)}",
        f"Total available positions: {int(coefficients['coefficient_index'].max()) + 1}",
        f"Union of nonzero positions: {sorted(union)}",
        f"Positions nonzero on every surface: {sorted(common)}",
        "",
        "Selected positions by surface:",
    ]
    for surface in surfaces:
        lines.append(f"  {surface}: {sorted(positions[surface])}")

    lines.extend(["", "Pairwise overlap (shared / union, Jaccard):"])
    for left_index, left in enumerate(surfaces):
        for right in surfaces[left_index + 1 :]:
            shared = len(positions[left] & positions[right])
            total = len(positions[left] | positions[right])
            jaccard = shared / total if total else 1.0
            lines.append(f"  {left} vs {right}: {shared} / {total} ({jaccard:.3f})")

    lines.extend(
        [
            "",
            "Interpretation:",
            "  Every term has the same DCT multiplier k*pi/N because N is shared.",
            "  Surfaces differ only when their selected position k or coefficient value differs.",
        ]
    )
    return "\n".join(lines) + "\n"


def cosine_coefficients_text(coefficients: pd.DataFrame) -> str:
    """Format only the nonzero coefficients multiplying cosine terms."""
    lines = [
        "Nonzero DCT coefficients before cosine terms",
        "=============================================",
        "",
    ]
    cosine_coefficients = coefficients[
        (coefficients["coefficient_index"] > 0) & coefficients["is_nonzero"]
    ]
    size = int(coefficients["coefficient_index"].max()) + 1
    for surface, group in cosine_coefficients.groupby("terrain", sort=True):
        lines.append(surface)
        lines.append("-" * len(surface))
        for row in group.itertuples(index=False):
            lines.append(
                f"k={row.coefficient_index:2d} | coefficient={row.coefficient:.10g} "
                f"| cosine multiplier={row.coefficient_index}*pi/{size}"
            )
        lines.append("")
    return "\n".join(lines)


def add_formula_label_prefix(formulas: list[str], prefix: str) -> list[str]:
    """Make baseline and optimized formula sections distinguishable in HTML."""
    return [
        formula.replace(r"\label{eq:", rf"\label{{eq:{prefix}_", 1)
        for formula in formulas
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show and compare formulas from the best Optuna DCT classifier."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--zero-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    models, _, config, result = train_best_models(args.data_dir)
    coefficients = coefficient_table(models)
    coefficients.to_csv(args.data_dir / "best_dct2_coefficients.csv", index=False)
    cosine_coefficients = coefficients[
        (coefficients["coefficient_index"] > 0) & coefficients["is_nonzero"]
    ].copy()
    baseline_coefficients = pd.read_csv(args.data_dir / "dct2_coefficients.csv")
    baseline_coefficients = baseline_coefficients[
        (baseline_coefficients["coefficient_index"] > 0)
        & (baseline_coefficients["coefficient"] != 0.0)
    ].copy()
    baseline_coefficients["model"] = "baseline"
    cosine_coefficients["model"] = f"best_trial_{result['trial']}"
    baseline_size = int(
        pd.read_csv(args.data_dir / "median_ke_by_omega.csv")
        .groupby("terrain")
        .size()
        .max()
    )
    baseline_coefficients["cosine_multiplier"] = baseline_coefficients[
        "coefficient_index"
    ].map(lambda index: f"{index}*pi/{baseline_size}")
    cosine_coefficients["cosine_multiplier"] = cosine_coefficients[
        "coefficient_index"
    ].map(lambda index: f"{index}*pi/{config.n_bins}")
    combined_coefficients = pd.concat(
        [baseline_coefficients, cosine_coefficients], ignore_index=True, sort=False
    )
    baseline_count = len(baseline_coefficients)
    combined_coefficients["trial"] = (
        [pd.NA] * baseline_count
        + [int(result["trial"])] * len(cosine_coefficients)
    )
    combined_coefficients["trial_accuracy"] = (
        [pd.NA] * baseline_count
        + [float(result["accuracy"])] * len(cosine_coefficients)
    )
    combined_coefficients[
        [
            "model",
            "trial",
            "trial_accuracy",
            "terrain",
            "coefficient_index",
            "coefficient",
            "cosine_multiplier",
        ]
    ].to_csv(args.data_dir / "best_dct_cosine_coefficients.csv", index=False)
    cosine_text = (
        f"Best Optuna trial: {result['trial']}\n"
        f"Accuracy: {result['accuracy']:.6f}\n"
        f"Configuration: window_size={config.window_size}, n_bins={config.n_bins}, "
        f"n_dct_coefficients={config.n_dct_coefficients}, "
        f"memory_margin={config.memory_margin:.10g}\n\n"
        + cosine_coefficients_text(coefficients)
    )
    (args.data_dir / "best_dct_cosine_coefficients.txt").write_text(
        cosine_text, encoding="utf-8"
    )

    formulas = [
        model_formula(
            surface,
            group.rename(columns={"is_nonzero": "is_nonzero"}),
            float(model.omega_grid.min()),
            float(model.omega_grid.max()),
            args.zero_tolerance,
        )
        for surface, model in sorted(models.items())
        for group in [
            coefficients.loc[coefficients["terrain"] == surface].rename(
                columns={"terrain": "terrain"}
            )
        ]
    ]
    baseline_formulas = load_formulas(args.data_dir, args.zero_tolerance)
    (args.data_dir / "best_dct_formulas.html").write_text(
        render_html(formulas), encoding="utf-8"
    )
    summary = summarize_similarity(coefficients)
    (args.data_dir / "best_dct_formula_similarity.txt").write_text(
        summary, encoding="utf-8"
    )
    combined_formulas = add_formula_label_prefix(baseline_formulas, "baseline")
    combined_formulas.extend(add_formula_label_prefix(formulas, "best_trial"))
    html_page = render_html(combined_formulas).replace(
        "<h1>DCT terrain models</h1>",
        f"<h1>Baseline models and best Optuna trial {result['trial']} &middot; accuracy {result['accuracy']:.4f}</h1>",
    )
    (Path(__file__).parent / "formulas.html").write_text(html_page, encoding="utf-8")

    print(f"Accuracy from best trial: {result['accuracy']:.6f}")
    print(
        "Configuration: "
        f"window_size={config.window_size}, n_bins={config.n_bins}, "
        f"n_dct_coefficients={config.n_dct_coefficients}, "
        f"memory_margin={config.memory_margin:.10g}"
    )
    print(summary)
    print(f"Saved formulas to {args.data_dir / 'best_dct_formulas.html'}")
    print(f"Saved formulas and similarity analysis to {Path(__file__).parent / 'formulas.html'}")


if __name__ == "__main__":
    main()
