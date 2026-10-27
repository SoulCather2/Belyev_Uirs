"""Compare nonzero DCT coefficient positions between terrain surfaces.

For both the baseline models and the best Optuna models, the script
  1. lists the sorted indices of nonzero coefficients for every surface,
  2. groups surfaces that use exactly the same indices,
  3. builds a surface x surface matrix (green = same index set,
     red = different) as an interactive Plotly HTML file.

The hover text of each cell shows how many indices differ and which ones
(symmetric difference / XOR of the two index sets).
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


DEFAULT_DATA_DIR = Path(__file__).parent / "data" / "borealtc"
MODEL_FILES = {
    "baseline": "dct2_coefficients.csv",
    "best": "best_dct2_coefficients.csv",
}
GREEN = "#2ca02c"
RED = "#d62728"
GREEN_CELL = "#b7e4c7"
RED_CELL = "#f5b7b1"


def nonzero_positions(coefficients: pd.DataFrame, tolerance: float) -> dict[str, list[int]]:
    """Return sorted nonzero coefficient indices (including index 0) per surface."""
    positions: dict[str, list[int]] = {}
    for surface, group in coefficients.groupby("terrain", sort=True):
        mask = group["coefficient"].abs() > tolerance
        positions[str(surface)] = sorted(group.loc[mask, "coefficient_index"].astype(int))
    return positions


def text_report(title: str, positions: dict[str, list[int]]) -> str:
    """Describe the index sets and which surfaces share exactly the same one."""
    lines = [title, "=" * len(title), ""]
    for surface, indices in positions.items():
        lines.append(f"{surface} ({len(indices)}): {indices}")

    groups: dict[tuple[int, ...], list[str]] = {}
    for surface, indices in positions.items():
        groups.setdefault(tuple(indices), []).append(surface)

    lines.append("")
    if len(groups) == 1:
        lines.append("All surfaces use exactly the same coefficient indices.")
    else:
        lines.append("Surfaces with identical index sets:")
        for indices, surfaces in groups.items():
            lines.append(f"  {', '.join(surfaces)}: {list(indices)}")
        lines.append("Index sets differ between surfaces.")
    return "\n".join(lines) + "\n"


def build_matrix(title: str, positions: dict[str, list[int]]) -> go.Figure:
    """Build a green/red surface x surface heatmap with XOR details on hover."""
    surfaces = list(positions)
    sets = {surface: set(indices) for surface, indices in positions.items()}

    status, counts, hover = [], [], []
    for row in surfaces:
        status_row, count_row, hover_row = [], [], []
        for column in surfaces:
            xor = sorted(sets[row] ^ sets[column])
            only_row = sorted(sets[row] - sets[column])
            only_column = sorted(sets[column] - sets[row])
            status_row.append(0 if not xor else 1)
            count_row.append(len(xor))
            hover_row.append(
                f"<b>{row} vs {column}</b><br>"
                f"Differing coefficients: {len(xor)}<br>"
                f"Numbers (XOR): {xor if xor else '-'}<br>"
                f"Only in {row}: {only_row if only_row else '-'}<br>"
                f"Only in {column}: {only_column if only_column else '-'}"
            )
        status.append(status_row)
        counts.append(count_row)
        hover.append(hover_row)

    figure = go.Figure(
        go.Heatmap(
            z=status,
            x=surfaces,
            y=surfaces,
            zmin=0,
            zmax=1,
            colorscale=[[0.0, GREEN], [0.5, GREEN], [0.5, RED], [1.0, RED]],
            showscale=False,
            xgap=3,
            ygap=3,
            text=counts,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            f"{title}<br><sup>Green: identical nonzero indices. Red: different. "
            "Number in a cell: count of differing indices.</sup>"
        ),
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed", "scaleanchor": "x"},
        width=620,
        height=640,
        template="plotly_white",
    )
    return figure


def build_cross_model_matrix(
    baseline: dict[str, list[int]], best: dict[str, list[int]]
) -> go.Figure:
    """Compare every baseline surface formula with every best-model formula."""
    baseline_surfaces = list(baseline)
    best_surfaces = list(best)
    baseline_sets = {surface: set(indices) for surface, indices in baseline.items()}
    best_sets = {surface: set(indices) for surface, indices in best.items()}

    status, counts, hover = [], [], []
    for baseline_surface in baseline_surfaces:
        status_row, count_row, hover_row = [], [], []
        for best_surface in best_surfaces:
            xor = sorted(
                baseline_sets[baseline_surface] ^ best_sets[best_surface]
            )
            only_baseline = sorted(
                baseline_sets[baseline_surface] - best_sets[best_surface]
            )
            only_best = sorted(
                best_sets[best_surface] - baseline_sets[baseline_surface]
            )
            status_row.append(0 if not xor else 1)
            count_row.append(len(xor))
            hover_row.append(
                f"<b>{baseline_surface} (baseline) vs {best_surface} (best)</b><br>"
                f"Differing coefficients: {len(xor)}<br>"
                f"Numbers (XOR): {xor if xor else '-'}<br>"
                f"Only in baseline: {only_baseline if only_baseline else '-'}<br>"
                f"Only in best: {only_best if only_best else '-'}"
            )
        status.append(status_row)
        counts.append(count_row)
        hover.append(hover_row)

    figure = go.Figure(
        go.Heatmap(
            z=status,
            x=[f"{surface} (best)" for surface in best_surfaces],
            y=[f"{surface} (baseline)" for surface in baseline_surfaces],
            zmin=0,
            zmax=1,
            colorscale=[[0.0, GREEN], [0.5, GREEN], [0.5, RED], [1.0, RED]],
            showscale=False,
            xgap=3,
            ygap=3,
            text=counts,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    figure.update_layout(
        title=(
            "Baseline vs best classifier: nonzero DCT coefficient indices"
            "<br><sup>Rows: baseline surfaces. Columns: best-model surfaces. "
            "Green: identical index sets. Red: different. "
            "Cell value: number of XOR indices.</sup>"
        ),
        xaxis={"side": "top"},
        yaxis={"autorange": "reversed", "scaleanchor": "x"},
        width=780,
        height=700,
        template="plotly_white",
    )
    return figure


def build_model_comparison(
    baseline: dict[str, list[int]], best: dict[str, list[int]], data_dir: Path
) -> None:
    """Write a per-surface, per-index baseline-versus-best comparison matrix."""
    indices = sorted(
        {
            index
            for model_positions in (baseline, best)
            for selected in model_positions.values()
            for index in selected
        }
    )
    matrix_path = data_dir / "dct_coefficient_position_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["terrain", *indices])
        for surface in baseline:
            baseline_indices = set(baseline[surface])
            best_indices = set(best[surface])
            row = []
            for index in indices:
                in_baseline = index in baseline_indices
                in_best = index in best_indices
                if in_baseline and in_best:
                    row.append("same_selected")
                elif in_baseline:
                    row.append("baseline_only")
                elif in_best:
                    row.append("best_only")
                else:
                    row.append("neither_selected")
            writer.writerow([surface, *row])

    table_rows = []
    for surface in baseline:
        baseline_indices = set(baseline[surface])
        best_indices = set(best[surface])
        cells = []
        for index in indices:
            in_baseline = index in baseline_indices
            in_best = index in best_indices
            same = in_baseline == in_best
            state = (
                "selected in both"
                if in_baseline and in_best
                else "baseline only"
                if in_baseline
                else "best only"
                if in_best
                else "selected in neither"
            )
            marker = (
                "X"
                if in_baseline and in_best
                else "B"
                if in_baseline
                else "O"
                if in_best
                else "-"
            )
            color = GREEN_CELL if same else RED_CELL
            cells.append(
                f'<td style="background:{color}" title="{html.escape(state)}">'
                f"{marker}</td>"
            )
        table_rows.append(f"<tr><th>{html.escape(surface)}</th>{''.join(cells)}</tr>")

    headers = "".join(f"<th>{index}</th>" for index in indices)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>DCT coefficient position comparison</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-size: 12px; }}
th, td {{ border: 1px solid #bbb; min-width: 32px; height: 28px; text-align: center; }}
th {{ background: #eef2f5; position: sticky; top: 0; }}
td {{ cursor: help; }}
</style></head><body>
<h1>DCT nonzero coefficient positions: baseline vs best</h1>
<p>Green: coefficient position matches. Red: position differs. X = selected in both; B = baseline only; O = best only; - = selected in neither. Hover over a cell for details.</p>
<div class="table-wrap"><table><tr><th>Surface</th>{headers}</tr>
{''.join(table_rows)}
</table></div></body></html>
"""
    html_path = data_dir / "dct_coefficient_position_matrix.html"
    html_path.write_text(page, encoding="utf-8")
    print(f"Saved {matrix_path}")
    print(f"Saved {html_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare nonzero DCT coefficient positions between surfaces."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--zero-tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    model_positions = {}
    for model, filename in MODEL_FILES.items():
        path = args.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. For 'best', run the best-trial formula script first."
            )
        model_positions[model] = nonzero_positions(
            pd.read_csv(path), args.zero_tolerance
        )

    shared_surfaces = set.intersection(
        *(set(positions) for positions in model_positions.values())
    )
    if not shared_surfaces:
        raise ValueError("Baseline and best models have no surfaces in common.")
    model_positions = {
        model: {
            surface: positions[surface]
            for surface in sorted(shared_surfaces)
        }
        for model, positions in model_positions.items()
    }

    reports = []
    for model, positions in model_positions.items():
        title = f"{model.capitalize()} models: nonzero DCT coefficient indices"
        reports.append(text_report(title, positions))

        html_path = args.data_dir / f"dct_position_matrix_{model}.html"
        build_matrix(title, positions).write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved {html_path}")

    cross_model_path = args.data_dir / "dct_position_matrix_baseline_vs_best.html"
    build_cross_model_matrix(
        model_positions["baseline"], model_positions["best"]
    ).write_html(cross_model_path, include_plotlyjs="cdn")
    print(f"Saved {cross_model_path}")

    report = "\n".join(reports)
    report_path = args.data_dir / "dct_position_comparison_by_surface.txt"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {report_path}")

    baseline = model_positions["baseline"]
    best = model_positions["best"]
    build_model_comparison(baseline, best, args.data_dir)
    comparison_lines = [
        "DCT nonzero-position comparison",
        "================================",
        "Green: baseline and best have the same position state.",
        "Red: baseline and best differ.",
        "",
    ]
    for surface in baseline:
        comparison_lines.extend(
            [
                f"{surface} baseline: {baseline[surface]}",
                f"{surface} best:     {best[surface]}",
            ]
        )
    comparison_path = args.data_dir / "dct_coefficient_position_comparison.txt"
    comparison_path.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
    print(f"Saved {comparison_path}")


if __name__ == "__main__":
    main()