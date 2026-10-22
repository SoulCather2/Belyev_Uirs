"""Print saved DCT terrain models as analytic LaTeX formulas."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>DCT terrain models</title>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
    <style>
        :root {{ color-scheme: light; font-family: Georgia, serif; }}
        body {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px; color: #1d2939; background: #f5f7fa; }}
        h1 {{ margin: 0 0 24px; font-size: 28px; font-weight: 600; }}
        section {{ margin: 18px 0; padding: 24px; overflow-x: auto; background: white; border: 1px solid #d9e0e8; border-radius: 10px; box-shadow: 0 5px 18px #1d29390d; }}
        section h2 {{ margin: 0 0 16px; font: 600 16px ui-sans-serif, system-ui, sans-serif; letter-spacing: .04em; text-transform: uppercase; color: #52606d; }}
        .formula {{ min-width: max-content; font-size: 1.08rem; }}
    </style>
</head>
<body>
    <h1>DCT terrain models</h1>
{sections}
</body>
</html>
"""


def format_number(value: float) -> str:
    """Format formula numbers with exactly two decimal places."""
    return f"{value:.2f}"


def latex_surface_name(surface: str) -> str:
    """Render a surface name safely as upright text in a math formula."""
    escaped = surface.replace("\\", r"\textbackslash{}").replace("_", r"\_")
    return rf"\mathrm{{{escaped}}}"


def signed_term(coefficient: float, expression: str) -> str:
    """Return a coefficient term with its leading plus sign."""
    return f" + {format_number(abs(coefficient))} {expression}"


def model_formula(
    surface: str,
    coefficients: pd.DataFrame,
    zero_tolerance: float,
) -> str:
    """Build one analytic DCT formula for a terrain surface."""
    coefficients = coefficients.sort_values("coefficient_index")
    values = coefficients["coefficient"].to_numpy(dtype=float)
    indices = coefficients["coefficient_index"].to_numpy(dtype=int)
    size = len(values)
    if size == 0:
        raise ValueError(f"No DCT coefficients found for {surface!r}")

    nonzero = ~np.isclose(values, 0.0, atol=zero_tolerance, rtol=0.0)
    if not nonzero.any():
        raise ValueError(f"All DCT coefficients are zero for {surface!r}")

    terms: list[str] = []
    for index, coefficient, present in zip(indices, values, nonzero):
        if not present:
            continue
        if index == 0:
            terms.append(rf"\frac{{{format_number(coefficient)}}}{{2}}")
            continue
        numerator = r"\pi" if index == 1 else rf"{index}\pi"
        cosine = (
            rf"\cos\left[\frac{{{numerator}}}{{{size}}}"
            rf"\left(\alpha + \frac{{1}}{{2}}\right)\right]"
            )
        terms.append(signed_term(coefficient, cosine))

    body = "".join(terms)
    name = latex_surface_name(surface)
    return (
        rf"\begin{{equation}}\label{{eq:{surface.lower()}}}"
        rf"f^{{{name}}}(\alpha) = \frac{{1}}{{{size}}}\left[ {body} \right]"
        r"\end{equation}"
    )


def load_formulas(data_dir: Path, zero_tolerance: float) -> list[str]:
    """Load model coefficients and create one formula per surface."""
    coefficients_path = data_dir / "dct2_coefficients.csv"
    grid_path = data_dir / "median_ke_by_omega.csv"
    coefficients = pd.read_csv(coefficients_path)
    grid = pd.read_csv(grid_path)

    formulas = []
    for surface, surface_coefficients in coefficients.groupby("terrain", sort=True):
        formulas.append(
        model_formula(str(surface), surface_coefficients, zero_tolerance)
        )
    return formulas


def render_html(formulas: list[str]) -> str:
    """Wrap equation environments in a browser-renderable MathJax page."""
    sections = []
    for formula in formulas:
        label_start = formula.index(r"\label{eq:")
        label_end = formula.index("}", label_start) + 1
        label = formula[label_start + len(r"\label{eq:") : label_end - 1]
        display_formula = formula[label_end : formula.rindex(r"\end{equation}")]
        sections.append(
            f'  <section><h2>{html.escape(label)}</h2>'
            f'<div class="formula">\\[{html.escape(display_formula)}\\]</div></section>'
        )
    return HTML_TEMPLATE.format(sections="\n".join(sections))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print saved DCT models as analytic LaTeX formulas."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "borealtc",
        help="Directory containing dct2_coefficients.csv and median_ke_by_omega.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write rendered HTML formulas to this path instead of printing fragments.",
    )
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-12,
        help="Absolute tolerance for treating a coefficient as zero.",
    )
    args = parser.parse_args()

    formulas = load_formulas(args.data_dir, args.zero_tolerance)
    if args.output is None:
        print("\n\n".join(formulas))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() != ".html":
        raise ValueError("--output must use the .html extension")
    args.output.write_text(render_html(formulas), encoding="utf-8")
    print(f"Saved rendered HTML formulas to {args.output}")


if __name__ == "__main__":
    main()
