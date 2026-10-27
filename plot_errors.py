from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

_EPOCH = pd.Timestamp("1970-01-01")


def plot_scatter_with_errors(
    data: pd.DataFrame,
    x: str,
    y: str | Sequence[str],
    wrong_cls: str,
    *,
    error_name: str | None = None,
    error_color: str = "red",
    error_opacity: float = 0.3,
    share_y: bool = True,
    **kwargs: Any,
) -> go.Figure:
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
            dt = dt.dt.tz_localize(None)
        coordinates = ((dt - _EPOCH) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        datetime_axis = True
    elif is_numeric_dtype(x_values.dtype) and not is_bool_dtype(x_values.dtype):
        coordinates = x_values.to_numpy(dtype=float, na_value=np.nan)
        datetime_axis = False
    else:
        raise TypeError(f"{x!r} must be numeric or datetime to shade intervals")

    kwargs.setdefault("template", "plotly_white")
    figure = px.scatter(data_frame=data, x=x, y=y, **kwargs)
    figure.update_layout(
        margin=dict(l=10, r=10, t=90 if figure.layout.title.text else 40, b=10),
        title=dict(x=0, xanchor="left", xref="paper", y=1, yanchor="top", yref="container", pad_t=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, title_text=""),
    )
    if not share_y:
        figure.update_yaxes(matches=None)

    valid = np.isfinite(coordinates)
    points = pd.DataFrame({"c": coordinates[valid], "err": errors[valid]})
    if points.empty:
        return figure
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
    ends = np.flatnonzero(diff == -1)
    intervals = [(edges[s], edges[e]) for s, e in zip(starts, ends)]

    def axis_value(value: float) -> pd.Timestamp | float:
        return _EPOCH + pd.to_timedelta(value, unit="s") if datetime_axis else float(value)

    # Bands are filled traces, not layout shapes: plotly.js toggles a legendgroup
    # of traces together, but a shape legend item only hides its own shape.
    y_by_axes: dict[tuple[str, str], list[np.ndarray]] = {}
    for trace in figure.data:
        values = pd.to_numeric(pd.Series(trace.y), errors="coerce").to_numpy(dtype=float)
        y_by_axes.setdefault((trace.xaxis or "x", trace.yaxis or "y"), []).append(values)

    def layout_name(ref: str) -> str:
        return ref[0] + "axis" + ref[1:]

    groups: dict[str, list[str]] = {}
    for _, yref in y_by_axes:
        matches = figure.layout[layout_name(yref)].matches
        groups.setdefault(matches or yref, []).append(yref)
    ranges = {}
    for refs in groups.values():
        values = np.concatenate([v for (_, yr), vs in y_by_axes.items() if yr in refs for v in vs])
        values = values[np.isfinite(values)]
        lo, hi = (values.min(), values.max()) if values.size else (0.0, 1.0)
        pad = (hi - lo) * 0.05 or 0.5
        for ref in refs:
            ranges[ref] = (lo - pad, hi + pad)

    band_x = []
    for left, right in intervals:
        band_x += [axis_value(left), axis_value(left), axis_value(right), axis_value(right), axis_value(left), None]
    label = error_name or wrong_cls
    bands = []
    for i, (xref, yref) in enumerate(y_by_axes):
        lo, hi = ranges[yref]
        bands.append(
            go.Scatter(
                x=band_x,
                y=[lo, hi, hi, lo, lo, None] * len(intervals),
                xaxis=xref,
                yaxis=yref,
                mode="none",
                fill="toself",
                fillcolor=error_color,
                opacity=error_opacity,
                name=label,
                legendgroup="classification_errors",
                legendrank=10_000,
                showlegend=i == 0,
                hoverinfo="skip",
            )
        )
        figure.layout[layout_name(yref)].range = [lo, hi]
    count = len(figure.data)
    figure.add_traces(bands)
    figure.data = figure.data[count:] + figure.data[:count]
    return figure


def plot_run(
    data: pd.DataFrame,
    terrain: str,
    run_idx: int,
    y: Sequence[str] = ("omega", "Ke", "speed"),
    x: str = "time",
    wrong_cls: str = "wrong",
    **kwargs: Any,
) -> go.Figure:
    """Plot one run of the predictions dataset: one panel per signal, errors shaded."""
    part = data[(data["terrain"] == terrain) & (data["run_idx"] == run_idx)]
    if part.empty:
        raise ValueError(f"No rows for terrain={terrain!r}, run_idx={run_idx}")
    signals = [c for c in y if c in part.columns]
    long = part.melt(id_vars=[x, wrong_cls], value_vars=signals, var_name="signal", value_name="value")
    kwargs.setdefault("title", f"Поверхность: {terrain}, прогон {run_idx} (ошибок {part[wrong_cls].mean():.0%})")
    fig = plot_scatter_with_errors(
        long, x=x, y="value", wrong_cls=wrong_cls, facet_row="signal", color="signal",
        error_name="Ошибка классификатора", share_y=False, **kwargs,
    )
    fig.update_traces(marker_size=4)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
