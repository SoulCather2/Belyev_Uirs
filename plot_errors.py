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

    figure = px.scatter(data_frame=data, x=x, y=y, **kwargs)

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
        error_name="Ошибка классификатора", **kwargs,
    )
    fig.update_yaxes(matches=None)
    fig.update_traces(marker_size=4)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
