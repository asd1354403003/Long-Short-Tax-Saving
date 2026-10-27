"""Local cross-sectional operators. These do not call WorldQuant BRAIN."""
from __future__ import annotations

import numpy as np
import pandas as pd


FACTORS = ("value", "quality", "momentum")
OPERATORS = {"raw", "rank", "zscore", "group_rank", "group_zscore", "group_mean"}


def _rank(values):
    """Percentile rank: 0 is lowest, 1 is highest; tied values share a rank."""
    if len(values) <= 1:
        return pd.Series(0.5, index=values.index)
    return (values.rank(method="average") - 1) / (len(values) - 1)


def _zscore(values):
    deviation = values.std(ddof=0)
    if not np.isfinite(deviation) or deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / deviation


def transform(values, pipeline, groups):
    """Apply operators in order, within the selected group only for group_*.

    group_mean is an equally weighted arithmetic group mean broadcast to each
    stock. vec_avg/vec_sum elsewhere aggregate factor columns for one stock.
    """
    functions = {"rank": _rank, "zscore": _zscore,
                 "mean": lambda series: pd.Series(series.mean(), index=series.index)}
    result = values.astype(float).copy()
    for operator in pipeline:
        if operator == "raw":
            continue
        if operator.startswith("group_"):
            function = functions[operator.removeprefix("group_")]
            result = result.groupby(groups, sort=False).transform(function)
        else:
            result = functions[operator](result)
    return result


def combine(scores, operator):
    """Aggregate across factors; never silently average fewer than all factors."""
    complete = scores.notna().all(axis=1)
    if operator == "vec_avg":
        result = scores.mean(axis=1)
    elif operator == "vec_sum":
        result = scores.sum(axis=1, min_count=scores.shape[1])
    else:
        raise ValueError("Unknown combine_operator: " + str(operator))
    return result.where(complete).round(12)
