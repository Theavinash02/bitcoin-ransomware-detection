"""Small helpers shared by the stage reports."""

import numpy as np


def fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{digits}f}" if isinstance(v, float) else f"{v:,}"


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def dev_suffix(dev: bool) -> str:
    return "_dev" if dev else ""
