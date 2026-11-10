import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class Calibrator:
    """Maps raw detector scores to probabilities. Stored as plain numbers, not a pickle."""

    def __init__(self, method: str, params: dict | None = None):
        self.method = method
        self.params = params or {}

    def fit(self, score: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(score, y)
            self.params = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
        else:
            lr = LogisticRegression(C=1e6).fit(_logit(score)[:, None], y)
            self.params = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype="float64")
        if self.method == "isotonic":
            return np.interp(score, self.params["x"], self.params["y"])
        z = self.params["a"] * _logit(score) + self.params["b"]
        return 1 / (1 + np.exp(-z))

    def save(self, path) -> None:
        path.write_text(json.dumps({"method": self.method, "params": self.params}))

    @classmethod
    def load(cls, path) -> "Calibrator":
        d = json.loads(path.read_text())
        return cls(d["method"], d["params"])


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def choose_calibrator(score: np.ndarray, y: np.ndarray, day: np.ndarray, folds: int = 5) -> dict:
    """Compare isotonic and sigmoid by out-of-fold Brier score over blocks of whole days."""
    days = np.sort(np.unique(day))
    blocks = np.array_split(days, folds)
    oof = {"isotonic": np.empty(len(y)), "sigmoid": np.empty(len(y))}
    for block in blocks:
        held = np.isin(day, block)
        for method in oof:
            cal = Calibrator(method).fit(score[~held], y[~held])
            oof[method][held] = cal.predict(score[held])
    brier = {m: float(brier_score_loss(y, p)) for m, p in oof.items()}
    # Isotonic is the default; sigmoid wins only if isotonic overfits out of fold.
    method = "isotonic" if brier["isotonic"] <= brier["sigmoid"] else "sigmoid"
    return {"method": method, "oof_brier": brier}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Return mean predicted vs observed rate per probability bin (bins by predicted quantile)."""
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return df.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), rows=("y", "size"))
