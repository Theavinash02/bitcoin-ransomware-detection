import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score


def threshold_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float) -> float:
    """Return the lowest threshold whose false-positive rate on (y, score) is at most fpr.

    A row is flagged when its score is strictly above the threshold.
    """
    neg = np.sort(score[y == 0])[::-1]
    n_allowed = int(np.floor(fpr * len(neg)))
    return float(neg[n_allowed]) if n_allowed < len(neg) else float(-np.inf)


def rates_at(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    flagged = score > threshold
    pos, neg = y == 1, y == 0
    return {
        "recall": float(flagged[pos].mean()) if pos.any() else float("nan"),
        "fpr": float(flagged[neg].mean()) if neg.any() else float("nan"),
        "flagged": int(flagged.sum()),
    }


def topk_per_day(y: np.ndarray, score: np.ndarray, day: np.ndarray, k: int) -> dict:
    """Return pooled precision of the top k scores on each day, and the perfect-ranker ceiling."""
    df = pd.DataFrame({"y": y, "s": score, "d": day})
    df["r"] = df.groupby("d")["s"].rank(method="first", ascending=False)
    hits = int(df.loc[df["r"] <= k, "y"].sum())
    days = df["d"].nunique()
    pos = df.groupby("d")["y"].sum()
    return {
        "precision_at_k": hits / (k * days),
        "ceiling": float(np.minimum(pos, k).sum() / (k * days)),
        "hits": hits,
        "days": int(days),
    }


def detector_metrics(y: np.ndarray, score: np.ndarray, day: np.ndarray, threshold: float, k: int) -> dict:
    """Return every detector metric for one model on one evaluation set."""
    top = topk_per_day(y, score, day, k)
    at = rates_at(y, score, threshold)
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, score)),
        "recall_at_fpr": at["recall"],
        "fpr_at_threshold": at["fpr"],
        "flagged": at["flagged"],
        "threshold": threshold,
        "precision_at_k": top["precision_at_k"],
        "precision_at_k_ceiling": top["ceiling"],
        "brier": float(brier_score_loss(y, np.clip(score, 0, 1))),
        "base_rate": float(y.mean()),
    }


def label_metrics(y: np.ndarray, score: np.ndarray, cutoff: float = 0.5) -> dict:
    """Return accuracy and F1 at a fixed cutoff, for the old-protocol comparison only."""
    pred = (score >= cutoff).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "f1_weighted": float(f1_score(y, pred, average="weighted")),
        "f1_macro": float(f1_score(y, pred, average="macro")),
    }
