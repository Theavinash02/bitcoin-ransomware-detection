import numpy as np
import pandas as pd

from ransomdet import data, features
from ransomdet.audit import BH_DAY_OFFSET, MIN_EDGE_SATOSHI, family_key


def payments(cfg: dict) -> pd.DataFrame:
    """Return one row per Ransomwhere bitcoin payment, deduplicated and filtered like BitcoinHeist."""
    records = data.load_ransomwhere(cfg)
    tx = pd.DataFrame(
        [
            {"address": r["address"], "family": r.get("family") or "Unlabeled", **t}
            for r in records
            if r.get("blockchain", "bitcoin") == "bitcoin"
            for t in (r.get("transactions") or [])
        ]
    )
    tx = tx.drop_duplicates(["address", "hash"])
    tx = tx[tx["amount"] >= MIN_EDGE_SATOSHI]
    ts = pd.to_datetime(tx["time"], unit="s") + BH_DAY_OFFSET
    tx["date"] = ts.dt.normalize()
    return tx


def address_days(cfg: dict, exclude: set[str]) -> pd.DataFrame:
    """Rebuild income and neighbors per address and day, then the portable feature groups."""
    tx = payments(cfg)
    tx = tx[~tx["address"].isin(exclude)]
    days = (
        tx.groupby(["address", "date"])
        .agg(income=("amount", "sum"), neighbors=("hash", "size"), family=("family", "first"))
        .reset_index()
    )
    # Graph features cannot be rebuilt; build_features needs the columns to exist.
    for col in ("count", "length", "weight", "looped"):
        days[col] = np.nan
    prices = pd.read_parquet(cfg["paths"]["processed"] / data.PRICE_PARQUET)
    return features.build_features(days, prices)


def family_slices(days: pd.DataFrame, bh_labels: list[str]) -> dict[str, pd.Series]:
    """Return address-day masks for the outside-test slices."""
    bh_keys = {family_key(str(f)) for f in bh_labels}
    known = days["family"].map(family_key).isin(bh_keys)
    unlabeled = days["family"] == "Unlabeled"
    return {
        "all addresses": pd.Series(True, index=days.index),
        "families also in BitcoinHeist": known & ~unlabeled,
        "families new to BitcoinHeist": ~known & ~unlabeled,
        "  of which NotNevada": days["family"] == "NotNevada",
        "  new families without NotNevada": ~known & ~unlabeled & (days["family"] != "NotNevada"),
        "Unlabeled": unlabeled,
        "payments after 2018": days["date"].dt.year > 2018,
    }
