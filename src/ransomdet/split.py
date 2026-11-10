import numpy as np
import pandas as pd

from ransomdet import data

TRAIN, VAL, TEST, DROPPED = 0, 1, 2, -1
NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def assign_split(df: pd.DataFrame, val_start: str, test_start: str) -> pd.Series:
    """Return each row's split code (0 train, 1 val, 2 test, -1 dropped).

    An address with rows in several splits keeps only the rows in its earliest split.
    """
    val_start, test_start = pd.Timestamp(val_start), pd.Timestamp(test_start)
    rank = np.where(df["date"] < val_start, TRAIN, np.where(df["date"] < test_start, VAL, TEST))
    rank = pd.Series(rank.astype("int8"), index=df.index)
    first = rank.groupby(df["address"], observed=True).transform("min")
    return rank.where(rank == first, DROPPED).astype("int8")


def check_split(df: pd.DataFrame, split: pd.Series) -> None:
    """Raise if splits overlap in time or share an address."""
    dates = {c: df.loc[split == c, "date"] for c in NAMES}
    for a, b in ((TRAIN, VAL), (VAL, TEST)):
        if len(dates[a]) and len(dates[b]) and dates[a].max() >= dates[b].min():
            raise AssertionError(f"{NAMES[a]} is not entirely before {NAMES[b]}")
    seen: set = set()
    for c in NAMES:
        addrs = set(df.loc[split == c, "address"].astype(str))
        if seen & addrs:
            raise AssertionError(f"{len(seen & addrs)} addresses of {NAMES[c]} appear in an earlier split")
        seen |= addrs


def split_path(cfg: dict, dev: bool):
    return cfg["paths"]["processed"] / ("splits_dev.parquet" if dev else "splits.parquet")


def build_splits(cfg: dict, dev: bool = False) -> pd.Series:
    """Write the split code for every row of the (dev) BitcoinHeist table, in row order."""
    df = data.load_bitcoinheist(cfg, dev=dev, columns=["address", "date"])
    split = assign_split(df, cfg["split"]["val_start"], cfg["split"]["test_start"])
    check_split(df, split)
    split.to_frame("split").to_parquet(split_path(cfg, dev), index=False)
    return split


def load_splits(cfg: dict, dev: bool = False) -> pd.Series:
    return pd.read_parquet(split_path(cfg, dev))["split"]


def thin_white(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Return df with only a random `fraction` of its white rows; ransomware rows are all kept."""
    white = df["is_ransom"] == 0
    rng = np.random.default_rng(seed)
    keep = ~white | (rng.random(len(df)) < fraction)
    return df[keep]
