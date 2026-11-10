import numpy as np
import pandas as pd

SATOSHI = 1e8
USD_STEPS = (50, 100, 500)
BTC_STEPS = (0.1, 0.5, 1.0)
PCT_FEATURES = ("income_usd", "count", "neighbors", "length")
N_QUANTILES = 101

BASE = ["length", "weight", "count", "looped", "neighbors", "income"]
GROUPS = {
    "A": ["log_income_btc", "log_count", "log_neighbors", "log_length"],
    "B": ["avg_payment_btc", "count_per_neighbor"],
    "C": ["income_usd", "log_income_usd"],
    "D": [f"round_usd_{s}" for s in USD_STEPS] + [f"round_btc_{s:g}" for s in BTC_STEPS] + ["round_min"],
    "E": [f"pct_{f}" for f in PCT_FEATURES],
}
# What Ransomwhere's payment list lets us rebuild (see the data audit).
PORTABLE = [
    "income",
    "neighbors",
    "log_income_btc",
    "log_neighbors",
    "avg_payment_btc",
    *GROUPS["C"],
    *GROUPS["D"],
]
LABEL_COLUMNS = {"label", "is_ransom", "address"}


def round_distance(x: np.ndarray, step: float) -> np.ndarray:
    """Return the distance from x to the nearest multiple of step, as a share of step (0 to 0.5)."""
    x = np.asarray(x, dtype="float64")
    return np.abs(x - np.round(x / step) * step) / step


def add_price(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return df with close_usd for each row's date."""
    p = prices.set_index("date")["close_usd"]
    out = df.copy()
    out["close_usd"] = p.reindex(pd.DatetimeIndex(out["date"])).to_numpy()
    if out["close_usd"].isna().any():
        raise ValueError("some dates have no BTC/USD price")
    return out


def day_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-day quantile tables (one row per date and quantile) for the Group E features."""
    qs = np.linspace(0, 1, N_QUANTILES)
    rows = []
    for date, g in df.groupby("date", sort=True):
        block = {"date": np.repeat(date, N_QUANTILES), "q": qs}
        for f in PCT_FEATURES:
            block[f] = np.quantile(g[f].to_numpy(dtype="float64"), qs)
        rows.append(pd.DataFrame(block))
    return pd.concat(rows, ignore_index=True)


def percentile_from_table(values: np.ndarray, table_values: np.ndarray) -> np.ndarray:
    """Return where each value falls in one day's quantile table, from 0 to 1."""
    n = len(table_values) - 1
    left = np.searchsorted(table_values, values, side="left")
    right = np.searchsorted(table_values, values, side="right")
    # A value equal to several quantiles (count = 1 on most rows) gets the middle of that run.
    tied = (left + right - 1) / 2 / n
    between = np.interp(values, table_values, np.linspace(0, 1, n + 1), left=0.0, right=1.0)
    return np.where(right > left, tied, between)


def add_percentiles(df: pd.DataFrame, quantiles: pd.DataFrame) -> pd.DataFrame:
    """Return df with Group E columns, looked up in the per-day quantile tables."""
    out = df.copy()
    tables = {d: g for d, g in quantiles.groupby("date")}
    dates = out["date"].to_numpy()
    codes, uniq = pd.factorize(dates)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
    for f in PCT_FEATURES:
        vals = out[f].to_numpy(dtype="float64")
        res = np.empty(len(out))
        for i, d in enumerate(uniq):
            rows = order[bounds[i] : bounds[i + 1]]
            table = tables.get(pd.Timestamp(d))
            if table is None:
                raise ValueError(f"no quantile table for {pd.Timestamp(d).date()}")
            res[rows] = percentile_from_table(vals[rows], table[f].to_numpy())
        out[f"pct_{f}"] = res
    return out


def build_features(
    df: pd.DataFrame, prices: pd.DataFrame, quantiles: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return df with every feature group added. Group E needs the per-day quantile tables.

    Works the same on the full table and on a single row typed into the demo.
    """
    out = add_price(df, prices)
    income_btc = out["income"].to_numpy(dtype="float64") / SATOSHI
    out["income_btc"] = income_btc
    out["log_income_btc"] = np.log1p(income_btc)
    for f in ("count", "neighbors", "length"):
        out[f"log_{f}"] = np.log1p(out[f].to_numpy(dtype="float64"))
    nb = np.maximum(out["neighbors"].to_numpy(dtype="float64"), 1)
    out["avg_payment_btc"] = income_btc / nb
    out["count_per_neighbor"] = out["count"].to_numpy(dtype="float64") / nb
    out["income_usd"] = income_btc * out["close_usd"].to_numpy()
    out["log_income_usd"] = np.log1p(out["income_usd"])
    round_cols = []
    for s in USD_STEPS:
        out[f"round_usd_{s}"] = round_distance(out["income_usd"], s)
        round_cols.append(f"round_usd_{s}")
    for s in BTC_STEPS:
        out[f"round_btc_{s:g}"] = round_distance(income_btc, s)
        round_cols.append(f"round_btc_{s:g}")
    out["round_min"] = out[round_cols].min(axis=1)
    if quantiles is not None:
        out = add_percentiles(out, quantiles)
    return out


FEATURES_PARQUET = "features.parquet"
FEATURES_DEV_PARQUET = "features_dev.parquet"
QUANTILES_PARQUET = "day_quantiles.parquet"


def build_feature_table(cfg: dict, dev: bool = False) -> pd.DataFrame:
    """Write every feature for every BitcoinHeist row, plus the per-day quantile tables."""
    from ransomdet import data

    proc = cfg["paths"]["processed"]
    prices = pd.read_parquet(proc / data.PRICE_PARQUET)
    if dev and (proc / QUANTILES_PARQUET).exists():
        # Percentiles in dev mode still come from the full data's daily tables.
        quantiles = pd.read_parquet(proc / QUANTILES_PARQUET)
    else:
        base = build_features(data.load_bitcoinheist(cfg), prices)
        quantiles = day_quantiles(base)
        quantiles.to_parquet(proc / QUANTILES_PARQUET, index=False)
        del base
    df = build_features(data.load_bitcoinheist(cfg, dev=dev), prices, quantiles)
    keep = ["address", "date", "label", "is_ransom", "year", "day", "close_usd", *BASE]
    keep += sorted({c for g in GROUPS.values() for c in g} | {"income_btc"})
    df = df[keep]
    for c in keep[7 + len(BASE) :]:
        df[c] = df[c].astype("float32")
    df.to_parquet(proc / (FEATURES_DEV_PARQUET if dev else FEATURES_PARQUET), index=False, compression="zstd")
    return df


def load_feature_table(cfg: dict, dev: bool = False, columns: list | None = None) -> pd.DataFrame:
    name = FEATURES_DEV_PARQUET if dev else FEATURES_PARQUET
    return pd.read_parquet(cfg["paths"]["processed"] / name, columns=columns)


def feature_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return the model input columns, refusing anything that carries the label."""
    leaked = LABEL_COLUMNS & set(columns)
    if leaked:
        raise ValueError(f"label columns in the feature list: {sorted(leaked)}")
    return df[columns].astype("float32")
