import io
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BH_CSV = "BitcoinHeistData.csv"
RW_JSON = "ransomwhere.json"
RW_META = "ransomwhere_meta.json"
PRICE_CSV = "btc_usd.csv"

BH_PARQUET = "bitcoinheist.parquet"
BH_DEV_PARQUET = "bitcoinheist_dev.parquet"
PRICE_PARQUET = "btc_usd.parquet"

# income is kept as int64 satoshi: float32 has 7 significant digits, which would
# destroy the exact amounts the round-amount features depend on.
BH_DTYPES = {
    "address": "string",
    "year": "int16",
    "day": "int16",
    "length": "int16",
    "weight": "float32",
    "count": "int32",
    "looped": "int32",
    "neighbors": "int32",
    "income": "float64",
    "label": "string",
}

USER_AGENT = "ransomdet-data-download"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def download_bitcoinheist(cfg: dict) -> Path:
    raw = cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / BH_CSV
    if out.exists():
        return out
    # Older checkouts kept the CSV at the repo root.
    legacy = raw.parents[1] / BH_CSV
    if legacy.exists():
        shutil.move(legacy, out)
        return out
    print("downloading BitcoinHeist from UCI")
    with zipfile.ZipFile(io.BytesIO(_fetch(cfg["sources"]["bitcoinheist_url"]))) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(name) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return out


def download_ransomwhere(cfg: dict) -> Path:
    raw = cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / RW_JSON
    if out.exists():
        return out
    record_id = cfg["sources"]["ransomwhere_record"]
    print("resolving latest Ransomwhere version on Zenodo")
    rec = json.loads(_fetch(f"https://zenodo.org/api/records/{record_id}/versions/latest"))
    f = next(f for f in rec["files"] if f["key"].endswith(".json"))
    out.write_bytes(_fetch(f["links"]["self"]))
    meta = {
        "record_id": rec["id"],
        "concept_record_id": rec.get("conceptrecid"),
        "version": rec["metadata"].get("version"),
        "publication_date": rec["metadata"].get("publication_date"),
        "license": rec["metadata"].get("license", {}).get("id"),
        "doi": rec.get("doi"),
        "file": f["key"],
        "checksum": f.get("checksum"),
    }
    (raw / RW_META).write_text(json.dumps(meta, indent=2))
    return out


def download_prices(cfg: dict) -> Path:
    raw = cfg["paths"]["raw"]
    raw.mkdir(parents=True, exist_ok=True)
    out = raw / PRICE_CSV
    if out.exists():
        return out
    print("downloading BTC/USD daily prices from Coin Metrics")
    out.write_bytes(_fetch(cfg["sources"]["price_url"]))
    return out


def download_all(cfg: dict) -> dict:
    return {
        "bitcoinheist": download_bitcoinheist(cfg),
        "ransomwhere": download_ransomwhere(cfg),
        "prices": download_prices(cfg),
    }


def add_date(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a `date` column built from year and day-of-year."""
    df["date"] = pd.to_datetime(df["year"].astype(str), format="%Y") + pd.to_timedelta(
        df["day"].astype("int32") - 1, unit="D"
    )
    return df


def convert_bitcoinheist(cfg: dict) -> Path:
    """Write the full BitcoinHeist table to Parquet with compact types."""
    src = cfg["paths"]["raw"] / BH_CSV
    out = cfg["paths"]["processed"] / BH_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src, dtype=BH_DTYPES, engine="pyarrow")
    # Exported as floats like 1e+08, but every value is a whole number of satoshi.
    if not np.all(np.mod(df["income"].to_numpy(), 1) == 0):
        raise ValueError("income has fractional satoshi values")
    df["income"] = df["income"].astype("int64")
    df["address"] = df["address"].astype("category")
    df["label"] = df["label"].astype("category")
    df["is_ransom"] = (df["label"] != "white").astype("int8")
    df = add_date(df)
    df.to_parquet(out, index=False, compression="zstd")
    return out


def make_dev_sample(cfg: dict) -> Path:
    """Write a small sample that keeps every ransomware row, for fast --dev runs."""
    df = pd.read_parquet(cfg["paths"]["processed"] / BH_PARQUET)
    ransom = df[df["is_ransom"] == 1]
    n_white = max(cfg["dev"]["rows"] - len(ransom), 0)
    white = df[df["is_ransom"] == 0].sample(n=n_white, random_state=cfg["seed"])
    dev = pd.concat([ransom, white]).sort_values(["year", "day"]).reset_index(drop=True)
    for col in ("address", "label"):
        dev[col] = dev[col].cat.remove_unused_categories()
    out = cfg["paths"]["processed"] / BH_DEV_PARQUET
    dev.to_parquet(out, index=False, compression="zstd")
    return out


def convert_prices(cfg: dict) -> Path:
    src = cfg["paths"]["raw"] / PRICE_CSV
    out = cfg["paths"]["processed"] / PRICE_PARQUET
    p = pd.read_csv(src)
    p = pd.DataFrame(
        {
            "date": pd.to_datetime(p["time"]).dt.tz_localize(None).dt.normalize(),
            "close_usd": p["PriceUSD"].astype("float64"),
        }
    )
    p.to_parquet(out, index=False)
    return out


def load_bitcoinheist(cfg: dict, dev: bool = False, columns: list | None = None) -> pd.DataFrame:
    name = BH_DEV_PARQUET if dev else BH_PARQUET
    return pd.read_parquet(cfg["paths"]["processed"] / name, columns=columns)


def load_ransomwhere(cfg: dict) -> list[dict]:
    data = json.loads((cfg["paths"]["raw"] / RW_JSON).read_text(encoding="utf-8"))
    return data["result"] if isinstance(data, dict) and "result" in data else data
