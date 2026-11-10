"""Rebuild tests/fixtures/ from the full data: `python scripts/make_fixture.py`."""

import pandas as pd

from ransomdet import data
from ransomdet.config import ROOT, load_config

OUT = ROOT / "tests" / "fixtures"


def main() -> None:
    cfg = load_config()
    df = data.load_bitcoinheist(cfg)
    df["address"] = df["address"].astype(str)
    df["label"] = df["label"].astype(str)
    parts = []
    # Enough ransomware in each split for the smoke test to have positives everywhere.
    for (start, end), n_white, n_ransom in (
        (("2013-01-01", "2016-07-01"), 2400, 900),
        (("2016-07-01", "2017-01-01"), 600, 300),
        (("2017-01-01", "2019-01-01"), 600, 200),
    ):
        win = df[(df["date"] >= start) & (df["date"] < end)]
        parts.append(win[win["is_ransom"] == 0].sample(n=n_white, random_state=0))
        parts.append(win[win["is_ransom"] == 1].sample(n=n_ransom, random_state=0))
    fx = pd.concat(parts).sort_values("date")
    cols = ["address", "year", "day", "length", "weight", "count", "looped", "neighbors", "income", "label"]
    OUT.mkdir(parents=True, exist_ok=True)
    fx[cols].to_csv(OUT / "bitcoinheist_5k.csv", index=False)
    prices = pd.read_parquet(cfg["paths"]["processed"] / data.PRICE_PARQUET)
    prices = prices[prices["date"].isin(set(fx["date"]))]
    prices.to_csv(OUT / "btc_usd_5k.csv", index=False)
    print(f"wrote {len(fx)} rows and {len(prices)} prices to {OUT}")


if __name__ == "__main__":
    main()
