import json
import re
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from ransomdet import data
from ransomdet.split import assign_split

MIN_EDGE_SATOSHI = 30_000_000  # BitcoinHeist dropped edges under 0.3 BTC
BH_DAY_OFFSET = timedelta(hours=-6)  # BitcoinHeist days are cut in UTC-6


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Return df as a GitHub Markdown table."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float | np.floating):
                cells.append(f"{v:,.2f}" if not float(v).is_integer() else f"{int(v):,}")
            elif isinstance(v, int | np.integer):
                cells.append(str(int(v)) if 1900 < v < 2100 else f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def family_key(name: str) -> str:
    """Return a normalized family name so the two datasets can be matched by name."""
    name = re.sub(r"^(montreal|princeton|padua)", "", name)
    name = name.split("(")[0]
    return re.sub(r"[^a-z0-9]", "", name.lower())


def mb(n_bytes: float) -> float:
    return round(n_bytes / 2**20, 1)


def size_section(cfg: dict, df: pd.DataFrame) -> tuple[list[str], dict]:
    raw, proc = cfg["paths"]["raw"], cfg["paths"]["processed"]
    stats = {
        "rows": len(df),
        "csv_mb": mb((raw / data.BH_CSV).stat().st_size),
        "parquet_mb": mb((proc / data.BH_PARQUET).stat().st_size),
        "dev_parquet_mb": mb((proc / data.BH_DEV_PARQUET).stat().st_size),
        "loaded_mb": mb(df.memory_usage(deep=True).sum()),
        "ransomwhere_json_mb": mb((raw / data.RW_JSON).stat().st_size),
        "price_csv_mb": mb((raw / data.PRICE_CSV).stat().st_size),
        "income_min_satoshi": int(df["income"].min()),
        "rows_below_0_3_btc": int((df["income"] < MIN_EDGE_SATOSHI).sum()),
    }
    dtypes = pd.DataFrame({"dtype": df.dtypes.astype(str)})
    lines = [
        "## 1. Download and convert",
        "",
        f"BitcoinHeist has {stats['rows']:,} rows. The CSV is {stats['csv_mb']} MB on disk, "
        f"the Parquet file is {stats['parquet_mb']} MB, and the table takes "
        f"{stats['loaded_mb']} MB in memory with the types below.",
        "",
        md_table(pd.DataFrame([{k: v for k, v in stats.items() if k.endswith("_mb")}]), False),
        "",
        md_table(dtypes.rename_axis("column")),
        "",
        "`income` is stored as int64 satoshi rather than float32. Every value is a whole "
        "number, and float32 keeps only about 7 significant digits, which would blur the exact "
        "amounts that the round-amount features need.",
        "",
        f"Smallest income is {stats['income_min_satoshi']:,} satoshi "
        f"({stats['income_min_satoshi'] / 1e8:g} BTC); {stats['rows_below_0_3_btc']:,} rows "
        "are below 0.3 BTC. This matches the paper's filter, which drops transfers under "
        "0.3 BTC before features are computed.",
        "",
    ]
    return lines, stats


def address_section(df: pd.DataFrame) -> tuple[list[str], dict]:
    g = df.groupby("address", observed=True)
    per_addr = g.agg(
        rows=("year", "size"),
        years=("year", "nunique"),
        labels=("label", "nunique"),
        ransom=("is_ransom", "max"),
    )
    buckets = pd.cut(
        per_addr["rows"],
        [0, 1, 2, 5, 10, 100, np.inf],
        labels=["1", "2", "3-5", "6-10", "11-100", "100+"],
    )
    dist = (
        pd.crosstab(buckets, per_addr["ransom"].map({0: "white", 1: "ransomware"}))
        .reindex(columns=["white", "ransomware"], fill_value=0)
        .rename_axis("rows per address")
    )
    multi_label = per_addr[per_addr["labels"] > 1]
    sub = df[df["address"].isin(multi_label.index[:10])]
    multi_examples = (
        sub.assign(label=sub["label"].astype(str))
        .groupby("address", observed=True)["label"]
        .agg(lambda s: ", ".join(sorted(set(s))))
    )
    stats = {
        "unique_addresses": len(per_addr),
        "rows": len(df),
        "max_rows_per_address": int(per_addr["rows"].max()),
        "addresses_multi_year": int((per_addr["years"] > 1).sum()),
        "ransom_addresses_multi_year": int(((per_addr["years"] > 1) & (per_addr["ransom"] == 1)).sum()),
        "addresses_multi_label": len(multi_label),
        "ransom_addresses": int(per_addr["ransom"].sum()),
    }
    lines = [
        "## 3. Repeated addresses",
        "",
        f"{stats['rows']:,} rows come from {stats['unique_addresses']:,} unique addresses. "
        f"{stats['addresses_multi_year']:,} addresses appear in more than one year "
        f"({stats['ransom_addresses_multi_year']:,} of them ransomware), and the busiest address "
        f"has {stats['max_rows_per_address']:,} rows. A random row split puts the same address "
        "on both sides, which is one reason the old 80/20 numbers look better than they are.",
        "",
        md_table(dist),
        "",
        f"Addresses with more than one label: {stats['addresses_multi_label']:,}.",
        "",
    ]
    if len(multi_label):
        lines += [md_table(multi_examples.to_frame("labels")), ""]
    return lines, stats


def year_section(df: pd.DataFrame, cfg: dict) -> tuple[list[str], dict]:
    ransom = df[df["is_ransom"] == 1]
    by_year = pd.DataFrame(
        {
            "white rows": df[df["is_ransom"] == 0].groupby("year").size(),
            "ransom rows": ransom.groupby("year").size(),
            "ransom addresses": ransom.groupby("year")["address"].nunique(),
            "ransom families": ransom.groupby("year")["label"].nunique(),
            "days": df.groupby("year")["day"].nunique(),
            "days with ransom": ransom.groupby("year")["day"].nunique(),
        }
    ).fillna(0)
    by_year["white per day"] = (by_year["white rows"] / by_year["days"]).round(1)
    by_year["ransom per day"] = (by_year["ransom rows"] / by_year["days"]).round(2)
    by_year = by_year.astype({c: "int64" for c in by_year.columns if "per day" not in c})

    fam = ransom.assign(label=ransom["label"].astype(str)).pivot_table(
        index="label", columns="year", values="address", aggfunc="nunique", fill_value=0
    )
    fam["total addresses"] = (
        ransom.assign(label=ransom["label"].astype(str)).groupby("label")["address"].nunique()
    )
    fam["total rows"] = ransom.assign(label=ransom["label"].astype(str)).groupby("label").size()
    fam = fam.sort_values("total addresses", ascending=False)
    fam.columns = [str(c) for c in fam.columns]

    split_lines, split_stats = split_proposal(df, cfg)
    lines = [
        "## 4. Ransomware by year and day",
        "",
        md_table(by_year.rename_axis("year")),
        "",
        "Unique ransomware addresses per family and year (the last two columns are totals):",
        "",
        md_table(fam.rename_axis("family")),
        "",
        *split_lines,
    ]
    return lines, {"by_year": by_year.reset_index().to_dict("records"), **split_stats}


# Each candidate is (validation start, test start); everything before the first is training.
SPLIT_CANDIDATES = {
    "A: train 2011-15, val 2016, test 2017-18": ("2016-01-01", "2017-01-01"),
    "B: train 2011-14, val 2015, test 2016-18": ("2015-01-01", "2016-01-01"),
    "C: train to Jun 2016, val Jul-Dec 2016, test 2017-18": ("2016-07-01", "2017-01-01"),
}


def split_proposal(df: pd.DataFrame, cfg: dict) -> tuple[list[str], dict]:
    rows = []
    detail = {}
    ransom_mask = df["is_ransom"] == 1
    for name, (val_start, test_start) in SPLIT_CANDIDATES.items():
        rank = assign_split(df, val_start, test_start)
        train_fams = set(df.loc[(rank == 0) & ransom_mask, "label"].astype(str))
        for i, part in enumerate(("train", "val", "test")):
            m = rank == i
            r = df.loc[m & ransom_mask]
            fams = set(r["label"].astype(str))
            r_new = r[~r["label"].astype(str).isin(train_fams)]
            rows.append(
                {
                    "candidate": name,
                    "split": part,
                    "rows": int(m.sum()),
                    "ransom rows": len(r),
                    "ransom addresses": r["address"].nunique(),
                    "families": len(fams),
                    "ransom rows from families absent in train": len(r_new) if i else 0,
                }
            )
        detail[name] = {"rows_dropped_for_overlap": int((rank == -1).sum())}
    table = pd.DataFrame(rows)
    dropped = pd.DataFrame(
        [
            {
                "candidate": k,
                "rows dropped (address seen in earlier split)": v["rows_dropped_for_overlap"],
            }
            for k, v in detail.items()
        ]
    )

    ransom = df[ransom_mask]
    span = (
        ransom.assign(label=ransom["label"].astype(str))
        .groupby("label")["date"]
        .agg(["min", "max"])
        .rename(columns={"min": "first day", "max": "last day"})
        .sort_values("first day")
    )
    span = span.apply(lambda c: c.dt.date.astype(str))

    k_lines, k_stats = k_proposal(df, cfg)
    lines = [
        "### Split proposal",
        "",
        "Each candidate applies the baseline split rule: an address that shows up in several splits "
        "is kept only in its earliest split. The last column counts ransomware rows whose "
        "family never appears in that candidate's training split.",
        "",
        md_table(table, False),
        "",
        md_table(dropped, False),
        "",
        "First and last day each family appears:",
        "",
        md_table(span.rename_axis("family")),
        "",
        *k_lines,
    ]
    return lines, {"split_candidates": table.to_dict("records"), "split_dropped": detail, **k_stats}


def topk_ceiling(pos_per_day: pd.Series, k: int) -> float:
    """Return the best pooled precision at K: hits over K alerts per day, if ranking were perfect."""
    return float(np.minimum(pos_per_day, k).sum() / (k * len(pos_per_day)))


def k_proposal(df: pd.DataFrame, cfg: dict) -> tuple[list[str], dict]:
    rows = []
    stats: dict = {"by_period": {}}
    # Windows of the proposed split C.
    val_start, test_start = SPLIT_CANDIDATES["C: train to Jun 2016, val Jul-Dec 2016, test 2017-18"]
    periods = {
        "validation (Jul-Dec 2016)": (df["date"] >= val_start) & (df["date"] < test_start),
        "test (2017-18)": df["date"] >= test_start,
    }
    for label, mask in periods.items():
        per_day = df[mask].groupby("date")["is_ransom"].agg(["sum", "size"])
        pos = per_day["sum"]
        row = {
            "period": label,
            "days": len(per_day),
            "rows per day": round(float(per_day["size"].mean()), 1),
            "ransom per day, mean": round(float(pos.mean()), 2),
            "median": float(pos.median()),
            "days with none": int((pos == 0).sum()),
        }
        for k in (5, 10, 20, 50):
            row[f"ceiling K={k}"] = round(topk_ceiling(pos, k), 3)
        rows.append(row)
        stats["by_period"][label] = row
    lines = [
        "### Top-K budget",
        "",
        "Ransomware rows per day, and the best pooled precision at K a perfect ranker could "
        "reach (hits divided by K alerts on every day). Days with fewer than K ransomware rows "
        "cap it below 1.",
        "",
        md_table(pd.DataFrame(rows), False),
        "",
    ]
    return lines, stats


def utc6_date(ts: int) -> datetime:
    return (datetime.fromtimestamp(ts, UTC) + BH_DAY_OFFSET).date()


def ransomwhere_frames(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    addrs = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k != "transactions"} | {"n_tx": len(r.get("transactions") or [])}
            for r in records
        ]
    )
    tx = pd.DataFrame(
        [
            {
                "address": r["address"],
                "family": r.get("family"),
                "blockchain": r.get("blockchain"),
                **t,
            }
            for r in records
            for t in (r.get("transactions") or [])
        ]
    )
    return addrs, tx


def ransomwhere_section(records: list[dict]) -> tuple[list[str], dict, pd.DataFrame, pd.DataFrame]:
    addrs, tx = ransomwhere_frames(records)
    addr_fields = sorted({k for r in records for k in r})
    tx_fields = sorted({k for r in records for t in (r.get("transactions") or []) for k in t})
    presence = pd.DataFrame(
        {
            "non-null": [int(tx[c].notna().sum()) for c in tx_fields],
            "share": [round(float(tx[c].notna().mean()), 4) for c in tx_fields],
        },
        index=tx_fields,
    )
    tx["dt"] = pd.to_datetime(tx["time"], unit="s", utc=True)
    btc = tx[tx["blockchain"] == "bitcoin"]
    by_chain = addrs.groupby("blockchain").size().to_frame("addresses")
    by_chain["transactions"] = tx.groupby("blockchain").size()
    by_chain = by_chain.fillna(0).astype("int64")
    by_year = btc.groupby(btc["dt"].dt.year).agg(
        transactions=("hash", "size"), addresses=("address", "nunique")
    )
    stats = {
        "addresses": len(addrs),
        "addresses_with_no_tx": int((addrs["n_tx"] == 0).sum()),
        "transactions": len(tx),
        "btc_transactions": len(btc),
        "families": int(addrs["family"].nunique()),
        "first_tx": str(tx["dt"].min().date()),
        "last_tx": str(tx["dt"].max().date()),
        "tx_missing_amount": int(tx["amount"].isna().sum()),
        "tx_missing_time": int(tx["time"].isna().sum()),
        "tx_nonpositive_amount": int((tx["amount"] <= 0).sum()),
        "btc_tx_below_0_3_btc": int((btc["amount"] < MIN_EDGE_SATOSHI).sum()),
        "duplicate_tx_rows": int(tx.duplicated(["address", "hash"]).sum()),
    }
    lines = [
        "## 5. Ransomwhere schema",
        "",
        f"Ransomwhere {{version}} has {stats['addresses']:,} addresses from {stats['families']} "
        f"families and {stats['transactions']:,} transactions, from {stats['first_tx']} to "
        f"{stats['last_tx']}. {stats['addresses_with_no_tx']:,} addresses list no transactions.",
        "",
        f"Address fields: {', '.join(f'`{f}`' for f in addr_fields)}.",
        "",
        f"Transaction fields: {', '.join(f'`{f}`' for f in tx_fields)}. Each transaction is a "
        "payment into the address (there is no direction field and every amount is positive "
        f"except {stats['tx_nonpositive_amount']:,}).",
        "",
        md_table(presence.rename_axis("transaction field")),
        "",
        f"Missing amount: {stats['tx_missing_amount']:,}. Missing timestamp: "
        f"{stats['tx_missing_time']:,}. Repeated (address, hash) pairs: "
        f"{stats['duplicate_tx_rows']:,}.",
        "",
        md_table(by_chain.rename_axis("blockchain")),
        "",
        "Bitcoin transactions by year (UTC):",
        "",
        md_table(by_year.rename_axis("year")),
        "",
        f"{stats['btc_tx_below_0_3_btc']:,} of {stats['btc_transactions']:,} bitcoin "
        "transactions are below 0.3 BTC and would be dropped to match BitcoinHeist's filter.",
        "",
    ]
    return lines, stats, addrs, tx


def overlap_section(df: pd.DataFrame, addrs: pd.DataFrame, tx: pd.DataFrame) -> tuple[list[str], dict]:
    btc_addrs = addrs[addrs["blockchain"] == "bitcoin"].copy()
    # Addresses never carry two labels (section 3), so the first label is the label.
    rw_set = set(btc_addrs["address"])
    hit = df.loc[df["address"].isin(rw_set), ["address", "label"]]
    bh_labels = (
        hit.assign(address=hit["address"].astype(str), label=hit["label"].astype(str))
        .drop_duplicates("address")
        .set_index("address")["label"]
        .rename("bh_label")
    )
    btc_addrs = btc_addrs.join(bh_labels, on="address")
    in_bh = btc_addrs["bh_label"].notna()
    btc_addrs["family_key"] = btc_addrs["family"].astype(str).map(family_key)
    bh_keys = {family_key(str(f)) for f in df.loc[df["is_ransom"] == 1, "label"].unique()}

    fam = (
        btc_addrs.groupby("family")
        .agg(
            addresses=("address", "size"),
            in_bitcoinheist=("bh_label", lambda s: int(s.notna().sum())),
            key=("family_key", "first"),
        )
        .sort_values("addresses", ascending=False)
    )
    fam["after removing overlap"] = fam["addresses"] - fam["in_bitcoinheist"]
    fam["family in BitcoinHeist"] = fam["key"].isin(bh_keys).map({True: "yes", False: "no"})
    new = fam[fam["family in BitcoinHeist"] == "no"].drop(columns=["key", "family in BitcoinHeist"])
    shared = fam[fam["family in BitcoinHeist"] == "yes"].drop(columns=["family in BitcoinHeist"])

    tx_btc = tx[tx["blockchain"] == "bitcoin"]
    tx_new_addr = tx_btc[~tx_btc["address"].isin(btc_addrs.loc[in_bh, "address"])]
    post2018 = tx_new_addr[pd.to_datetime(tx_new_addr["time"], unit="s").dt.year > 2018]
    overlap_labels = btc_addrs.loc[in_bh, "bh_label"].astype(str).value_counts().to_frame("addresses")

    stats = {
        "btc_addresses": len(btc_addrs),
        "in_bitcoinheist": int(in_bh.sum()),
        "in_bitcoinheist_as_white": int((btc_addrs["bh_label"] == "white").sum()),
        "families_shared": int(len(shared)),
        "families_new": int(len(new)),
        "addresses_new_family_after_overlap": int(new["after removing overlap"].sum()),
        "addresses_shared_family_after_overlap": int(shared["after removing overlap"].sum()),
        "addresses_with_post2018_payments": int(post2018["address"].nunique()),
    }
    lines = [
        "## 6. Overlap with BitcoinHeist",
        "",
        f"{stats['in_bitcoinheist']:,} of {stats['btc_addresses']:,} Ransomwhere bitcoin addresses "
        f"are already in BitcoinHeist, {stats['in_bitcoinheist_as_white']:,} of them labeled "
        "`white` there. Their BitcoinHeist labels:",
        "",
        md_table(overlap_labels.rename_axis("BitcoinHeist label")),
        "",
        "Families are matched by name after lowercasing and dropping the "
        "`montreal`/`princeton`/`padua` prefixes and anything in brackets, so the match is "
        "approximate.",
        "",
        f"Families also in BitcoinHeist ({stats['families_shared']}, "
        f"{stats['addresses_shared_family_after_overlap']:,} addresses after removing overlap):",
        "",
        md_table(shared.drop(columns=["key"]).rename_axis("Ransomwhere family")),
        "",
        f"Families new to BitcoinHeist ({stats['families_new']}, "
        f"{stats['addresses_new_family_after_overlap']:,} addresses after removing overlap):",
        "",
        md_table(new.rename_axis("Ransomwhere family")),
        "",
        f"Non-overlapping addresses with at least one payment after 2018: "
        f"{stats['addresses_with_post2018_payments']:,}.",
        "",
    ]
    return lines, stats


def price_section(cfg: dict, df: pd.DataFrame, tx: pd.DataFrame) -> tuple[list[str], dict]:
    prices = pd.read_parquet(cfg["paths"]["processed"] / data.PRICE_PARQUET)
    have = set(prices["date"].dt.date)
    bh_dates = set(pd.to_datetime(df["date"].unique()).date)
    btc_tx = tx[tx["blockchain"] == "bitcoin"]
    rw_dates = {utc6_date(int(t)) for t in btc_tx["time"].dropna()}
    bh_missing = sorted(bh_dates - have)
    rw_missing = sorted(rw_dates - have)
    stats = {
        "price_first": str(prices["date"].min().date()),
        "price_last": str(prices["date"].max().date()),
        "price_days": len(prices),
        "price_nulls": int(prices["close_usd"].isna().sum()),
        "bh_dates": len(bh_dates),
        "bh_dates_missing_price": len(bh_missing),
        "rw_dates": len(rw_dates),
        "rw_dates_missing_price": len(rw_missing),
        "rw_missing_examples": [str(d) for d in rw_missing[:10]],
        "bh_missing_examples": [str(d) for d in bh_missing[:10]],
    }
    full = pd.date_range(prices["date"].min(), prices["date"].max(), freq="D")
    stats["gaps_inside_series"] = int(len(full) - prices["date"].nunique())
    lines = [
        "## 7. Price coverage",
        "",
        f"The Coin Metrics series runs from {stats['price_first']} to {stats['price_last']} "
        f"({stats['price_days']:,} days, {stats['gaps_inside_series']} missing days inside that "
        f"range, {stats['price_nulls']} null prices).",
        "",
        f"BitcoinHeist dates without a price: {stats['bh_dates_missing_price']} of "
        f"{stats['bh_dates']:,}. Ransomwhere bitcoin payment dates (UTC-6 days) without a price: "
        f"{stats['rw_dates_missing_price']} of {stats['rw_dates']:,}.",
        "",
    ]
    if bh_missing or rw_missing:
        lines += [
            f"Examples: BitcoinHeist {stats['bh_missing_examples']}, Ransomwhere "
            f"{stats['rw_missing_examples']}.",
            "",
        ]
    lines += [
        "Coin Metrics `PriceUSD` is a UTC end-of-day reference rate, while BitcoinHeist days are "
        "cut in UTC-6. A payment late on a UTC-6 day is priced with the close 6 hours before it "
        "happened, which is close enough for dollar amounts rounded to $50 or more.",
        "",
    ]
    return lines, stats


def run_audit(cfg: dict, dev: bool = False) -> dict:
    """Write reports/data_audit.md and .json; return the stats."""
    df = data.load_bitcoinheist(cfg, dev=dev)
    records = data.load_ransomwhere(cfg)
    meta_path = cfg["paths"]["raw"] / data.RW_META
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    out: dict = {"dev": dev, "ransomwhere_meta": meta}
    body: list[str] = []

    lines, out["sizes"] = size_section(cfg, df)
    body += lines
    body += [
        "## 2. Feature definitions",
        "",
        "Checked against the paper and written up in `data/README.md`. The ones that matter "
        "for this project: `income` is the total satoshi paid into the address that day, "
        "`neighbors` is the number of transactions that pay into it that day, and `count`, "
        "`weight`, `length` and `looped` describe the chains of earlier transactions that lead "
        "to the address within the same day, so they need the whole day's transaction graph.",
        "",
    ]
    lines, out["addresses"] = address_section(df)
    body += lines
    lines, out["years"] = year_section(df, cfg)
    body += lines
    lines, out["ransomwhere"], addrs, tx = ransomwhere_section(records)
    body += [line.replace("{version}", f"v{meta.get('version', '?')}") for line in lines]
    lines, out["overlap"] = overlap_section(df, addrs, tx)
    body += lines
    lines, out["prices"] = price_section(cfg, df, tx)
    body += lines
    return out, body


PORTABLE_FEATURES = [
    ("income", "yes", "sum of incoming amounts per address and UTC-6 day, after the 0.3 BTC filter"),
    ("neighbors", "yes", "number of incoming transactions per address and UTC-6 day"),
    ("Group A (log scale)", "partly", "log1p of income_btc and neighbors; count and length are missing"),
    ("Group B (ratios)", "partly", "income / neighbors only; count / neighbors needs count"),
    ("Group C (dollar value)", "yes", "income_btc times the day's close"),
    ("Group D (round amounts)", "yes", "computed from income_btc and income_usd"),
    ("count, weight, length, looped", "no", "need the full day's transaction graph"),
    ("Group E (same-day percentiles)", "no", "needs every other address active that day"),
]


def _cand(result: dict, name_prefix: str, split: str) -> dict:
    rows = result["years"]["split_candidates"]
    return next(r for r in rows if r["candidate"].startswith(name_prefix) and r["split"] == split)


def summary_lines(result: dict, cfg: dict) -> list[str]:
    """Return the checkpoint summary: proposed split, K and the Ransomwhere feature plan."""
    by_year = {r["year"]: r for r in result["years"]["by_year"]}
    last = max(by_year)
    a_train, a_val, a_test = (_cand(result, "A", s) for s in ("train", "val", "test"))
    c_train, c_val, c_test = (_cand(result, "C", s) for s in ("train", "val", "test"))
    kp = result["years"]["by_period"]
    rw, ov = result["ransomwhere"], result["overlap"]

    def share(r: dict) -> str:
        return f"{r['ransom rows from families absent in train'] / max(r['ransom rows'], 1):.0%}"

    lines = [
        "## Checkpoint summary",
        "",
        "### Proposed split",
        "",
        f"I propose candidate C: train on everything before 2016-07-01, validate on July to "
        f"December 2016, and test on 2017-{last}. The starting proposal (A) trains on "
        f"{a_train['families']} families and {a_train['ransom addresses']:,} ransomware "
        f"addresses, and {share(a_val)} of its validation ransomware rows and {share(a_test)} of "
        f"its test rows come from families it never saw in training. That turns the main test "
        f"into an unseen-family test and leaves model selection tuned on families absent from "
        f"training. The generalization tests already measure unseen families properly.",
        "",
        f"Under C, training holds {c_train['families']} families and "
        f"{c_train['ransom addresses']:,} ransomware addresses, validation has "
        f"{c_val['ransom rows']:,} ransomware rows ({share(c_val)} from families absent in "
        f"training) and test has {c_test['ransom rows']:,} ({share(c_test)}).",
        "",
        f"{last} contributes only {by_year[last]['ransom rows']:,} ransomware rows, so the test "
        f"period is in practice 2017. I would keep {last} in test for the false-positive rate "
        f"and say this in the report.",
        "",
        "### Proposed K",
        "",
        f"I propose K = 10 alerts per day, reported as pooled precision (hits / (10 x days)). "
        f"Test days average {kp['test (2017-18)']['ransom per day, mean']} ransomware rows and "
        f"{kp['test (2017-18)']['days with none']} of {kp['test (2017-18)']['days']} test days have none, "
        f"so even a perfect ranker reaches only {kp['test (2017-18)']['ceiling K=10']} at K = 10. "
        f"The report will print that ceiling next to the model's number.",
        "",
        "### What Ransomwhere can support",
        "",
        f"Every one of the {rw['transactions']:,} transactions has an amount and a timestamp, so "
        f"income and neighbors can be rebuilt per address and day. Before that, "
        f"{rw['duplicate_tx_rows']:,} repeated (address, hash) rows need dropping, and "
        f"{rw['btc_tx_below_0_3_btc']:,} transactions under 0.3 BTC are removed to match "
        f"BitcoinHeist's filter.",
        "",
        md_table(pd.DataFrame(PORTABLE_FEATURES, columns=["feature", "rebuildable", "how / why not"]), False),
        "",
        f"After removing the {ov['in_bitcoinheist']:,} addresses already in BitcoinHeist, the "
        f"outside test has {ov['addresses_shared_family_after_overlap']:,} addresses from "
        f"families BitcoinHeist knows and {ov['addresses_new_family_after_overlap']:,} from "
        f"{ov['families_new']} families it does not (counts before the 0.3 BTC filter). The "
        "new-family slice is dominated by NotNevada and the `Unlabeled` group; I would report "
        "NotNevada as its own row and drop `Unlabeled` from the family slices.",
        "",
        "### Open questions for Avi",
        "",
        "1. Split C, or keep A as the main split?",
        "2. K = 10?",
        "3. Price source: Coin Metrics community data is CC BY-NC 4.0 (non-commercial, with "
        "credit). That fits a portfolio project; say if you want a source with a looser license.",
        "4. Python 3.11 is not installed on this machine, so the environment is 3.14 with "
        "`requires-python >= 3.11`. Install 3.11 if you want the pinned version tested.",
        "",
    ]
    return lines
