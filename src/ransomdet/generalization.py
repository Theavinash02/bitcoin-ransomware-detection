import json

import pandas as pd

from ransomdet import data, features
from ransomdet.evaluate import detector_metrics, threshold_at_fpr
from ransomdet.final_model import load_final_report
from ransomdet.reporting import dev_suffix
from ransomdet.runlog import timed
from ransomdet.train import fit_tuned, load_split_frames


def leave_one_family_out(cfg: dict, dev: bool, p3: dict) -> list[dict]:
    """Retrain without each large family and measure recall on all of its rows."""
    from ransomdet import split

    feats, params = p3["features"], p3["best_params"]
    cols = ["address", "date", "label", "is_ransom", *feats]
    df = features.load_feature_table(cfg, dev=dev, columns=cols)
    codes = split.load_splits(cfg, dev=dev).to_numpy()
    df["label"] = df["label"].astype(str)
    frac = cfg["sampling"]["white_train_fraction"]
    train_all = split.thin_white(df[codes == split.TRAIN], frac, cfg["seed"])
    val_all, test = df[codes == split.VAL], df[codes == split.TEST]
    test_white = test[test["is_ransom"] == 0]

    ransom = df[df["is_ransom"] == 1]
    sizes = ransom.groupby("label")["address"].nunique()
    families = sizes[sizes >= cfg["generalization"]["min_family_addresses"]].sort_values(ascending=False)
    rows = []
    for fam, n_addr in families.items():
        tr, va = train_all[train_all["label"] != fam], val_all[val_all["label"] != fam]
        b = fit_tuned(
            cfg,
            params,
            features.feature_matrix(tr, feats),
            tr["is_ransom"].to_numpy(),
            features.feature_matrix(va, feats),
            va["is_ransom"].to_numpy(),
        )
        s_val = b.predict(features.feature_matrix(va, feats))
        thr = threshold_at_fpr(va["is_ransom"].to_numpy(), s_val, cfg["metrics"]["target_fpr"])
        fam_rows = df[df["label"] == fam]
        hit = pd.Series(b.predict(features.feature_matrix(fam_rows, feats)) > thr, index=fam_rows.index)
        fpr = float((b.predict(features.feature_matrix(test_white, feats)) > thr).mean())
        ref = p3["test"]["by_family"].get(fam)
        rows.append(
            {
                "family": fam,
                "addresses": int(n_addr),
                "rows": len(fam_rows),
                "years": f"{fam_rows['date'].dt.year.min()}-{fam_rows['date'].dt.year.max()}",
                "held_out_recall": float(hit.mean()),
                "held_out_address_recall": float(hit.groupby(fam_rows["address"].astype(str)).max().mean()),
                "in_training_test_recall": ref["recall"] if ref else None,
                "in_training_test_rows": ref["rows"] if ref else 0,
                "test_white_fpr": fpr,
            }
        )
        print(f"LOFO {fam}: recall {rows[-1]['held_out_recall']:.3f}")
    return rows


def ransomwhere_test(cfg: dict, dev: bool, p3: dict) -> dict:
    """Train the portable model on BitcoinHeist and score Ransomwhere addresses it has never seen."""
    from ransomdet import ransomwhere

    params = p3["best_params"]
    feats = features.PORTABLE
    train, val, test = load_split_frames(cfg, dev)
    b = fit_tuned(
        cfg,
        params,
        features.feature_matrix(train, feats),
        train["is_ransom"].to_numpy(),
        features.feature_matrix(val, feats),
        val["is_ransom"].to_numpy(),
    )
    s_val = b.predict(features.feature_matrix(val, feats))
    thr = threshold_at_fpr(val["is_ransom"].to_numpy(), s_val, cfg["metrics"]["target_fpr"])
    s_te = b.predict(features.feature_matrix(test, feats))
    y_te = test["is_ransom"].to_numpy()
    bh_test = detector_metrics(y_te, s_te, test["date"].to_numpy(), thr, cfg["metrics"]["top_k"])
    white = y_te == 0
    white_addr = test["address"].astype(str).to_numpy()[white]
    white_addr_fpr = float(pd.Series(s_te[white] > thr).groupby(white_addr).max().mean())

    full = data.load_bitcoinheist(cfg, columns=["address", "label"])
    bh_addrs = set(full["address"].astype(str).unique())
    bh_labels = [x for x in full["label"].astype(str).unique() if x != "white"]
    del full
    days = ransomwhere.address_days(cfg, exclude=bh_addrs)
    days["hit"] = b.predict(features.feature_matrix(days, feats)) > thr
    slices = []
    for name, mask in ransomwhere.family_slices(days, bh_labels).items():
        d = days[mask]
        if not len(d):
            continue
        by_addr = d.groupby("address")["hit"].max()
        slices.append(
            {
                "slice": name,
                "addresses": int(by_addr.size),
                "address_days": len(d),
                "families": int(d["family"].nunique()),
                "address_recall": float(by_addr.mean()),
                "address_day_recall": float(d["hit"].mean()),
            }
        )
    top_fams = (
        days[days["family"] != "Unlabeled"]
        .groupby(["family", "address"])["hit"]
        .max()
        .groupby("family")
        .agg(["size", "mean"])
        .rename(columns={"size": "addresses", "mean": "address_recall"})
        .sort_values("addresses", ascending=False)
        .head(12)
    )
    return {
        "features": feats,
        "threshold": thr,
        "bh_test": bh_test,
        "bh_test_white_address_fpr": white_addr_fpr,
        "slices": slices,
        "top_families": top_fams.reset_index().to_dict("records"),
        "date_range": [str(days["date"].min().date()), str(days["date"].max().date())],
    }


def run_generalization(cfg: dict, dev: bool = False) -> dict:
    """Write reports/generalization.md: leave-one-family-out and the Ransomwhere outside test."""
    suffix = dev_suffix(dev)
    p3 = load_final_report(cfg, dev)
    stats: dict = {}
    with timed(stats):
        lofo = leave_one_family_out(cfg, dev, p3)
        rw = ransomwhere_test(cfg, dev, p3)

    fpr_pct = cfg["metrics"]["target_fpr"] * 100
    lofo_rows = [
        "| Held-out family | Addresses | Rows | Years | Recall held out (rows) | Recall held out "
        "(addresses) | Recall in final model, test years | Test white FPR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in lofo:
        ref = "-"
        if r["in_training_test_recall"] is not None:
            ref = f"{r['in_training_test_recall']:.3f} ({r['in_training_test_rows']:,} rows)"
        lofo_rows.append(
            f"| {r['family']} | {r['addresses']:,} | {r['rows']:,} | {r['years']} | "
            f"{r['held_out_recall']:.3f} | {r['held_out_address_recall']:.3f} | {ref} | "
            f"{r['test_white_fpr'] * 100:.2f}% |"
        )
    rw_rows = [
        "| Slice | Addresses | Address-days | Families | Address recall | Address-day recall |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for s in rw["slices"]:
        rw_rows.append(
            f"| {s['slice']} | {s['addresses']:,} | {s['address_days']:,} | {s['families']} | "
            f"{s['address_recall']:.3f} | {s['address_day_recall']:.3f} |"
        )
    fam_rows = ["| Family | Addresses | Address recall |", "| --- | --- | --- |"]
    for f in rw["top_families"]:
        fam_rows.append(f"| {f['family']} | {f['addresses']:,} | {f['address_recall']:.3f} |")
    allrow = rw["slices"][0]
    flag = " --dev" if dev else ""
    feat_list = ", ".join(f"`{c}`" for c in rw["features"])
    bt = rw["bh_test"]
    lines = [
        "# Ransomware the model never saw",
        "",
        f"Generated by `ransomdet generalize{flag}`. Runtime {stats['runtime_s']} s, peak memory "
        f"{stats['peak_rss_mb']:.0f} MB.",
        "",
        *(["Dev sample only: these numbers are not the real results.", ""] if dev else []),
        "## Leave one family out (BitcoinHeist)",
        "",
        f"Each family with at least {cfg['generalization']['min_family_addresses']} addresses is "
        f"removed from training and validation. The detector is retrained with the final model settings "
        f"and its "
        f"{fpr_pct:g}% FPR threshold is fixed again on the remaining validation rows. Recall is "
        "measured on every row of the held-out family, from any year. The reference column is the "
        "same family's test-year recall from the final model, which saw the family in training.",
        "",
        *lofo_rows,
        "",
        "## Outside test on Ransomwhere",
        "",
        f"The portable model uses only what the Ransomwhere payment list can rebuild: {feat_list}. "
        "It is trained on BitcoinHeist with the final model split and settings, and its threshold is "
        f"fixed at {fpr_pct:g}% FPR on validation. On the BitcoinHeist test years it reaches PR-AUC "
        f"{bt['pr_auc']:.3f}, recall {bt['recall_at_fpr']:.3f} and an address-day FPR of "
        f"{bt['fpr_at_threshold'] * 100:.2f}%. {rw['bh_test_white_address_fpr'] * 100:.2f}% of white "
        "test addresses are flagged on at least one day.",
        "",
        "The test set is every Ransomwhere bitcoin address that is not in BitcoinHeist, with payments "
        f"from {rw['date_range'][0]} to {rw['date_range'][1]}. Repeated payment rows and payments "
        "under 0.3 BTC are dropped first, as BitcoinHeist did. An address counts as caught when any "
        f"of its days crosses the threshold. Overall address recall: {allrow['address_recall']:.3f} "
        f"on {allrow['addresses']:,} addresses.",
        "",
        *rw_rows,
        "",
        "Largest labeled families in the outside test:",
        "",
        *fam_rows,
        "",
        "Caveats: `income` and `neighbors` are rebuilt from the Ransomwhere payment list and follow "
        "the BitcoinHeist definitions only approximately. BitcoinHeist counts every transaction "
        "paying the address that day; Ransomwhere lists only the payments its contributors "
        "attributed. Ransomwhere has no benign addresses, so the false-positive rate comes from the "
        "BitcoinHeist test years. Addresses with more payment days get more chances to be caught.",
        "",
        "Could not be rebuilt: `count`, `weight`, `length`, `looped` (they need the whole day of "
        "transactions), `count_per_neighbor`, and Group E (it needs every other address active "
        "that day).",
        "",
    ]
    reports = cfg["paths"]["reports"]
    (reports / f"generalization{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {"dev": dev, "lofo": lofo, "ransomwhere": rw, "run": stats}
    (reports / f"generalization{suffix}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {reports / f'generalization{suffix}.md'}")
    return payload
