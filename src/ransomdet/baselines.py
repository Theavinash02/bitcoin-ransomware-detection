import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ransomdet import data, split
from ransomdet.evaluate import detector_metrics, label_metrics, rates_at, threshold_at_fpr
from ransomdet.reporting import fmt
from ransomdet.runlog import timed
from ransomdet.train import fit_lgbm, model_dir

ORIGINAL_8 = ["year", "day", "length", "weight", "count", "looped", "neighbors", "income"]

ORIGINAL_6 = ["length", "weight", "count", "looped", "neighbors", "income"]


BASELINES = {"baseline_a": ORIGINAL_8, "baseline_b": ORIGINAL_6}

RUN_LABELS = {
    "old_protocol": "Old protocol (random 80/20, RF)",
    "baseline_a": "Baseline A (8 features, incl. year/day)",
    "baseline_b": "Baseline B (6 features, no year/day)",
}


def train_baselines(cfg: dict, df: pd.DataFrame, codes: pd.Series, out_dir) -> tuple[pd.DataFrame, dict]:
    """Train Baselines A and B on split C; return val/test predictions and run info."""
    train = split.thin_white(df[codes == split.TRAIN], cfg["sampling"]["white_train_fraction"], cfg["seed"])
    val, test = df[codes == split.VAL], df[codes == split.TEST]
    preds = pd.concat([val[["date", "is_ransom", "label"]], test[["date", "is_ransom", "label"]]])
    preds["split"] = np.r_[np.full(len(val), split.VAL), np.full(len(test), split.TEST)].astype("int8")

    info: dict = {
        "train_rows_after_thinning": len(train),
        "train_ransom_rows": int(train["is_ransom"].sum()),
        "val_rows": len(val),
        "test_rows": len(test),
        "rows_dropped_for_address_overlap": int((codes == split.DROPPED).sum()),
        "models": {},
    }
    for name, feats in BASELINES.items():
        stats: dict = {}
        with timed(stats):
            booster = fit_lgbm(cfg, train[feats], train["is_ransom"], val[feats], val["is_ransom"])
            preds[name] = np.r_[booster.predict(val[feats]), booster.predict(test[feats])]
        booster.save_model(out_dir / f"{name}.txt")
        stats["best_iteration"] = booster.best_iteration
        stats["features"] = feats
        info["models"][name] = stats
        print(f"{name}: {booster.best_iteration} trees, {stats['runtime_s']} s")
    return preds, info


def train_old_protocol(cfg: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rerun the FYP setup with binary labels: random stratified 80/20, original RF settings.

    The forest is fit on the thinned 80% for memory. A random 1/8 of that 80% is held back
    only to fix the FPR threshold, since this protocol has no validation split.
    """
    oc = cfg["old_protocol"]
    idx_tr, idx_te = train_test_split(
        df.index, test_size=oc["test_size"], stratify=df["is_ransom"], random_state=cfg["seed"]
    )
    idx_fit, idx_thr = train_test_split(
        idx_tr, test_size=1 / 8, stratify=df.loc[idx_tr, "is_ransom"], random_state=cfg["seed"]
    )
    fit = split.thin_white(df.loc[idx_fit], cfg["sampling"]["white_train_fraction"], cfg["seed"])
    rf = RandomForestClassifier(
        n_estimators=oc["rf"]["n_estimators"],
        max_depth=oc["rf"]["max_depth"],
        min_samples_leaf=oc["rf"]["min_samples_leaf"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["seed"],
    )
    stats: dict = {}
    with timed(stats):
        rf.fit(fit[ORIGINAL_8].to_numpy(), fit["is_ransom"].to_numpy())
        thr = df.loc[idx_thr]
        te = df.loc[idx_te]
        out = pd.concat([thr[["date", "is_ransom", "label"]], te[["date", "is_ransom", "label"]]])
        out["split"] = np.r_[np.full(len(thr), split.VAL), np.full(len(te), split.TEST)].astype("int8")
        out["old_protocol"] = np.r_[
            rf.predict_proba(thr[ORIGINAL_8].to_numpy())[:, 1],
            rf.predict_proba(te[ORIGINAL_8].to_numpy())[:, 1],
        ]
    stats.update(
        fit_rows_after_thinning=len(fit),
        threshold_rows=len(thr),
        test_rows=len(te),
        test_addresses_also_in_fit=int(te["address"].isin(set(fit["address"])).sum()),
    )
    print(f"old_protocol: {stats['runtime_s']} s")
    return out, stats


def run_baselines(cfg: dict, dev: bool = False) -> dict:
    """Train every baseline run and save predictions and run info under models/baselines/."""
    out_dir = model_dir(cfg, dev, "baselines")
    df = data.load_bitcoinheist(cfg, dev=dev)
    codes = split.load_splits(cfg, dev=dev)
    stats: dict = {}
    with timed(stats):
        preds, info = train_baselines(cfg, df, codes, out_dir)
        old_preds, info["old_protocol"] = train_old_protocol(cfg, df)
    info["run"] = stats
    info["dev"] = dev
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    old_preds.to_parquet(out_dir / "predictions_old_protocol.parquet", index=False)
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info


def score_run(preds: pd.DataFrame, col: str, cfg: dict) -> dict:
    val, test = preds[preds["split"] == 1], preds[preds["split"] == 2]
    m = cfg["metrics"]
    thr = threshold_at_fpr(val["is_ransom"].to_numpy(), val[col].to_numpy(), m["target_fpr"])
    out = {}
    for name, part in (("val", val), ("test", test)):
        out[name] = detector_metrics(
            part["is_ransom"].to_numpy(), part[col].to_numpy(), part["date"].to_numpy(), thr, m["top_k"]
        )
    out["test"]["by_year"] = {
        int(y): rates_at(g["is_ransom"].to_numpy(), g[col].to_numpy(), thr)
        | {"rows": len(g), "positives": int(g["is_ransom"].sum())}
        for y, g in test.groupby(test["date"].dt.year)
    }
    return out


def plot_pr_curves(results: dict, curves: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    fig, ax = plt.subplots(figsize=(6.5, 5))
    styles = {"old_protocol": ":", "baseline_a": "-", "baseline_b": "--"}
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        auc = results[name]["test"]["pr_auc"]
        ax.plot(r, p, styles[name], label=f"{RUN_LABELS[name]}, PR-AUC {auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Test precision-recall (old protocol is on its own random test set)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_baselines_report(cfg: dict, dev: bool = False) -> dict:
    """Score the baseline predictions; write reports/baselines.md, .json and the PR figure."""
    import json

    from ransomdet.train import model_dir

    mdir = model_dir(cfg, dev, "baselines")
    info = json.loads((mdir / "run_info.json").read_text())
    preds = pd.read_parquet(mdir / "predictions.parquet")
    old = pd.read_parquet(mdir / "predictions_old_protocol.parquet")

    results = {name: score_run(preds, name, cfg) for name in ("baseline_a", "baseline_b")}
    results["old_protocol"] = score_run(old, "old_protocol", cfg)
    old_test = old[old["split"] == 2]
    results["old_protocol"]["test"] |= label_metrics(
        old_test["is_ransom"].to_numpy(), old_test["old_protocol"].to_numpy()
    )

    reports = cfg["paths"]["reports"]
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_dev" if dev else ""
    fig_path = fig_dir / f"baselines_pr_curves{suffix}.png"
    test = preds[preds["split"] == 2]
    curves = {
        "old_protocol": (old_test["is_ransom"].to_numpy(), old_test["old_protocol"].to_numpy()),
        "baseline_a": (test["is_ransom"].to_numpy(), test["baseline_a"].to_numpy()),
        "baseline_b": (test["is_ransom"].to_numpy(), test["baseline_b"].to_numpy()),
    }
    plot_pr_curves(results, curves, fig_path)

    m = cfg["metrics"]
    k, fpr_pct = m["top_k"], m["target_fpr"] * 100
    rows = [
        f"| Run | Test PR-AUC | Recall at FPR {fpr_pct:g}% | Test FPR at that threshold | "
        f"Precision in top {k}/day (ceiling) | Brier (uncalibrated) | Val PR-AUC | "
        "Accuracy | Weighted F1 |",
        "| " + " | ".join(["---"] * 9) + " |",
    ]
    for name in ("old_protocol", "baseline_a", "baseline_b"):
        t, v = results[name]["test"], results[name]["val"]
        rows.append(
            f"| {RUN_LABELS[name]} | {fmt(t['pr_auc'])} | {fmt(t['recall_at_fpr'])} | "
            f"{fmt(t['fpr_at_threshold'] * 100, 2)}% | {fmt(t['precision_at_k'])} "
            f"({fmt(t['precision_at_k_ceiling'])}) | {fmt(t['brier'], 4)} | {fmt(v['pr_auc'])} | "
            f"{fmt(t.get('accuracy'), 4)} | {fmt(t.get('f1_weighted'), 4)} |"
        )

    year_rows = [
        "| Run | Year | Rows | Ransomware rows | Recall | FPR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("baseline_a", "baseline_b"):
        for y, r in results[name]["test"]["by_year"].items():
            year_rows.append(
                f"| {RUN_LABELS[name]} | {y} | {r['rows']:,} | {r['positives']:,} | "
                f"{fmt(r['recall'])} | {fmt(r['fpr'] * 100, 2)}% |"
            )

    a, b = results["baseline_a"]["test"], results["baseline_b"]["test"]
    o = info["old_protocol"]
    rf = cfg["old_protocol"]["rf"]
    flag = " --dev" if dev else ""
    lines = [
        "# Forward-in-time baselines",
        "",
        f"Generated by `ransomdet train{flag}` and `ransomdet evaluate{flag}`. Training took "
        f"{info['run']['runtime_s']} s with peak memory {info['run']['peak_rss_mb']:.0f} MB.",
        "",
    ]
    if dev:
        lines += ["Dev sample only: these numbers are not the real results.", ""]
    lines += [
        f"Removing `year` and `day` moves test PR-AUC from {a['pr_auc']:.3f} (A) to "
        f"{b['pr_auc']:.3f} (B), and recall at {fpr_pct:g}% FPR from {a['recall_at_fpr']:.3f} to "
        f"{b['recall_at_fpr']:.3f}. The old random-split protocol scores "
        f"{results['old_protocol']['test']['pr_auc']:.3f} PR-AUC on its own test set.",
        "",
        "## Results",
        "",
        *rows,
        "",
        f"Each threshold is fixed on validation at {fpr_pct:g}% FPR, then applied to test. "
        f"Precision in the top {k} is pooled over all test days (hits / ({k} x days)). The "
        "ceiling is what a perfect ranker would get, since most test days have fewer than "
        f"{k} ransomware rows. Brier scores come from a class-weighted model trained on thinned "
        "data, so they are not calibrated yet (see the final model report). Accuracy and "
        "weighted F1 are given for the old protocol only, at a 0.5 cutoff.",
        "",
        "Leave-one-family-out and Ransomwhere recall are in the generalization report.",
        "",
        f"![PR curves](figures/{fig_path.name})",
        "",
        "## Test years separately",
        "",
        *year_rows,
        "",
        "## Split used",
        "",
        f"Split C: train before {cfg['split']['val_start']}, validation until "
        f"{cfg['split']['test_start']}, test after. {info['rows_dropped_for_address_overlap']:,} "
        "rows were dropped because their address already appeared in an earlier split. "
        f"Training keeps {cfg['sampling']['white_train_fraction']:.0%} of white rows: "
        f"{info['train_rows_after_thinning']:,} rows, {info['train_ransom_rows']:,} of them "
        f"ransomware. Validation ({info['val_rows']:,} rows) and test ({info['test_rows']:,} "
        "rows) use every row.",
        "",
        "## Old protocol details",
        "",
        f"Random stratified 80/20 split over all rows, binary labels, the FYP forest settings "
        f"({rf['n_estimators']} trees, max depth {rf['max_depth']}, min leaf "
        f"{rf['min_samples_leaf']}). For memory, the forest is fit on the 80% with white rows "
        f"thinned the same way ({o['fit_rows_after_thinning']:,} rows). A random eighth of the "
        f"80% ({o['threshold_rows']:,} rows) is held back only to fix the FPR threshold. "
        f"{o['test_addresses_also_in_fit']:,} of the {o['test_rows']:,} test rows belong to "
        "addresses the forest saw in training.",
        "",
        "## Runs",
        "",
        "| Model | Trees | Fit time (s) |",
        "| --- | --- | --- |",
        *[
            f"| {RUN_LABELS[n]} | {s['best_iteration']} (early stopped) | {s['runtime_s']} |"
            for n, s in info["models"].items()
        ],
        f"| {RUN_LABELS['old_protocol']} | {rf['n_estimators']} | {o['runtime_s']} |",
        "",
    ]
    (reports / f"baselines{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "dev": dev,
        "config": {"split": cfg["split"], "metrics": m},
        "results": results,
        "runs": info,
    }
    (reports / f"baselines{suffix}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {reports / f'baselines{suffix}.md'}")
    return payload
