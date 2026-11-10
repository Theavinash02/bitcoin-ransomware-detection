import json
import shutil

import numpy as np
import pandas as pd

from ransomdet import features
from ransomdet.calibrate import Calibrator, choose_calibrator, reliability
from ransomdet.evaluate import detector_metrics, rates_at, threshold_at_fpr
from ransomdet.reporting import dev_suffix, fmt, pyplot
from ransomdet.runlog import timed
from ransomdet.train import fit_tuned, load_split_frames, model_dir

FINAL = "final"


def chosen_features(cfg: dict, dev: bool) -> list[str]:
    path = cfg["paths"]["reports"] / f"feature_ablation{dev_suffix(dev)}.json"
    return json.loads(path.read_text())["chosen_features"]


def tune(cfg: dict, X_tr, y_tr, X_val, y_val, val_day) -> tuple[dict, list[dict]]:
    """Random search on validation PR-AUC; return the best parameters and every trial."""
    space = cfg["final_model"]["search"]
    rng = np.random.default_rng(cfg["seed"])
    trials = []
    seen = set()
    while len(trials) < cfg["final_model"]["trials"]:
        params = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(params.items()))
        if key in seen:
            continue
        seen.add(key)
        params = {k: (float(v) if isinstance(v, float) else int(v)) for k, v in params.items()}
        b = fit_tuned(cfg, params, X_tr, y_tr, X_val, y_val)
        s = b.predict(X_val)
        m = detector_metrics(
            y_val,
            s,
            val_day,
            threshold_at_fpr(y_val, s, cfg["metrics"]["target_fpr"]),
            cfg["metrics"]["top_k"],
        )
        trials.append(params | {"trees": b.best_iteration, "val_pr_auc": m["pr_auc"]})
        print(f"trial {len(trials)}: val PR-AUC {m['pr_auc']:.3f}")
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return {k: best[k] for k in space}, trials


def _log_features(X: pd.DataFrame) -> pd.DataFrame:
    # Heavy-tailed counts and amounts swamp a linear model; log1p the non-negative columns.
    X = X.copy()
    for c in X.columns:
        if (X[c] >= 0).all():
            X[c] = np.log1p(X[c])
    return X


def comparison_models(cfg: dict, X_tr, y_tr, X_val, X_te) -> dict:
    """Return val/test scores for Logistic Regression and Random Forest + SMOTE."""
    from imblearn.over_sampling import SMOTE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cfg["seed"])
    )
    lr.fit(_log_features(X_tr), y_tr)
    out["logistic_regression"] = (
        lr.predict_proba(_log_features(X_val))[:, 1],
        lr.predict_proba(_log_features(X_te))[:, 1],
    )
    print("logistic regression done")

    ratio = cfg["final_model"]["smote_ratio"]
    if (y_tr == 1).sum() / (y_tr == 0).sum() < ratio:
        sm = SMOTE(sampling_strategy=ratio, random_state=cfg["seed"])
        X_sm, y_sm = sm.fit_resample(X_tr.to_numpy(), y_tr)
    else:
        # The dev sample is already more balanced than the target.
        X_sm, y_sm = X_tr.to_numpy(), y_tr
    rf_cfg = cfg["old_protocol"]["rf"]
    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["seed"],
    )
    rf.fit(X_sm, y_sm)
    out["rf_smote"] = (rf.predict_proba(X_val.to_numpy())[:, 1], rf.predict_proba(X_te.to_numpy())[:, 1])
    print("random forest + SMOTE done")
    return out


def score_at_val_threshold(cfg: dict, y_val, s_val, y_te, s_te, day_te) -> tuple[float, dict]:
    thr = threshold_at_fpr(y_val, s_val, cfg["metrics"]["target_fpr"])
    return thr, detector_metrics(y_te, s_te, day_te, thr, cfg["metrics"]["top_k"])


def plot_reliability(rel_val: pd.DataFrame, rel_te: pd.DataFrame, path) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([1e-4, 1], [1e-4, 1], color="grey", lw=1)
    ax.plot(rel_val["predicted"], rel_val["observed"], "o-", label="validation (fit here)")
    ax.plot(rel_te["predicted"], rel_te["observed"], "s--", label="test")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Mean calibrated probability (bin)")
    ax.set_ylabel("Observed ransomware rate")
    ax.set_title("Reliability, 10 equal-size bins", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr(curves: dict, path, title: str) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    top = 0.0
    for name, (y, s, style) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, linestyle=style, label=f"{name}, PR-AUC {average_precision_score(y, s):.3f}")
        # Ignore the first few alerts, where precision jumps around on a handful of rows.
        top = max(top, p[r >= 0.01].max())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.02, top * 1.3))
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_final_model(cfg: dict, dev: bool = False) -> dict:
    """Tune, calibrate and fix operating points on validation, then score the test years once."""
    suffix = dev_suffix(dev)
    feats = chosen_features(cfg, dev)
    out_dir = model_dir(cfg, dev, FINAL)
    stats: dict = {}
    with timed(stats):
        train, val, test = load_split_frames(cfg, dev)
        X_tr, X_val, X_te = (features.feature_matrix(d, feats) for d in (train, val, test))
        y_tr, y_val, y_te = (d["is_ransom"].to_numpy() for d in (train, val, test))
        d_val, d_te = val["date"].to_numpy(), test["date"].to_numpy()

        best, trials = tune(cfg, X_tr, y_tr, X_val, y_val, d_val)
        booster = fit_tuned(cfg, best, X_tr, y_tr, X_val, y_val)
        raw_val, raw_te = booster.predict(X_val), booster.predict(X_te)

        choice = choose_calibrator(raw_val, y_val, d_val)
        cal = Calibrator(choice["method"]).fit(raw_val, y_val)
        p_val, p_te = cal.predict(raw_val), cal.predict(raw_te)

        # Alerts use the raw score: calibration is monotone but isotonic steps create ties.
        thr, test_m = score_at_val_threshold(cfg, y_val, raw_val, y_te, raw_te, d_te)
        val_m = detector_metrics(y_val, raw_val, d_val, thr, cfg["metrics"]["top_k"])
        for m, p, y in ((val_m, p_val, y_val), (test_m, p_te, y_te)):
            m["brier"] = float(np.mean((p - y) ** 2))
            m["brier_base_rate"] = float(np.mean((y.mean() - y) ** 2))
        test_m["by_year"] = {
            int(yr): rates_at(y_te[m], raw_te[m], thr)
            | {"rows": int(m.sum()), "positives": int(y_te[m].sum())}
            for yr in np.unique(pd.DatetimeIndex(d_te).year)
            for m in [pd.DatetimeIndex(d_te).year == yr]
        }
        test_m["by_family"] = (
            pd.DataFrame({"label": test["label"].astype(str).to_numpy(), "hit": raw_te > thr, "y": y_te})
            .query("y == 1")
            .groupby("label")["hit"]
            .agg(["size", "mean"])
            .rename(columns={"size": "rows", "mean": "recall"})
            .sort_values("rows", ascending=False)
            .to_dict("index")
        )

        comps = comparison_models(cfg, X_tr, y_tr, X_val, X_te)
        comp_m = {}
        for name, (sv, st) in comps.items():
            _, comp_m[name] = score_at_val_threshold(cfg, y_val, sv, y_te, st, d_te)
            comp_m[name]["val_pr_auc"] = detector_metrics(y_val, sv, d_val, 0.5, 10)["pr_auc"]

    booster.save_model(out_dir / "model.txt")
    cal.save(out_dir / "calibrator.json")
    threshold = {
        "raw_threshold": thr,
        "calibrated_at_threshold": float(cal.predict(np.array([thr]))[0]),
        "target_fpr": cfg["metrics"]["target_fpr"],
        "top_k_per_day": cfg["metrics"]["top_k"],
        "calibration": choice,
    }
    (out_dir / "threshold.json").write_text(json.dumps(threshold, indent=2))
    (out_dir / "features.json").write_text(json.dumps(feats, indent=2))
    shutil.copy(cfg["paths"]["processed"] / features.QUANTILES_PARQUET, out_dir / features.QUANTILES_PARQUET)
    pd.DataFrame(
        {
            "date": test["date"].to_numpy(),
            "label": test["label"].astype(str).to_numpy(),
            "y": y_te,
            "raw": raw_te,
            "p": p_te,
        }
    ).to_parquet(out_dir / "test_predictions.parquet", index=False)
    model_mb = (out_dir / "model.txt").stat().st_size / 2**20

    reports = cfg["paths"]["reports"]
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rel_val, rel_te = reliability(y_val, p_val), reliability(y_te, p_te)
    plot_reliability(rel_val, rel_te, fig_dir / f"final_model_reliability{suffix}.png")

    p1 = json.loads((reports / f"baselines{suffix}.json").read_text())
    p1_preds = pd.read_parquet(model_dir(cfg, dev, "baselines") / "predictions.parquet")
    p1_te = p1_preds[p1_preds["split"] == 2]
    plot_pr(
        {
            "Final LightGBM": (y_te, raw_te, "-"),
            "Baseline A": (p1_te["is_ransom"].to_numpy(), p1_te["baseline_a"].to_numpy(), ":"),
            "Baseline B": (p1_te["is_ransom"].to_numpy(), p1_te["baseline_b"].to_numpy(), "-."),
            "Logistic Regression": (y_te, comps["logistic_regression"][1], "--"),
            "Random Forest + SMOTE": (y_te, comps["rf_smote"][1], (0, (1, 3))),
        },
        fig_dir / f"final_model_pr_curves{suffix}.png",
        "Test precision-recall, 2017-18",
    )

    k, fpr_pct = cfg["metrics"]["top_k"], cfg["metrics"]["target_fpr"] * 100
    rows = [
        f"| Model | Features | Test PR-AUC | Recall at FPR {fpr_pct:g}% | Test FPR | "
        f"Precision in top {k}/day (ceiling {test_m['precision_at_k_ceiling']:.3f}) | Val PR-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def row(name, nfeat, m, val_auc):
        return (
            f"| {name} | {nfeat} | {fmt(m['pr_auc'])} | {fmt(m['recall_at_fpr'])} | "
            f"{fmt(m['fpr_at_threshold'] * 100, 2)}% | {fmt(m['precision_at_k'])} | {fmt(val_auc)} |"
        )

    rows.append(row("**Final LightGBM (tuned)**", len(feats), test_m, val_m["pr_auc"]))
    for key, label in (("baseline_a", "Baseline A"), ("baseline_b", "Baseline B")):
        r = p1["results"][key]
        rows.append(row(label, len(p1["runs"]["models"][key]["features"]), r["test"], r["val"]["pr_auc"]))
    rows.append(
        row(
            "Logistic Regression",
            len(feats),
            comp_m["logistic_regression"],
            comp_m["logistic_regression"]["val_pr_auc"],
        )
    )
    rows.append(
        row("Random Forest + SMOTE", len(feats), comp_m["rf_smote"], comp_m["rf_smote"]["val_pr_auc"])
    )

    fam_rows = ["| Family | Test rows | Recall at the threshold |", "| --- | --- | --- |"]
    for fam, r in test_m["by_family"].items():
        fam_rows.append(f"| {fam} | {r['rows']:,} | {fmt(r['recall'])} |")
    year_rows = ["| Year | Rows | Ransomware rows | Recall | FPR |", "| --- | --- | --- | --- | --- |"]
    for yr, r in test_m["by_year"].items():
        year_rows.append(
            f"| {yr} | {r['rows']:,} | {r['positives']:,} | {fmt(r['recall'])} | {fmt(r['fpr'] * 100, 2)}% |"
        )
    trial_df = pd.DataFrame(trials).sort_values("val_pr_auc", ascending=False).head(5)
    trial_rows = [
        "| " + " | ".join(trial_df.columns) + " |",
        "| " + " | ".join(["---"] * len(trial_df.columns)) + " |",
    ]
    for t in trial_df.itertuples(index=False):
        trial_rows.append("| " + " | ".join(fmt(v, 4) if isinstance(v, float) else str(v) for v in t) + " |")
    rel_rows = ["| Bin | Mean predicted | Observed | Rows |", "| --- | --- | --- | --- |"]
    for b, r in rel_te.iterrows():
        rel_rows.append(f"| {b + 1} | {r['predicted']:.4f} | {r['observed']:.4f} | {int(r['rows']):,} |")

    flag = " --dev" if dev else ""
    lines = [
        "# Final model and operating point",
        "",
        f"Generated by `ransomdet final{flag}`. Runtime {stats['runtime_s']} s, peak memory "
        f"{stats['peak_rss_mb']:.0f} MB.",
        "",
        *(["Dev sample only: these numbers are not the real results.", ""] if dev else []),
        f"On the test years the tuned model reaches PR-AUC {test_m['pr_auc']:.3f} and recall "
        f"{test_m['recall_at_fpr']:.3f} at the validation-fixed {fpr_pct:g}% FPR threshold, against "
        f"{p1['results']['baseline_b']['test']['pr_auc']:.3f} and "
        f"{p1['results']['baseline_b']['test']['recall_at_fpr']:.3f} for Baseline B. The test base rate "
        f"is {test_m['base_rate']:.4f}, so a random ranker would score about that PR-AUC.",
        "",
        "## Test results",
        "",
        *rows,
        "",
        f"All thresholds are fixed on validation at {fpr_pct:g}% FPR. Precision in the top {k} is pooled "
        "over all test days; the ceiling is the best a perfect ranker could do. Comparison models use "
        "the same split and features. Logistic Regression gets log1p of non-negative features and "
        f"standard scaling; the forest uses the FYP settings after SMOTE to a "
        f"{cfg['final_model']['smote_ratio']:g} minority ratio on the thinned training data.",
        "",
        "![PR curves](figures/final_model_pr_curves" + suffix + ".png)",
        "",
        "## Test years and families",
        "",
        *year_rows,
        "",
        *fam_rows,
        "",
        "## Calibration",
        "",
        f"Calibrator: {choice['method']}, chosen by out-of-fold Brier score on validation over blocks "
        f"of whole days (isotonic {choice['oof_brier']['isotonic']:.4f}, sigmoid "
        f"{choice['oof_brier']['sigmoid']:.4f}). Brier on validation {val_m['brier']:.4f} (predicting "
        f"the base rate gives {val_m['brier_base_rate']:.4f}); on test {test_m['brier']:.4f} (base "
        f"rate {test_m['brier_base_rate']:.4f}). The validation ransomware rate is "
        f"{val_m['base_rate']:.3f} and the test rate {test_m['base_rate']:.4f}, so probabilities "
        "calibrated on validation run high on test.",
        "",
        *rel_rows,
        "",
        "![Reliability](figures/final_model_reliability" + suffix + ".png)",
        "",
        "## Operating points",
        "",
        f"Saved to `models/{FINAL}/threshold.json`: raw score threshold {thr:.6f} (calibrated "
        f"probability {threshold['calibrated_at_threshold']:.4f}) for {fpr_pct:g}% FPR on validation, "
        f"and a budget of {k} alerts per day. Alerts compare the raw score, since a calibrator "
        f"can be flat in places and turn distinct scores into ties. Model file: {model_mb:.2f} MB.",
        "",
        "## Tuning",
        "",
        f"{len(trials)} random trials, scored by validation PR-AUC. Best five:",
        "",
        *trial_rows,
        "",
    ]
    (reports / f"final_model{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "dev": dev,
        "features": feats,
        "best_params": best,
        "trees": booster.best_iteration,
        "trials": trials,
        "calibration": choice,
        "threshold": threshold,
        "val": val_m,
        "test": test_m,
        "comparison": comp_m,
        "reliability_test": rel_te.reset_index().to_dict("records"),
        "model_mb": model_mb,
        "run": stats,
    }
    (reports / f"final_model{suffix}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {reports / f'final_model{suffix}.md'}")
    return payload


def load_final_report(cfg: dict, dev: bool) -> dict:
    return json.loads((cfg["paths"]["reports"] / f"final_model{dev_suffix(dev)}.json").read_text())
