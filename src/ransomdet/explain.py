import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from ransomdet import features
from ransomdet.baselines import BASELINES
from ransomdet.final_model import FINAL
from ransomdet.reporting import dev_suffix, pyplot
from ransomdet.runlog import timed
from ransomdet.train import load_split_frames, model_dir


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    """Return per-row SHAP contributions (log-odds) from LightGBM, without the bias column."""
    contrib = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def shap_sample(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Return up to n test rows, with ransomware rows filling at most half of them."""
    ransom = test[test["is_ransom"] == 1]
    ransom = ransom.sample(n=min(len(ransom), n // 2), random_state=seed)
    white = test[test["is_ransom"] == 0].sample(n=n - len(ransom), random_state=seed)
    return pd.concat([ransom, white])


def plot_importance_bar(imp: pd.Series, path, title: str) -> None:
    plt = pyplot()
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(imp) + 1))
    ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_xlabel("Mean |SHAP| (log-odds)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_beeswarm(sv: pd.DataFrame, X: pd.DataFrame, order: list[str], path) -> None:
    plt = pyplot()
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.2))
    for i, f in enumerate(reversed(order)):
        color = X[f].rank(pct=True).to_numpy()
        y = i + rng.uniform(-0.3, 0.3, len(sv))
        sc = ax.scatter(sv[f], y, c=color, cmap="coolwarm", s=4, alpha=0.6, linewidths=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)))
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value (log-odds, positive pushes toward ransomware)")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("Feature value (percentile in sample)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dependence(x: pd.Series, s: pd.Series, y: np.ndarray, path, xlabel: str, logx: bool) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, mask, color in (("white", y == 0, "#7f8c8d"), ("ransomware", y == 1, "#c0392b")):
        ax.scatter(x[mask], s[mask], s=5, alpha=0.45, color=color, label=label, linewidths=0)
    if logx:
        ax.set_xscale("log")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SHAP value (log-odds)")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_side_by_side(a: pd.Series, b: pd.Series, path) -> None:
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, imp, title in ((axes[0], a, "Baseline A (with year and day)"), (axes[1], b, "Final model")):
        imp = (imp / imp.sum()).sort_values().tail(10)
        colors = ["#c0392b" if f in ("year", "day") else "#3b6ea5" for f in imp.index]
        ax.barh(imp.index, imp.values, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Share of mean |SHAP|")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def misses(cfg: dict, test: pd.DataFrame, raw: np.ndarray, thr: float) -> tuple[pd.DataFrame, dict]:
    """Compare caught and missed test ransomware addresses at the fixed threshold."""
    r = test[test["is_ransom"] == 1].copy()
    r["hit"] = raw[test["is_ransom"].to_numpy() == 1] > thr
    r["address"] = r["address"].astype(str)
    r["label"] = r["label"].astype(str)
    caught = r.groupby("address")["hit"].transform("max")
    r["group"] = np.where(caught, "caught", "missed")
    cols = [
        "income_usd",
        "income_btc",
        "round_min",
        "round_usd_100",
        "round_btc_0.1",
        "neighbors",
        "count",
        "length",
        "weight",
        "pct_income_usd",
    ]
    cols = [c for c in cols if c in r.columns]
    med = r.groupby("group")[cols].median().T
    white = test[test["is_ransom"] == 0][cols].median().rename("white (test)")
    med = med.join(white)
    by_fam = r.groupby(["label", "group"])["address"].nunique().unstack(fill_value=0)
    info = {
        "addresses": int(r["address"].nunique()),
        "caught_addresses": int(r.loc[r["group"] == "caught", "address"].nunique()),
        "by_family": by_fam.to_dict("index"),
        "round_btc_share": {
            g: float((r.loc[r["group"] == g, "round_btc_0.1"] < 0.01).mean()) for g in ("caught", "missed")
        },
        "round_btc_share_white": float((test.loc[test["is_ransom"] == 0, "round_btc_0.1"] < 0.01).mean()),
    }
    # After-the-fact check, not a model: how far the round-BTC pattern alone goes on test.
    rule = test["round_btc_0.1"].to_numpy() < 0.01
    y = test["is_ransom"].to_numpy() == 1
    info["round_rule"] = {
        "recall": float(rule[y].mean()),
        "precision": float(y[rule].mean()) if rule.any() else 0.0,
        "base_rate": float(y.mean()),
    }
    return med, info


def run_explanation(cfg: dict, dev: bool = False) -> dict:
    """Write reports/explanation.md with SHAP figures and the caught-vs-missed comparison."""
    suffix = dev_suffix(dev)
    reports = cfg["paths"]["reports"]
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    mdir = model_dir(cfg, dev, FINAL)
    feats = json.loads((mdir / "features.json").read_text())
    thr = json.loads((mdir / "threshold.json").read_text())["raw_threshold"]
    booster = lgb.Booster(model_file=str(mdir / "model.txt"))
    base_a = lgb.Booster(model_file=str(model_dir(cfg, dev, "baselines") / "baseline_a.txt"))

    stats: dict = {}
    with timed(stats):
        _, _, test = load_split_frames(cfg, dev)
        sample = shap_sample(test, cfg["explanation"]["shap_rows"], cfg["seed"])
        X = features.feature_matrix(sample, feats)
        sv = shap_values(booster, X)
        imp = sv.abs().mean().sort_values(ascending=False)
        sv_a = shap_values(base_a, features.feature_matrix(sample, BASELINES["baseline_a"]))
        imp_a = sv_a.abs().mean().sort_values(ascending=False)
        raw = booster.predict(features.feature_matrix(test, feats))
        med, miss = misses(cfg, test, raw, thr)

    top = list(imp.index[:12])
    plot_importance_bar(
        imp, fig_dir / f"explanation_importance{suffix}.png", "Final model, global importance"
    )
    plot_beeswarm(sv, X, top, fig_dir / f"explanation_beeswarm{suffix}.png")
    y = sample["is_ransom"].to_numpy()
    round_feat = max(
        [c for c in feats if c.startswith("round_")], key=lambda c: imp.get(c, 0), default="round_min"
    )
    plot_dependence(
        X[round_feat],
        sv[round_feat],
        y,
        fig_dir / f"explanation_dependence_round{suffix}.png",
        f"{round_feat} (0 = exactly round)",
        False,
    )
    money = next(f for f in ("income_usd", "log_income_usd", "income") if f in feats)
    plot_dependence(
        X[money].clip(lower=1e-9),
        sv[money],
        y,
        fig_dir / f"explanation_dependence_income_usd{suffix}.png",
        f"{money} (log scale)",
        True,
    )
    plot_side_by_side(imp_a, imp, fig_dir / f"explanation_side_by_side{suffix}.png")

    date_share = float(imp_a[["year", "day"]].sum() / imp_a.sum())
    caught_share = miss["caught_addresses"] / max(miss["addresses"], 1)
    fam = pd.DataFrame(miss["by_family"]).T.fillna(0).astype(int)
    if "missed" not in fam:
        fam["missed"] = 0
    if "caught" not in fam:
        fam["caught"] = 0
    fam = fam.sort_values("missed", ascending=False)
    biggest_missed = fam.index[0] if len(fam) else "-"
    findings = [
        f"The three largest drivers are {', '.join(f'`{f}`' for f in imp.index[:3])}; together they carry "
        f"{imp.iloc[:3].sum() / imp.sum():.0%} of the mean absolute SHAP value.",
        f"`year` and `day` carry {date_share:.0%} of Baseline A's mean absolute SHAP value. The final "
        "model has no date inputs; date enters only through the dollar price lookup and the same-day "
        "percentiles.",
        f"Only {miss['caught_addresses']:,} of {miss['addresses']:,} test ransomware addresses "
        f"({caught_share:.1%}) have a day above the threshold. {biggest_missed} accounts for "
        f"{fam.loc[biggest_missed, 'missed'] if len(fam) else 0:,} of the missed addresses.",
        f"Caught addresses have median income ${med.loc['income_usd', 'caught']:,.0f} against "
        f"${med.loc['income_usd', 'missed']:,.0f} for missed ones and "
        f"${med.loc['income_usd', 'white (test)']:,.0f} for white rows.",
        f"Most test ransomware is paid in round BTC: {miss['round_btc_share']['missed']:.0%} of missed "
        f"rows sit within 1% of a 0.1 BTC multiple, against {miss['round_btc_share_white']:.0%} of white "
        f"rows. That rule alone, checked after the fact on test, finds "
        f"{miss['round_rule']['recall']:.0%} of test ransomware rows at "
        f"{miss['round_rule']['precision']:.1%} precision "
        f"(base rate {miss['round_rule']['base_rate']:.1%}). The model does not rank these "
        "rows high enough to pass a 0.1% FPR threshold, because round amounts are common enough among "
        "white rows that the signal alone is far from that strict.",
    ]
    med_rows = ["| Feature | Caught | Missed | White (test) |", "| --- | --- | --- | --- |"]
    for f, r in med.iterrows():
        med_rows.append(
            f"| `{f}` | {r.get('caught', np.nan):,.4g} | {r.get('missed', np.nan):,.4g} | "
            f"{r['white (test)']:,.4g} |"
        )
    fam_rows = ["| Family | Caught addresses | Missed addresses |", "| --- | --- | --- |"]
    for f, r in fam.iterrows():
        fam_rows.append(f"| {f} | {r['caught']:,} | {r['missed']:,} |")
    imp_rows = ["| Feature | Mean abs SHAP | Share |", "| --- | --- | --- |"]
    for f, v in imp.items():
        imp_rows.append(f"| `{f}` | {v:.4f} | {v / imp.sum():.1%} |")

    flag = " --dev" if dev else ""
    lines = [
        "# What drives the detector",
        "",
        f"Generated by `ransomdet explain{flag}`. Runtime {stats['runtime_s']} s, peak memory "
        f"{stats['peak_rss_mb']:.0f} MB.",
        "",
        *(["Dev sample only: these numbers are not the real results.", ""] if dev else []),
        f"SHAP values come from LightGBM's built-in contributions (`pred_contrib=True`) on {len(sample):,} "
        f"test rows, {int(y.sum()):,} of them ransomware. Values are in log-odds of the raw model, "
        "before calibration.",
        "",
        "## Findings",
        "",
        *[f"{i}. {t}" for i, t in enumerate(findings, 1)],
        "",
        "## Global importance",
        "",
        f"![Importance](figures/explanation_importance{suffix}.png)",
        "",
        f"![Beeswarm](figures/explanation_beeswarm{suffix}.png)",
        "",
        *imp_rows,
        "",
        "## Dependence",
        "",
        f"![Round amounts](figures/explanation_dependence_round{suffix}.png)",
        "",
        f"![Dollar income](figures/explanation_dependence_income_usd{suffix}.png)",
        "",
        "## Date shortcut: Baseline A vs final model",
        "",
        f"![Side by side](figures/explanation_side_by_side{suffix}.png)",
        "",
        "## Caught vs missed test ransomware",
        "",
        "An address is caught when any of its test days crosses the final model's threshold. "
        "Medians per row:",
        "",
        *med_rows,
        "",
        *fam_rows,
        "",
    ]
    (reports / f"explanation{suffix}.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "dev": dev,
        "importance": imp.to_dict(),
        "baseline_a_importance": imp_a.to_dict(),
        "baseline_a_date_share": date_share,
        "findings": findings,
        "medians": med.to_dict(),
        "misses": miss,
        "sample_rows": len(sample),
        "run": stats,
    }
    (reports / f"explanation{suffix}.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {reports / f'explanation{suffix}.md'}")
    return payload
