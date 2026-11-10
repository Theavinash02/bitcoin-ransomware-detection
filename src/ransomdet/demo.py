import json
import shutil
from pathlib import Path

import pandas as pd

from ransomdet import data, features
from ransomdet.config import ROOT
from ransomdet.final_model import FINAL
from ransomdet.reporting import dev_suffix
from ransomdet.train import load_split_frames, model_dir

APP_DIR = ROOT / "app"
RAW_FIELDS = ["date", "income", "count", "neighbors", "length", "weight", "looped"]


def demo_sample(test: pd.DataFrame, n: int, ransom_share: float, seed: int) -> pd.DataFrame:
    """Return at most n test rows, enriched to about ransom_share ransomware."""
    ransom = test[test["is_ransom"] == 1]
    n_r = min(len(ransom), int(n * ransom_share))
    white = test[test["is_ransom"] == 0].sample(n=n - n_r, random_state=seed)
    out = pd.concat([ransom.sample(n=n_r, random_state=seed), white]).sample(frac=1, random_state=seed)
    out = out[["address", *RAW_FIELDS, "label", "is_ransom"]].copy()
    out["address"] = out["address"].astype(str)
    out["label"] = out["label"].astype(str)
    return out.reset_index(drop=True)


def results_summary(reports: Path, suffix: str) -> dict:
    """Return the tables the Results tab shows, copied from the reports."""
    p1 = json.loads((reports / f"baselines{suffix}.json").read_text())
    p3 = json.loads((reports / f"final_model{suffix}.json").read_text())
    p4 = json.loads((reports / f"generalization{suffix}.json").read_text())
    headline = [
        {
            "Run": "Old protocol (random 80/20, RF)",
            "Test PR-AUC": p1["results"]["old_protocol"]["test"]["pr_auc"],
            "Recall at FPR 0.1%": p1["results"]["old_protocol"]["test"]["recall_at_fpr"],
            "Note": "random split, same addresses on both sides",
        },
        {
            "Run": "Baseline B (6 original features, time split)",
            "Test PR-AUC": p1["results"]["baseline_b"]["test"]["pr_auc"],
            "Recall at FPR 0.1%": p1["results"]["baseline_b"]["test"]["recall_at_fpr"],
            "Note": "trained before Jul 2016, tested on 2017-18",
        },
        {
            "Run": "Final LightGBM (time split)",
            "Test PR-AUC": p3["test"]["pr_auc"],
            "Recall at FPR 0.1%": p3["test"]["recall_at_fpr"],
            "Note": f"{len(p3['features'])} features, threshold fixed on validation",
        },
    ]
    lofo = [
        {
            "Held-out family": r["family"],
            "Addresses": r["addresses"],
            "Recall held out": r["held_out_recall"],
            "Test white FPR (%)": round(r["test_white_fpr"] * 100, 2),
        }
        for r in p4["lofo"]
    ]
    rw = [
        {
            "Slice": s["slice"].strip(),
            "Addresses": s["addresses"],
            "Address recall": s["address_recall"],
        }
        for s in p4["ransomwhere"]["slices"]
    ]
    return {
        "headline": headline,
        "test_base_rate": p3["test"]["base_rate"],
        "lofo": lofo,
        "ransomwhere": rw,
    }


def export_demo(cfg: dict, dev: bool = False) -> Path:
    """Copy the model files, a small enriched sample and the results into app/artifacts/."""
    suffix = dev_suffix(dev)
    out = APP_DIR / ("artifacts_dev" if dev else "artifacts")
    out.mkdir(parents=True, exist_ok=True)
    mdir = model_dir(cfg, dev, FINAL)
    for name in (
        "model.txt",
        "calibrator.json",
        "threshold.json",
        "features.json",
        features.QUANTILES_PARQUET,
    ):
        shutil.copy(mdir / name, out / name)
    shutil.copy(cfg["paths"]["processed"] / data.PRICE_PARQUET, out / data.PRICE_PARQUET)

    _, _, test = load_split_frames(cfg, dev)
    d = cfg["demo"]
    demo_sample(test, d["rows"], d["ransom_share"], cfg["seed"]).to_parquet(
        out / "demo_sample.parquet", index=False
    )

    reports = cfg["paths"]["reports"]
    (out / "results.json").write_text(json.dumps(results_summary(reports, suffix), indent=2))
    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    for name in ("final_model_pr_curves", "explanation_side_by_side"):
        shutil.copy(reports / "figures" / f"{name}{suffix}.png", figs / f"{name}.png")

    # The Space has no access to the repo, so it gets its own copy of the feature code.
    pkg = APP_DIR / "ransomdet"
    pkg.mkdir(exist_ok=True)
    src = Path(features.__file__).parent
    for name in ("features.py", "calibrate.py"):
        shutil.copy(src / name, pkg / name)
    (pkg / "__init__.py").write_text(
        '"""Feature code copied from the main repo by `ransomdet export-demo`."""\n'
    )

    sizes = {p.name: round(p.stat().st_size / 2**20, 2) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps({"dev": dev, "sizes_mb": sizes}, indent=2))
    print(f"wrote {out} ({sum(sizes.values()):.1f} MB)")
    if not dev:
        from ransomdet.readme import update_readme

        if update_readme(ROOT / "README.md", reports):
            print("updated the results block in README.md")
    return out
