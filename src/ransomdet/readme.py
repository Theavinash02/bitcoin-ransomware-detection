"""The results block in README.md, generated from reports/*.json so the numbers never drift."""

import json
from pathlib import Path

START, END = "<!-- results:start -->", "<!-- results:end -->"


def _load(reports: Path, name: str) -> dict:
    return json.loads((reports / f"{name}.json").read_text())


def results_block(reports: Path) -> str:
    """Return the README results tables as Markdown."""
    p1, p3, p4 = (_load(reports, n) for n in ("baselines", "final_model", "generalization"))
    old = p1["results"]["old_protocol"]["test"]
    b = p1["results"]["baseline_b"]["test"]
    f = p3["test"]
    lines = [
        "| Setup | Test PR-AUC | Recall at 0.1% FPR | Accuracy |",
        "| --- | --- | --- | --- |",
        f"| FYP protocol: random 80/20 split, Random Forest | {old['pr_auc']:.3f} | "
        f"{old['recall_at_fpr']:.3f} | {old['accuracy']:.4f} |",
        f"| Time split, 6 original features (Baseline B) | {b['pr_auc']:.3f} | "
        f"{b['recall_at_fpr']:.3f} | - |",
        f"| Time split, final LightGBM, {len(p3['features'])} features | {f['pr_auc']:.3f} | "
        f"{f['recall_at_fpr']:.3f} | - |",
        "",
        f"Time split: train before {p1['config']['split']['val_start']}, validate until "
        f"{p1['config']['split']['test_start']}, test on 2017-18. The test base rate is "
        f"{f['base_rate']:.4f}, which is what a random ranker scores as PR-AUC. On validation the final "
        f"model reaches PR-AUC {p3['val']['pr_auc']:.3f}.",
        "",
        "Leave one family out (recall on the held-out family, threshold re-fixed at 0.1% FPR):",
        "",
        "| Held-out family | Addresses | Recall | Test white FPR |",
        "| --- | --- | --- | --- |",
        *[
            f"| {r['family']} | {r['addresses']:,} | {r['held_out_recall']:.3f} | "
            f"{r['test_white_fpr'] * 100:.2f}% |"
            for r in p4["lofo"]
        ],
        "",
        "Ransomwhere outside test (addresses not in BitcoinHeist, portable features only):",
        "",
        "| Slice | Addresses | Address recall |",
        "| --- | --- | --- |",
        *[
            f"| {s['slice'].strip()} | {s['addresses']:,} | {s['address_recall']:.3f} |"
            for s in p4["ransomwhere"]["slices"]
        ],
    ]
    return "\n".join(lines)


def update_readme(readme: Path, reports: Path) -> bool:
    """Rewrite the block between the markers in README.md; return True if it changed."""
    text = readme.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{results_block(reports)}\n{END}{tail}"
    if new != text:
        readme.write_text(new, encoding="utf-8")
    return new != text


def readme_block(readme: Path) -> str:
    text = readme.read_text(encoding="utf-8")
    return text.split(START, 1)[1].split(END, 1)[0].strip()
