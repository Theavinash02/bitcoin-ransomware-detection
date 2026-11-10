"""Write notebooks/report.ipynb: `python scripts/make_report_notebook.py`."""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "report.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    }


CELLS = [
    md("""
# Bitcoin ransomware detection: report

A walkthrough of the results. All the work happens in the `ransomdet` package; this notebook only
loads what `ransomdet all` wrote to `reports/` and shows it. Run `ransomdet all` (or `make all`) first.
"""),
    code("""
import json

from IPython.display import Image, Markdown, display

from ransomdet.config import ROOT, load_config

cfg = load_config()
REPORTS = ROOT / "reports"


def show(name):
    text = (REPORTS / f"{name}.md").read_text(encoding="utf-8")
    display(Markdown(text.replace("](figures/", f"]({REPORTS}/figures/")))


def load(name):
    return json.loads((REPORTS / f"{name}.json").read_text())
"""),
    md("## Data audit"),
    code('show("data_audit")'),
    md("## Forward-in-time baselines"),
    code('show("baselines")\ndisplay(Image(REPORTS / "figures" / "baselines_pr_curves.png"))'),
    md("## Feature groups"),
    code('show("feature_ablation")'),
    md("## Final model"),
    code(
        'show("final_model")\n'
        'display(Image(REPORTS / "figures" / "final_model_pr_curves.png"))\n'
        'display(Image(REPORTS / "figures" / "final_model_reliability.png"))'
    ),
    md("## Ransomware the model never saw"),
    code('show("generalization")'),
    md("## What drives the detector"),
    code(
        'show("explanation")\n'
        'for f in ("importance", "beeswarm", "dependence_round", "dependence_income_usd", "side_by_side"):\n'
        '    display(Image(REPORTS / "figures" / f"explanation_{f}.png"))'
    ),
    md("## Headline numbers"),
    code("""
p1, p3 = load("baselines"), load("final_model")
{
    "old protocol PR-AUC": p1["results"]["old_protocol"]["test"]["pr_auc"],
    "final model test PR-AUC": p3["test"]["pr_auc"],
    "final model recall at 0.1% FPR": p3["test"]["recall_at_fpr"],
    "test base rate": p3["test"]["base_rate"],
}
"""),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote notebooks/report.ipynb")
