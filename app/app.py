"""Gradio demo: score a BitcoinHeist address-day for ransomware risk."""

import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import gradio as gr  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ransomdet import features  # noqa: E402
from ransomdet.calibrate import Calibrator  # noqa: E402

ART = HERE / os.environ.get("RANSOMDET_ARTIFACTS", "artifacts")

booster = lgb.Booster(model_file=str(ART / "model.txt"))
calibrator = Calibrator.load(ART / "calibrator.json")
threshold = json.loads((ART / "threshold.json").read_text())
FEATS = json.loads((ART / "features.json").read_text())
quantiles = pd.read_parquet(ART / "day_quantiles.parquet")
prices = pd.read_parquet(ART / "btc_usd.parquet")
sample = pd.read_parquet(ART / "demo_sample.parquet")
results = json.loads((ART / "results.json").read_text())
FIRST_DAY, LAST_DAY = quantiles["date"].min().date(), quantiles["date"].max().date()

# Free Gradio Spaces run on ZeroGPU, which requires a @spaces.GPU function. The model runs on
# CPU either way; outside a Space the package is absent and the decorator does nothing.
try:
    import spaces

    on_gpu = spaces.GPU(duration=10)
except ImportError:

    def on_gpu(fn):
        return fn


@on_gpu
def score(row: pd.DataFrame) -> tuple[float, bool, pd.Series]:
    """Return calibrated risk, whether it is flagged, and the SHAP contributions for one row."""
    feats = features.build_features(row, prices, quantiles)
    X = features.feature_matrix(feats, FEATS)
    raw = float(booster.predict(X)[0])
    contrib = booster.predict(X, pred_contrib=True)[0, :-1]
    prob = float(calibrator.predict(np.array([raw]))[0])
    return prob, raw > threshold["raw_threshold"], pd.Series(contrib, index=FEATS)


def shap_figure(contrib: pd.Series):
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(top.index, top.values, color=["#c0392b" if v > 0 else "#2e86c1" for v in top.values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Push toward ransomware (log-odds)")
    ax.set_title("Top reasons for this score", fontsize=10)
    fig.tight_layout()
    return fig


def verdict_text(prob: float, flagged: bool, truth: str | None = None) -> str:
    lines = [
        f"**Risk score:** {prob:.3f} (calibrated on Jul-Dec 2016)",
        f"**Flagged at the 0.1% FPR threshold:** {'yes' if flagged else 'no'} "
        f"(the threshold sits at a risk score of {threshold['calibrated_at_threshold']:.3f})",
    ]
    if truth is not None:
        is_ransom = truth != "white"
        right = flagged == is_ransom
        lines.append(f"**True label:** {truth} ({'right' if right else 'wrong'})")
    return "\n\n".join(lines)


def make_row(date, income_btc, count, neighbors, length, weight, looped) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp(date).normalize()],
            "income": [round(float(income_btc) * 1e8)],
            "count": [int(count)],
            "neighbors": [int(neighbors)],
            "length": [int(length)],
            "weight": [float(weight)],
            "looped": [int(looped)],
        }
    )


def random_address(seed: float | None = None):
    r = sample.sample(n=1, random_state=None if seed is None else int(seed)).iloc[0]
    row = make_row(
        r["date"], r["income"] / 1e8, r["count"], r["neighbors"], r["length"], r["weight"], r["looped"]
    )
    prob, flagged, contrib = score(row)
    fields = (
        f"`{r['address']}` on {pd.Timestamp(r['date']).date()}: income {r['income'] / 1e8:g} BTC, "
        f"count {r['count']}, neighbors {r['neighbors']}, length {r['length']}, "
        f"weight {r['weight']:.4g}, looped {r['looped']}"
    )
    return fields, verdict_text(prob, flagged, r["label"]), shap_figure(contrib)


def manual(date, income_btc, count, neighbors, length, weight, looped):
    try:
        day = pd.Timestamp(date).date()
    except (ValueError, TypeError):
        raise gr.Error("Enter the date as YYYY-MM-DD.") from None
    if not FIRST_DAY <= day <= LAST_DAY:
        raise gr.Error(f"Dates must be between {FIRST_DAY} and {LAST_DAY}.")
    if income_btc < 0.3:
        raise gr.Error("BitcoinHeist only keeps transfers of at least 0.3 BTC.")
    prob, flagged, contrib = score(make_row(day, income_btc, count, neighbors, length, weight, looped))
    return verdict_text(prob, flagged), shap_figure(contrib)


def table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(3)
    return df


ABOUT = f"""
This model ranks Bitcoin address-days by ransomware risk. It was trained on the
[BitcoinHeist](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) dataset
(2011-2018) with a time split: train before July 2016, validate on July-December 2016, test on 2017-18.

**What it uses:** the BitcoinHeist graph features for one address on one day, the dollar value of the
income at that day's BTC price, how close the amount is to round dollar and BTC figures, and where the
address ranks among all addresses seen that day.

**What it cannot do:** look up a live address. It only scores the fields BitcoinHeist provides, for dates
between {FIRST_DAY} and {LAST_DAY}. The benign side of the data is a sample of 1,000 addresses per day, so
precision here says nothing about real-world alert rates. Detection of ransomware families it never saw
is weak; see the Results tab.

**The sample** behind "Random address" is enriched to about 30% ransomware so the examples are
interesting. The real test years are {results["test_base_rate"]:.1%} ransomware.

**API.** The Gradio API is on. From Python:

```python
from gradio_client import Client
client = Client("<your-space-id>")
print(client.predict("2017-03-01", 1.0, 1, 2, 10, 0.5, 0, api_name="/manual"))
```

With curl:

```bash
curl -X POST <space-url>/gradio_api/call/manual -H "Content-Type: application/json" \\
  -d '{{"data": ["2017-03-01", 1.0, 1, 2, 10, 0.5, 0]}}'
```

**Data credits.** BitcoinHeist by Akcora, Li, Gel and Kantarcioglu (CC BY 4.0). Ransomwhere by Cable et
al. (CC BY 4.0). BTC/USD prices from the Coin Metrics community API (CC BY-NC 4.0).
"""

with gr.Blocks(title="Bitcoin ransomware detector") as demo:
    gr.Markdown("# Bitcoin ransomware detector\nScores one address-day from the BitcoinHeist dataset.")
    with gr.Tab("Try it"):
        with gr.Row():
            btn = gr.Button("Random address", variant="primary")
        fields = gr.Markdown()
        with gr.Row():
            verdict = gr.Markdown()
            plot = gr.Plot()
        btn.click(random_address, outputs=[fields, verdict, plot], api_name="random")

        gr.Markdown("### Or enter the raw fields")
        with gr.Row():
            date = gr.Textbox(label="Date (YYYY-MM-DD)", value="2017-03-01")
            income = gr.Number(label="Income (BTC)", value=1.0)
            count = gr.Number(label="count", value=1, precision=0)
            neighbors = gr.Number(label="neighbors", value=2, precision=0)
        with gr.Row():
            length = gr.Number(label="length", value=10, precision=0)
            weight = gr.Number(label="weight", value=0.5)
            looped = gr.Number(label="looped", value=0, precision=0)
            go = gr.Button("Score")
        with gr.Row():
            verdict2 = gr.Markdown()
            plot2 = gr.Plot()
        go.click(
            manual,
            inputs=[date, income, count, neighbors, length, weight, looped],
            outputs=[verdict2, plot2],
            api_name="manual",
        )
    with gr.Tab("Results"):
        gr.Markdown(
            "All numbers are copied from the project reports. Test years are 2017-18; "
            f"{results['test_base_rate']:.2%} of test rows are ransomware."
        )
        gr.Dataframe(table(results["headline"]), label="Old protocol vs time split")
        gr.Image(str(ART / "figures" / "final_model_pr_curves.png"), label="Test precision-recall")
        gr.Dataframe(table(results["lofo"]), label="Leave one family out")
        gr.Dataframe(table(results["ransomwhere"]), label="Ransomwhere outside test")
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
