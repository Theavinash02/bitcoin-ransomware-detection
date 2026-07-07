# Bitcoin Ransomware Detection

A machine learning analysis that classifies Bitcoin addresses as ransomware-associated or benign
using transaction graph features.

Built as a Final Year Project (B.Tech Information Technology, Anna University, 2024).

---

## Problem Statement

Bitcoin has been widely used by ransomware operators to collect payments anonymously. This
project uses the [BitcoinHeist dataset](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset)
to classify Bitcoin addresses as belonging to one of several known ransomware families or as
benign (`white`), based on nothing but transaction graph statistics.

The label distribution is heavily skewed — 98.6% of addresses are benign, and several ransomware
families are represented by only a handful of known addresses in the entire dataset. That
imbalance, and being honest about what it does to a naive accuracy metric, is the central
methodological thread of this project.

---

## Dataset

**BitcoinHeist Ransomware Address Dataset** — UCI Machine Learning Repository
*Farrukhjon Masudov et al. — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)*

| Feature | Type | Description |
|---|---|---|
| `year` | int | Year of transaction |
| `day` | int | Day of year (1–365) |
| `length` | int | Length of the transaction chain |
| `weight` | float | Fraction of BTC received from a single source |
| `count` | int | Number of transactions at the address |
| `looped` | binary | Whether coins were sent back to the source |
| `neighbors` | int | Number of distinct addresses transacted with |
| `income` | float | Total BTC income received |

**Target:** 29 ransomware family labels + `white` (benign)
**Dataset size:** 2,916,697 addresses

---

## Methodology

1. **Rare-class handling:** 11 of the 29 ransomware families have fewer than 10 total samples in
   the entire dataset. These are grouped into a single `rare_family` bucket before splitting,
   rather than silently dropped — a documented, honest trade-off in resolution.
2. **Stratified split:** an 80/20 train/test split, stratified by label, so every class keeps its
   proportional representation in both splits — important on data this imbalanced.
3. **Baseline model comparison:** Logistic Regression, Decision Tree, and Random Forest are
   compared using 5-fold stratified cross-validation on the training set (accuracy, weighted F1,
   macro F1) before committing to a final model.
4. **SMOTE, leakage-safe:** the chosen model (Random Forest) is combined with SMOTE inside a
   single pipeline, so oversampling is refit independently within each cross-validation fold's
   training portion — never applied once to the whole training set beforehand, and never applied
   to the test set.
5. **Final evaluation:** the SMOTE + Random Forest pipeline is refit once on the full training set
   and evaluated on the held-out, untouched test set.
6. **SHAP interpretation:** `TreeExplainer` on the final Random Forest, computed on a
   representative 8,000-row sample, explaining which features actually drive each prediction.

---

## Results

**Baseline comparison** (5-fold stratified cross-validation on the training set, no resampling):

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|
| Logistic Regression | 0.9858 ± 0.0000 | 0.9787 ± 0.0000 | 0.0523 ± 0.0000 |
| Decision Tree | 0.9823 ± 0.0002 | 0.9827 ± 0.0001 | 0.2209 ± 0.0160 |
| Random Forest | 0.9867 ± 0.0001 | 0.9808 ± 0.0001 | 0.1044 ± 0.0021 |

Decision Tree edges out Random Forest on weighted and macro F1 point estimates here, but with
roughly 8x the fold-to-fold variance — an unstable, fold-dependent edge rather than a reliable
one. Random Forest is carried forward for its stability and because its ensemble structure is
what makes SHAP's `TreeExplainer` straightforward to apply cleanly.

**Final model** — SMOTE + Random Forest, evaluated on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 98.65% |
| Weighted F1 | 0.9803 |
| Macro F1 | 0.1150 |

(Cross-validation on the training set gave a consistent 0.1246 ± 0.0145 macro F1, so this isn't a
single lucky or unlucky draw.)

The Random Forest's trees are capped (`max_depth=20`, `min_samples_leaf=25`) so that SHAP's
`TreeExplainer` can run without exhausting memory. That constraint has a real, measurable cost:
several ransomware families with thousands of test samples and high precision (e.g.
`montrealCryptoLocker`, `paduaCryptoWall`, `princetonLocky`) collapse to near-zero recall — the
model is usually right when it does flag them, but a shallower forest can route far fewer test
cases to them in the first place. SMOTE helps the model learn these classes exist; it can't fully
offset the depth constraint. This is reported honestly as a trade-off between interpretability
and raw recall on rare classes, not as a claim that the imbalance correction "didn't work."

**SHAP interpretation:** `year` is the single dominant feature for both the benign (`white`) and
the largest ransomware family (`paduaCryptoWall`) classes, by a wide margin over every other
feature. That's a genuinely useful but slightly uncomfortable finding: `count` and `neighbors` —
the features that, per the payment-consolidation intuition, should be the clearest ransomware
signal — rank behind purely temporal features for `paduaCryptoWall`. The likely explanation is
that this was a real campaign active within a narrow window of the 2011–2018 dataset, so the
model may be partly keying on *when* an address was active rather than a transaction-shape
fingerprint that would generalize beyond this dataset — see the notebook's Discussion section for
the full discussion of this and other limitations.

---

## Project Structure

```
├── bitcoin_ransomware.ipynb    # Full analysis: EDA, preprocessing, modeling, SHAP
├── requirements.txt
└── .gitignore
```

---

## Reproducing This Analysis

```bash
# 1. Clone the repo
git clone https://github.com/theavinash02/bitcoin-ransomware-detection.git
cd bitcoin-ransomware-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Get BitcoinHeistData.csv from:
# https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
# Place it in the project root folder

# 4. Run the notebook end-to-end
jupyter notebook bitcoin_ransomware.ipynb
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML models | scikit-learn (Logistic Regression, Decision Tree, Random Forest) |
| Class imbalance | imbalanced-learn (SMOTE) |
| Interpretability | SHAP |
| Data processing | pandas, numpy |
| Visualisation | seaborn, matplotlib |

---

## Limitations

- This is a research/learning project on a public benchmark dataset covering 2011–2018 Bitcoin
  activity — not a live monitoring feed. Nothing here claims real-time ransomware detection or
  production readiness.
- Grouping 11 ultra-rare families into a single `rare_family` bucket is a real loss of
  resolution — those families remain indistinguishable from each other in this analysis.
- SMOTE interpolates within existing minority-class structure; it cannot manufacture genuinely
  new information about families that had only a handful of real addresses to begin with.
- SHAP explains association and feature attribution learned by this specific model — it is not a
  causal account of *why* ransomware addresses behave the way they do.

---

## Dataset Credit

> Farrukhjon Masudov, Taha Belkhouja, Yan Shi, Janardhan Rao Doppa, Haifeng Chen, Srikanth V. Krishnamurthy.
> *BitcoinHeist: Topological Data Analysis for Ransomware Detection on the Bitcoin Blockchain.*
> IJCAI 2020. [UCI Repository](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) — CC BY 4.0.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
