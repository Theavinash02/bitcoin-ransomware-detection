# Bitcoin Ransomware Detection — Revamp Spec

Nov 9, 2026 · @Avinash

## Purpose and how to use this spec

This spec turns the FYP notebook in `bitcoin-ransomware-detection` into a reproducible, honestly evaluated ransomware detector with a public demo. Claude Code builds it one phase at a time and stops at each phase's checkpoint to report results before moving on.

**Current state:** one notebook; 19-class Random Forest + SMOTE; random 80/20 split; 98.65% accuracy but 0.115 macro F1; SHAP shows `year` is the dominant feature. `requirements.txt` lists Flask, but there is no app, script or saved model.

**Target state:** a Python package with CLI commands, a time-based evaluation, payment-behavior features, an outside test on newer ransomware (Ransomwhere), SHAP explanations, and a Gradio demo on Hugging Face Spaces.

**How to hand it to Claude Code:**

1. Export this doc as Markdown and save it in the repo as `docs/SPEC.md`.
2. Start Claude Code in the repo and say: "Read docs/SPEC.md. Do Phase 0 only, then stop at its checkpoint."
3. Review the checkpoint output, answer its questions, then ask for the next phase.

Each item is marked **Core** (must ship) or **Stretch** (only if time allows). Build all Core items first.

## Objective and success metrics

**Objective:** each day, rank active Bitcoin addresses by ransomware risk, so an analyst who reviews the top-ranked addresses catches as many ransom-payment addresses as possible, with few false alarms, including from ransomware families the model never saw.

**Primary task (Core):** binary classification, `is_ransom` = 1 for any ransomware family, 0 for `white`.

**Secondary task (Stretch):** family attribution, run only on addresses the detector flags.

| Metric | Where | Role |
| --- | --- | --- |
| Recall at FPR = 0.1% (threshold fixed on validation) | Test years | Headline number |
| PR-AUC | Test years | Main ranking quality |
| Precision in the top K per day (K set in Phase 0; mean over test days) | Test years | Analyst-budget view |
| Recall of a held-out family at the fixed threshold | Leave-one-family-out | Generalization to unseen families |
| Recall at the fixed threshold | Ransomwhere outside test | Generalization to newer ransomware |
| Brier score + reliability plot | Test years | Calibration check |

With about 1,000 benign rows per day in this dataset, FPR 0.1% means roughly one false alarm per day. Precision depends on the dataset's sampled class ratio, so it is reported but never compared with real-world rates.

Accuracy and weighted F1 are never headline numbers. They may appear only in the "old protocol" comparison row.

No target values are fixed in advance. The bar to beat is the Phase 1 baseline, and every number must come from a command that regenerates it.

## Constraints and data sources

Everything must run on an 8 GB RAM laptop, CPU only, with peak memory under about 5 GB.

**Environment:** Python 3.11, `pyproject.toml`, dependencies pinned. Core libraries: pandas, pyarrow, numpy, lightgbm, scikit-learn, imbalanced-learn (comparison only), shap, matplotlib, gradio, pyyaml, pytest, ruff. No deep-learning frameworks.

**Memory rules:**

- Load with compact types: float32 for floats, int8/int16/int32 for ints, `category` for labels and addresses.
- Convert the CSV to Parquet once; every later step reads Parquet.
- Training only: keep a random 15% of `white` rows (configurable, 10–20%). Do not reweight them: calibration on the full validation years (Phase 3) restores correct probabilities.
- Validation and test always use every row, never thinned.
- A dev mode (`--dev`) uses a 200k-row sample that keeps all ransomware rows, so every command can be tested in seconds.
- Cross-validation folds and retrains run one at a time. Set `n_jobs` from config.
- SHAP runs on at most 5,000 rows.
- Every command logs runtime and peak RSS to its report.

| Source | What it gives | Notes |
| --- | --- | --- |
| [BitcoinHeist (UCI #526)](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset) | 2.9M address-day rows, 8 features, 29 families + `white`, 2011–2018 | Main dataset. CC BY 4.0. Benign rows are capped at 1,000 per day and small transfers (0.3 BTC per UCI; confirm in the paper) were removed, so class ratios are not real-world ones. |
| [Ransomwhere (Zenodo)](https://zenodo.org/records/6512123) | Ransomware payment addresses, transactions and families; v1.1.0 published Oct 2024 | Outside test only. CC BY 4.0. Use the latest version and record it in `data/README.md`. |
| Daily BTC/USD price, 2011–2018+ | One closing price per day | Claude Code picks a free public source, records URL and license. Needed for Phase 2. |

Raw data, Parquet files and full-size models are never committed to git.

## Repo layout and commands

The notebook's logic moves into a `ransomdet` package; the notebook becomes a thin report that imports it.

```
bitcoin-ransomware-detection/
├── pyproject.toml
├── Makefile
├── configs/default.yaml        # paths, split years, sampling rate, seeds, thresholds
├── src/ransomdet/
│   ├── data.py                 # download, Parquet conversion, dtypes, dev sample
│   ├── ransomwhere.py          # load + rebuild address-day features
│   ├── split.py                # time split with address grouping
│   ├── features.py             # all feature groups, shared by training and the app
│   ├── train.py                # LightGBM + comparison models
│   ├── calibrate.py            # calibration + threshold selection
│   ├── evaluate.py             # metrics, curves, reports
│   ├── explain.py              # SHAP
│   └── cli.py
├── app/                        # Gradio demo (Hugging Face Space)
│   ├── app.py
│   └── requirements.txt
├── models/                     # gitignored except demo artifacts
├── reports/                    # metrics JSON, Markdown tables, figures
├── notebooks/
│   ├── report.ipynb            # readable walkthrough using the package
│   └── archive/fyp_original.ipynb
├── tests/
├── docs/SPEC.md, docs/MODEL_CARD.md
├── data/README.md              # sources, versions, licenses (data itself gitignored)
└── .github/workflows/ci.yml
```

| Command | Does |
| --- | --- |
| `ransomdet download` | Fetches BitcoinHeist, Ransomwhere and BTC/USD prices into `data/raw/` |
| `ransomdet prepare` | Builds Parquet files, the audit report and the splits |
| `ransomdet train --config configs/default.yaml` | Trains the detector and comparison models |
| `ransomdet evaluate` | Writes metrics and figures to `reports/` |
| `ransomdet explain` | Writes SHAP figures and tables |
| `ransomdet export-demo` | Writes the small model + sample files the app needs |
| `make all` | Runs everything in order; `make dev` does the same with `--dev` |

## Ground rules and non-goals

**Ground rules for Claude Code:**

- One phase at a time. Stop at each checkpoint with a short summary, the report path, and any open questions.
- Model and feature choices use validation years only. Test years are scored once per phase, for reporting.
- Every number in README, model card or demo is copied from a file in `reports/`, never typed by hand or estimated.
- Seeds fixed; all settings in `configs/default.yaml`; every report regenerated by a command.
- Run `--dev` first for every new command, then the full data.
- Ask before adding any dependency not listed in Constraints.
- If a result contradicts this spec's assumptions (for example, no date shortcut appears), report it plainly; do not tune until it matches.

**Non-goals:** live address lookup from a blockchain API; real-time monitoring; graph neural networks or deep learning; switching to the Elliptic++ dataset; any claim of production readiness.

## Writing style for code, comments and README

Everything a reader sees should sound like Avi wrote it: plain, specific, and short.

**Code comments and docstrings:**

- Comment why, not what. Skip any comment that only restates the line below it.
- One short line is usually enough. No banner comments or section dividers.
- Docstrings: one sentence on what the function returns; describe arguments only when they are not obvious.
- No leftover assistant phrasing ("As requested", "Here we", "Note that"). No emoji.

**README, model card and reports:**

- Write as if explaining the project to an ML engineer friend, in first person where it fits ("I split by time because…").
- Lead with the result, then how to run it. Keep the README under about 150 lines; details go in `docs/` or the report notebook.
- Avoid filler words: robust, seamless, leverage, comprehensive, cutting-edge, delve, genuinely, honest/honestly. The old README leans on "honest" heavily; say what was done instead.
- No long em-dash chains or "not X, but Y" sentences; use commas and full stops.
- No marketing tone. Every number comes from `reports/`.

## Phase 0: Data checks (Core)

Phase 0 writes `reports/phase0_data_audit.md` and decides how splitting and the outside test will work. No modeling yet.

1. **Download and convert.** Scripted download of all three sources; BitcoinHeist to Parquet with compact types. Report row count, file sizes and loaded memory.
2. **Feature definitions.** Check all 8 definitions against the BitcoinHeist paper (IJCAI 2020) and write them in `data/README.md`. The current README describes `count` and `neighbors` wrongly; UCI describes `count` as a merge measure, not a transaction count.
3. **Repeated addresses.** Count unique addresses vs rows. Report rows-per-address distribution, how many addresses span more than one year, and any address with more than one label.
4. **Ransomware by year and day.** Table of ransomware rows and unique addresses per year and per family, plus average benign and ransomware rows per day. Propose train / validation / test years so each split has enough positives (starting proposal: train 2011–2015, validation 2016, test 2017–2018), and propose K for the top-K metric.
5. **Ransomwhere schema.** Load the JSON and list its fields. Confirm every incoming transaction has an amount and a timestamp. Report date range, families and address count.
6. **Overlap.** Count Ransomwhere addresses already in BitcoinHeist. List Ransomwhere families that are not in BitcoinHeist, and how many addresses each has after removing overlaps.
7. **Price coverage.** Confirm the BTC/USD series covers every date in both datasets; report gaps.

**Checkpoint:** stop and show the audit, the proposed split years, and which features Ransomwhere can support. Wait for approval before Phase 1. (Audit also lists the proposed K.)

## Phase 1: Honest baseline (Core)

Phase 1 measures how well the original 8 features really work when tested forward in time. These are the "before" numbers.

**Split rules:**

- Use the years approved in Phase 0.
- No address may appear in more than one split. If an address has rows in several splits, keep it only in the earliest split and drop its later rows; report how many rows this removes.
- `split.py` must assert time order and zero address overlap. A test covers this.

**Runs** (all on the same split; results in `reports/phase1.md` and `reports/phase1.json`):

| Run | Features | Model |
| --- | --- | --- |
| Old protocol (reference) | 8 original, random 80/20 split | Original RF config, binary labels |
| Baseline A | 8 original incl. `year`, `day` | LightGBM, class weights |
| Baseline B | 6 original, no `year`/`day` | LightGBM, class weights |

- LightGBM uses early stopping on validation PR-AUC.
- Report every metric from the objective table, plus PR curves on one figure.
- The old-protocol row may be computed on the thinned training data if memory requires; say so in the report.

**Checkpoint:** show the table and a two-sentence reading of the gap between A and B (evidence of the date shortcut, or not).

## Phase 2: Feature engineering (Core)

Phase 2 adds features based on how ransom payments work and measures what each group adds on the validation years. Raw `year` and `day` stay out of the model; they are used only to look up the date.

| Group | Features | Idea |
| --- | --- | --- |
| A. Scale | `income_btc` = income / 1e8; log1p of income_btc, count, neighbors, length | Heavy-tailed values become usable |
| B. Ratios | income / neighbors (average incoming payment, once Phase 0 confirms what neighbors counts), count / neighbors | Shape of the payment pattern |
| C. Dollar value | `income_usd` = income_btc × that day's BTC/USD close; log1p | Ransom demands are set in dollars |
| D. Round amounts | Relative distance of income_usd to the nearest multiple of $50, $100, $500; of income_btc to 0.1, 0.5, 1 BTC; minimum of these | Victims pay a fixed demand |
| E. Same-day context | Percentile rank of income_usd, count, neighbors, length among all addresses that day | Handles activity drift without memorizing dates |

**Rules:**

- All feature code lives in `features.py` and is reused unchanged by the app.
- No feature may use labels or data from later days. Group E uses only same-day rows, which would be available at end of day in real use.
- Save per-day quantile tables for Group E (about 101 quantiles × 4 features × each day) so the app can compute percentiles for one manual input.
- Unit tests: round-amount distance on known values; percentile function on a toy frame; no label column reaches the feature matrix.

**Ablation** (validation years, Baseline B config): base 6 features, then base + each group alone, then all groups. Keep groups that improve validation PR-AUC or recall at FPR 0.1%. The test years are not used in this phase.

**Checkpoint:** show the ablation table and the chosen feature set.

## Phase 3: Final model and operating point (Core)

Phase 3 tunes the detector, calibrates it, fixes the alert threshold on validation, and only then scores the test years once.

1. **Tune** LightGBM on validation PR-AUC: a small search (at most 30 trials) over num_leaves, learning_rate, min_child_samples, feature_fraction and class weight.
2. **Calibrate** on validation with isotonic regression (sigmoid if isotonic overfits). Report the Brier score and a reliability plot.
3. **Fix two operating points on validation:** the threshold giving FPR = 0.1%, and the top-K-per-day budget from Phase 0. Save both to `models/threshold.json`.
4. **Score the test years once** with the final model, threshold and budget.
5. **Comparison rows** on the same split and features: Logistic Regression, and Random Forest + SMOTE (trained on the thinned training data).

**Saved artifacts:** LightGBM model file, calibrator, threshold JSON, feature list, per-day quantile tables, metrics JSON. The model file must be under 20 MB for the demo.

**Checkpoint:** show the final test table (all objective metrics, with Baselines A and B and the comparison rows beside it) and the PR curve.

## Phase 4: Does it catch ransomware it never saw? (Core)

Phase 4 is the project's main result: how much detection survives on families and periods absent from training.

### 4A. Leave one family out (BitcoinHeist)

- For each family with at least 200 addresses: remove all its rows from training and validation, retrain with the Phase 3 config, re-fix the FPR 0.1% threshold on validation.
- Measure recall on all of that family's rows. As a reference, give that family's test-year recall from the Phase 3 model, where it was in training (blank if it has no test-year rows).
- Report the FPR on the test years' `white` rows alongside, so the threshold is shown to stay honest.
- Output: one table, a row per held-out family.

### 4B. Outside test on Ransomwhere

1. **Portable features.** From Ransomwhere's transactions, rebuild per address and day: income received, and `neighbors` as the number of incoming transactions that day (only if Phase 0 confirms that definition). First drop single transfers below the minimum BitcoinHeist used, so both datasets are filtered alike. Add Groups A–D computed from these. Group E, `length`, `weight`, `count` and `looped` cannot be rebuilt.
2. **Portable model.** Train a separate LightGBM on BitcoinHeist using only the portable features, with the Phase 3 split, calibration and threshold procedure.
3. **Test set.** Ransomwhere addresses not in BitcoinHeist. Report three slices: families also in BitcoinHeist, families new to it, and payments after 2018.
4. **Metric.** Address-level recall: an address counts as caught if any of its days crosses the threshold. The FPR comes from BitcoinHeist's test years, since Ransomwhere has no benign addresses.
5. **Caveat in the report.** Rebuilt features follow BitcoinHeist's definitions only approximately; state this beside the numbers.

**Checkpoint:** show both tables and a short list of anything that could not be rebuilt.

## Phase 5: Explanation (Core)

Phase 5 shows which payment behaviors drive the final detector once the date shortcut is gone.

- SHAP values via LightGBM's built-in contributions (`pred_contrib=True`) on 5,000 test rows, with all ransomware rows in the sample up to half of it.
- Figures: global importance bar chart, beeswarm, and dependence plots for round-amount distance and `income_usd`.
- Side-by-side importance: Baseline A (with date) vs final model, to show the shortcut is gone.
- **Misses:** compare feature distributions of caught vs missed ransomware addresses at the fixed threshold; write 3–5 findings in plain sentences.
- Output: `reports/phase5.md` with figures.

**Checkpoint:** show the figures and the findings.

## Phase 6: Packaging, demo and README (Core)

Phase 6 makes the project runnable from a fresh clone and puts a live demo behind a link.

### Code quality

- Tests: split (time order, no address overlap), features (round amounts, percentiles, no label leakage), and a smoke test that trains and evaluates on a 5,000-row fixture in under a minute.
- GitHub Actions: ruff + pytest on the fixture, on every push.
- `make all` from a fresh clone reproduces every report on the 8 GB machine.

### Gradio demo (Hugging Face Space)

- Ships only: the model file, calibrator, threshold, feature list, per-day quantile tables, and `demo_sample.parquet` (at most 5,000 test-year rows, about 30% ransomware so examples are interesting; the page states this sample is enriched).
- **Try it tab:** a "Random address" button picks a sample row and shows its raw fields and true label; or the visitor enters raw fields (date, income in BTC, count, neighbors, length, weight, looped). Features are computed with `features.py`.
- **Output:** calibrated risk score, flagged or not at the FPR 0.1% threshold, likely family (only if the Stretch family stage exists), a SHAP bar chart of the top reasons, and right/wrong for sampled rows.
- **Results tab:** the headline table (old protocol vs new), PR curve, leave-one-family-out and Ransomwhere tables.
- **About tab:** what the model does, what it cannot do (no live address lookup), data credits.
- Keep Gradio's API enabled and show a 3-line Python and `curl` example in the About tab.

### Docs

- README rewrite: problem, headline results table copied from `reports/`, demo link and screenshot, how to reproduce, limitations (including that benign rows are a sample of 1,000 per day), data credits and licenses. Use the corrected feature definitions from Phase 0.
- `docs/MODEL_CARD.md`: data, intended use, metrics, limitations.
- `notebooks/report.ipynb` replaces the old notebook, which moves to `notebooks/archive/`.
- Remove Flask from requirements.

**Checkpoint:** share the Space link and the new README.

## Stretch goals

Start these only after every Core checkpoint is approved, in this order.

1. **Family attribution stage.** A LightGBM over the families (rare ones grouped as before), trained only on ransomware rows, applied to flagged addresses. Report macro F1 and top-3 accuracy on the test years; add the family to the demo output.
2. **Per-family SHAP.** Compare the top drivers for the 4–5 largest families.
3. **Positive-unlabeled learning.** Treat `white` as unlabeled rather than benign; compare with the Phase 3 model on the same metrics.
4. **Rolling-window evaluation.** Retrain each quarter and test on the next, to show how performance decays over time.

## Definition of done and open questions

The project is done when every box below is ticked.

- [ ] Fresh clone + `make all` reproduces every report on the 8 GB machine, peak RAM under 5 GB
- [ ] Phase 0–6 checkpoints approved
- [ ] README headline numbers match `reports/` exactly
- [ ] Leave-one-family-out and Ransomwhere tables published in README and demo
- [ ] Hugging Face Space live and linked from README
- [ ] CI green on main
- [ ] Model card written; README and code comments follow the writing-style section

**Open questions (answered in Phase 0 unless noted):**

- What exactly do `count` and `neighbors` measure in the BitcoinHeist paper? This decides which features Ransomwhere can rebuild.
- Does Ransomwhere give an amount and timestamp for every incoming transaction?
- Which split years and which K for the top-K metric?
- Which BTC/USD price source, and under what license?
- Decided: move the old notebook to `notebooks/archive/` and keep it; Avi decides later whether to delete it.
