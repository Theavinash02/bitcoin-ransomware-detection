# Model card: BitcoinHeist ransomware detector

## What it is

A LightGBM binary classifier that scores one Bitcoin address on one day for how likely it is to be
a ransomware payment address. The output is a probability calibrated on July to December 2016 (a
sigmoid fit) and a flag at the threshold that gave a 0.1% false-positive rate on that period.
Settings, feature list and threshold are in `models/final/` and copied into `app/artifacts/`.

## Data

- Training and evaluation: [BitcoinHeist](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset),
  2,916,697 address-day rows from 2011 to 2018, 41,413 of them labeled with one of 28 ransomware
  families, the rest `white`. Benign rows are capped at 1,000 per day and transfers under 0.3 BTC
  are removed before the features are computed.
- Split: train before 2016-07-01, validation July to December 2016, test 2017-18. An address that
  appears in more than one period is kept only in the earliest one. Training keeps 15% of white rows.
- Outside test: [Ransomwhere](https://zenodo.org/records/13999026) v1.1.0, bitcoin addresses not in
  BitcoinHeist.
- Prices: Coin Metrics community daily `PriceUSD`, joined on the BitcoinHeist day.

## Inputs

21 features: the six BitcoinHeist graph features (`length`, `weight`, `count`, `looped`,
`neighbors`, `income`), average payment size and count per neighbor, income in dollars, distance of
the dollar and BTC income to round amounts, and the address's percentile among all addresses on the
same day for dollar income, count, neighbors and length. `year` and `day` are not inputs. Feature
definitions are in [data/README.md](../data/README.md) and the code in `src/ransomdet/features.py`.

## Intended use

A portfolio and research project about evaluating ransomware detectors forward in time. It is meant
to be read with its reports, not used to make decisions about real addresses.

Out of scope: live address lookup, real-time monitoring, any decision about a real person or
service, and any claim of production readiness.

## Metrics

From [reports/final_model.md](../reports/final_model.md) and [reports/generalization.md](../reports/generalization.md). Test is
2017-18, where 0.49% of rows are ransomware.

| Metric | Validation (Jul-Dec 2016) | Test (2017-18) |
| --- | --- | --- |
| PR-AUC | 0.325 | 0.015 |
| Recall at the 0.1% FPR threshold | 0.056 | 0.001 |
| False-positive rate at that threshold | 0.10% | 0.04% |
| Brier score (base-rate Brier in brackets) | 0.0292 (0.0354) | 0.0069 (0.0049) |

Leave-one-family-out recall for the five families with at least 200 addresses is between 0.000 and
0.002. On Ransomwhere, a model limited to the rebuildable features finds 45% of unseen addresses from
families BitcoinHeist already has, and none from new families or from payments after 2018.

## Limitations

- Forward in time the model is weak. Validation PR-AUC is 0.325; test PR-AUC is 0.015, three times
  the base rate. 93% of test ransomware is Cerber, so the test result is mostly about one family in
  one year.
- It does not generalize to unseen families in either test.
- Benign rows are a sample of 1,000 per day, so precision and alert counts do not transfer to the
  real network.
- `white` means "not on the three source lists". Some of those addresses may be ransomware.
- Probabilities are calibrated where ransomware was 3.7% of rows; on the test years they overstate
  risk.
- Ransomwhere features are rebuilt from attributed payments and match BitcoinHeist's definitions only
  approximately.

## Ethical notes

Bitcoin addresses are pseudonymous, not anonymous. A false flag on a real address could harm the
person behind it, which is one more reason this model should not be used on live data.
