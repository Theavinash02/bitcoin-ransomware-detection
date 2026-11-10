"""Train and evaluate end to end on the 5,000-row fixture, with no access to data/."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ransomdet import features, split
from ransomdet.calibrate import Calibrator
from ransomdet.data import BH_DTYPES, add_date
from ransomdet.evaluate import detector_metrics, threshold_at_fpr
from ransomdet.train import fit_lgbm

FIX = Path(__file__).parent / "fixtures"
CFG = {
    "seed": 0,
    "n_jobs": 2,
    "lightgbm": {
        "n_estimators": 200,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 10,
        "feature_fraction": 0.9,
        "early_stopping_rounds": 20,
    },
}


def load_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIX / "bitcoinheist_5k.csv", dtype=BH_DTYPES)
    df["income"] = df["income"].astype("int64")
    df["is_ransom"] = (df["label"] != "white").astype("int8")
    return add_date(df)


def test_train_and_evaluate_on_fixture():
    start = time.perf_counter()
    df = load_fixture()
    assert len(df) == 5000
    prices = pd.read_csv(FIX / "btc_usd_5k.csv", parse_dates=["date"])

    base = features.build_features(df, prices)
    df = features.build_features(df, prices, features.day_quantiles(base))
    codes = split.assign_split(df, "2016-07-01", "2017-01-01")
    split.check_split(df, codes)
    train = split.thin_white(df[codes == split.TRAIN], 0.5, seed=0)
    val, test = df[codes == split.VAL], df[codes == split.TEST]

    feats = features.BASE + features.GROUPS["C"] + features.GROUPS["D"] + features.GROUPS["E"]
    X = {k: features.feature_matrix(d, feats) for k, d in (("tr", train), ("val", val), ("te", test))}
    booster = fit_lgbm(CFG, X["tr"], train["is_ransom"], X["val"], val["is_ransom"])
    s_val, s_te = booster.predict(X["val"]), booster.predict(X["te"])
    y_val, y_te = val["is_ransom"].to_numpy(), test["is_ransom"].to_numpy()

    thr = threshold_at_fpr(y_val, s_val, 0.01)
    m = detector_metrics(y_te, s_te, test["date"].to_numpy(), thr, k=10)
    assert 0 <= m["pr_auc"] <= 1 and m["positives"] > 0
    # A model that learned nothing would sit near the base rate.
    assert m["pr_auc"] > m["base_rate"]

    cal = Calibrator("sigmoid").fit(s_val, y_val)
    p = cal.predict(s_te)
    assert np.all((p >= 0) & (p <= 1))
    assert time.perf_counter() - start < 60
