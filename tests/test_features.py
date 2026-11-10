import numpy as np
import pandas as pd
import pytest

from ransomdet.features import (
    GROUPS,
    build_features,
    day_quantiles,
    feature_matrix,
    percentile_from_table,
    round_distance,
)


def test_round_distance_known_values():
    assert np.allclose(round_distance([500, 525, 550, 30], 50), [0, 0.5, 0, 0.4])
    assert np.allclose(round_distance([1.0, 0.95, 0.3], 0.5), [0, 0.1, 0.4])


def test_percentile_on_toy_day():
    table = np.quantile(np.arange(1, 101, dtype=float), np.linspace(0, 1, 101))
    p = percentile_from_table(np.array([0.0, 1.0, 50.5, 100.0, 500.0]), table)
    assert p[0] == 0.0 and p[-1] == 1.0
    assert abs(p[2] - 0.5) < 0.01
    # A value repeated across the table lands in the middle of its run.
    flat = np.array([1.0] * 80 + list(np.linspace(2, 10, 21)))
    assert abs(percentile_from_table(np.array([1.0]), flat)[0] - 79 / 2 / 100) < 1e-9


def toy_frame():
    dates = pd.to_datetime(["2016-03-01"] * 4 + ["2016-03-02"] * 4)
    return pd.DataFrame(
        {
            "address": list("abcdefgh"),
            "date": dates,
            "income": [1e8, 2e8, 5e7, 3e8, 1e8, 1e8, 4e8, 6e8],
            "count": [1, 2, 1, 5, 1, 1, 3, 1],
            "neighbors": [1, 2, 1, 3, 2, 1, 1, 4],
            "length": [0, 2, 8, 144, 0, 4, 2, 10],
            "weight": [1.0, 0.5, 0.2, 1.0, 1.0, 1.0, 0.3, 0.1],
            "looped": [0, 0, 0, 1, 0, 0, 0, 0],
            "label": ["white"] * 7 + ["princetonCerber"],
            "is_ransom": [0] * 7 + [1],
        }
    )


def toy_prices():
    return pd.DataFrame({"date": pd.to_datetime(["2016-03-01", "2016-03-02"]), "close_usd": [400.0, 500.0]})


def test_build_features_dollar_and_percentiles():
    df = build_features(toy_frame(), toy_prices())
    q = day_quantiles(df)
    out = build_features(toy_frame(), toy_prices(), q)
    assert out.loc[0, "income_usd"] == 400.0
    assert out.loc[0, "round_usd_100"] == 0.0
    assert out.loc[3, "pct_length"] == 1.0
    for col in GROUPS["E"]:
        assert out[col].between(0, 1).all()


def test_no_label_reaches_feature_matrix():
    df = build_features(toy_frame(), toy_prices())
    with pytest.raises(ValueError, match="label columns"):
        feature_matrix(df, ["income", "is_ransom"])
    X = feature_matrix(df, GROUPS["C"] + GROUPS["D"])
    assert not {"label", "is_ransom", "address"} & set(X.columns)
