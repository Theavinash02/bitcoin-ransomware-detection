import pandas as pd
import pytest

from ransomdet.split import DROPPED, TEST, TRAIN, VAL, assign_split, check_split, thin_white


def toy():
    return pd.DataFrame(
        {
            "address": ["a", "a", "b", "c", "b"],
            "date": pd.to_datetime(["2015-05-01", "2017-02-01", "2016-08-01", "2017-03-01", "2018-01-05"]),
            "is_ransom": [1, 1, 0, 0, 0],
        }
    )


def test_address_kept_only_in_earliest_split():
    split = assign_split(toy(), "2016-07-01", "2017-01-01")
    assert list(split) == [TRAIN, DROPPED, VAL, TEST, DROPPED]


def test_check_split_passes_on_assigned_split():
    df = toy()
    check_split(df, assign_split(df, "2016-07-01", "2017-01-01"))


def test_check_split_catches_shared_address():
    df = toy()
    raw = pd.Series([TRAIN, TEST, VAL, TEST, TEST], dtype="int8")
    with pytest.raises(AssertionError, match="appear in an earlier split"):
        check_split(df, raw)


def test_check_split_catches_time_order():
    df = toy()
    raw = pd.Series([VAL, DROPPED, TRAIN, TEST, DROPPED], dtype="int8")
    with pytest.raises(AssertionError, match="not entirely before"):
        check_split(df, raw)


def test_thin_white_keeps_every_ransom_row():
    df = pd.DataFrame({"is_ransom": [1] * 50 + [0] * 10_000})
    out = thin_white(df, 0.15, seed=0)
    assert (out["is_ransom"] == 1).sum() == 50
    assert 1_300 < (out["is_ransom"] == 0).sum() < 1_700
