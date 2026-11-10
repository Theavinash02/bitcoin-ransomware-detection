import pandas as pd

from ransomdet.audit import family_key, topk_ceiling
from ransomdet.data import add_date


def test_family_key_matches_across_datasets():
    assert family_key("princetonLocky") == family_key("Locky")
    assert family_key("montrealJigSaw") == family_key("Jigsaw")
    assert family_key("Netwalker (Mailto)") == "netwalker"


def test_add_date_uses_day_of_year():
    df = add_date(pd.DataFrame({"year": [2016, 2016], "day": [1, 366]}))
    assert list(df["date"].dt.strftime("%Y-%m-%d")) == ["2016-01-01", "2016-12-31"]


def test_topk_ceiling():
    assert topk_ceiling(pd.Series([0, 10, 3]), 5) == (0 + 5 + 3) / 15
