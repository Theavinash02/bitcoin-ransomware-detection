import numpy as np

from ransomdet.evaluate import rates_at, threshold_at_fpr, topk_per_day


def test_threshold_at_fpr_flags_at_most_target_share_of_negatives():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(10_000), np.ones(100)].astype(int)
    s = np.r_[rng.random(10_000), rng.random(100) + 0.5]
    t = threshold_at_fpr(y, s, 0.001)
    assert rates_at(y, s, t)["fpr"] <= 0.001
    # One step lower would flag more than the target.
    assert (s[y == 0] >= t).mean() > 0.001


def test_topk_pooled_precision_and_ceiling():
    y = np.array([1, 0, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6])
    d = np.array([1, 1, 1, 2, 2, 2])
    out = topk_per_day(y, s, d, k=2)
    # Day 1 top two: 1 hit. Day 2 top two (0.7, 0.6): 1 hit.
    assert out["precision_at_k"] == 2 / 4
    # Day 1 has one positive, day 2 has two.
    assert out["ceiling"] == 3 / 4
