import lightgbm as lgb
import numpy as np

from ransomdet import split


def fit_lgbm(cfg: dict, X_tr, y_tr, X_val, y_val, pos_weight_scale: float = 1.0) -> lgb.Booster:
    """Return a LightGBM detector early-stopped on validation PR-AUC."""
    p = cfg["lightgbm"]
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": p["learning_rate"],
        "num_leaves": p["num_leaves"],
        "min_child_samples": p["min_child_samples"],
        "feature_fraction": p["feature_fraction"],
        # Weight the rare class up to balance the thinned training set.
        "scale_pos_weight": pos_weight_scale * float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)),
        "num_threads": cfg["n_jobs"],
        "seed": cfg["seed"],
        "deterministic": True,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(X_tr, y_tr)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=p["n_estimators"],
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(p["early_stopping_rounds"], verbose=False)],
    )


def tuned_config(cfg: dict, params: dict) -> dict:
    """Return cfg with its lightgbm block overridden by one trial's parameters."""
    out = dict(cfg)
    out["lightgbm"] = cfg["lightgbm"] | {k: v for k, v in params.items() if k != "pos_weight_scale"}
    return out


def fit_tuned(cfg: dict, params: dict, X_tr, y_tr, X_val, y_val) -> lgb.Booster:
    return fit_lgbm(tuned_config(cfg, params), X_tr, y_tr, X_val, y_val, params.get("pos_weight_scale", 1.0))


def model_dir(cfg: dict, dev: bool, name: str):
    d = cfg["paths"]["models"] / (f"{name}_dev" if dev else name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_split_frames(cfg: dict, dev: bool, columns: list | None = None):
    """Return (thinned train, val, test) feature frames for split C."""
    from ransomdet import features

    df = features.load_feature_table(cfg, dev=dev, columns=columns)
    codes = split.load_splits(cfg, dev=dev).to_numpy()
    train = split.thin_white(df[codes == split.TRAIN], cfg["sampling"]["white_train_fraction"], cfg["seed"])
    return train, df[codes == split.VAL], df[codes == split.TEST]


def val_metrics(cfg: dict, y: np.ndarray, s: np.ndarray, day: np.ndarray) -> dict:
    from ransomdet.evaluate import detector_metrics, threshold_at_fpr

    thr = threshold_at_fpr(y, s, cfg["metrics"]["target_fpr"])
    return detector_metrics(y, s, day, thr, cfg["metrics"]["top_k"])
