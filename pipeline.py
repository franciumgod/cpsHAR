import math
from pathlib import Path

import numpy as np

from models.LGBM import LightGBMClassifierSK
from models.XGB import XGBoostClassifierSK


def resolve_dataset_file(data_arg):
    if data_arg is None:
        return "cps_data_multi_label.pkl"

    text = str(data_arg).strip().lower()
    if text in {"", "raw", "origin", "original", "none"}:
        return "cps_data_multi_label.pkl"

    if text.isdigit():
        step = int(text)
        return f"cps_windows_2s_2000hz_step_{step}.pkl"

    return str(data_arg).strip()


def parse_bool_arg(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_augment_count(value, default_when_true=1):
    if value is None:
        return 0

    if isinstance(value, bool):
        return default_when_true if value else 0

    text = str(value).strip().lower()
    if text in {"", "0", "false", "no", "n", "off", "none", "null"}:
        return 0
    if text in {"1", "true", "yes", "y", "on"}:
        return max(1, int(default_when_true))

    try:
        parsed = int(text)
    except ValueError:
        return 0
    return max(0, parsed)


def parse_fold_list(text):
    if not text:
        return [1, 2, 3, 4]
    fold_ids = []
    for part in str(text).split(","):
        value = int(part.strip())
        if value < 1 or value > 4:
            raise ValueError(f"Fold id must be in [1, 4], but got {value}.")
        fold_ids.append(value)
    return fold_ids


def sample_windows_for_timeline(X, max_windows):
    if X is None:
        return None

    X = np.asarray(X)
    n = len(X)
    if n == 0 or max_windows <= 0 or n <= max_windows:
        return X

    idx = np.linspace(0, n - 1, max_windows, dtype=int)
    return X[idx]


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    return obj


def build_special_signal_combo_map(sensor_cols):
    combo_by_class = {
        "Driving(curve)": ["Acc.x", "Acc.z", "Gyro.norm"],
        "Driving(straight)": ["Acc.y", "Acc.z", "Gyro.y", "Gyro.z", "Acc.norm", "Gyro.norm"],
        "Lifting(lowering)": ["Acc.z", "Gyro.x", "Baro.x"],
        "Lifting(raising)": ["Acc.x", "Acc.z", "Gyro.x", "Baro.x", "Acc.norm", "Gyro.norm"],
        "Stationary processes": ["Acc.y", "Gyro.x", "Gyro.z", "Baro.x", "Acc.norm", "Gyro.norm"],
        "Turntable wrapping": ["Acc.z", "Gyro.z", "Gyro.norm"],
    }
    sensor_idx = {name: idx for idx, name in enumerate(sensor_cols)}
    combo_idx = {}
    for class_name, names in combo_by_class.items():
        idxs = [sensor_idx[name] for name in names if name in sensor_idx]
        if idxs:
            combo_idx[class_name] = idxs
    return combo_idx


def build_model(
    args,
    target_vals,
    config,
    preprocess_order,
    respect_preprocess_order_for_model_subsample=False,
):
    if respect_preprocess_order_for_model_subsample:
        model_subsample_method = (
            "false" if preprocess_order == "subsample_first" else args.subsample_method
        )
        model_subsample_n = (
            args.train_sample_num if preprocess_order == "downsample_first" else 10 ** 12
        )
    else:
        model_subsample_method = "false"
        model_subsample_n = 10 ** 12

    model_downsample_method = "false"
    use_signal_combo = parse_bool_arg(getattr(args, "signal_combo", False), default=False)
    augment_count = parse_augment_count(getattr(args, "sample_augment", False))
    augment_target = str(getattr(args, "augment_target", "multilabel")).strip()
    augment_method = str(getattr(args, "augment_method", "jitter")).strip().lower()
    tta_count = parse_augment_count(getattr(args, "tta", False))
    tta_method = str(getattr(args, "tta_method", "jitter")).strip().lower()
    signal_combo_map = (
        build_special_signal_combo_map(config.data.sensor_cols) if use_signal_combo else None
    )

    model_name = str(args.model).lower() if args.model else "xgboost"
    if model_name == "lightgbm":
        return LightGBMClassifierSK(
            target_vals,
            use_val_in_train=parse_bool_arg(args.train_with_val, default=False),
            subsample_method=model_subsample_method,
            n_train_data_samples=model_subsample_n,
            boosting_type=getattr(args, "boosting_type", "lgbt"),
            max_depth=getattr(args, "max_depth", 6),
            num_leaves=getattr(args, "num_leaves", 63),
            colsample_bytree=getattr(args, "colsample_bytree", 0.8),
            n_estimators=getattr(args, "n_estimators", 1000),
            learning_rate=getattr(args, "learning_rate", 0.05),
            subsample=getattr(args, "subsample", 0.8),
            reg_alpha=getattr(args, "reg_alpha", 0.0),
            reg_lambda=getattr(args, "reg_lambda", 0.0),
            drop_rate=getattr(args, "drop_rate", 0.1),
            max_drop=getattr(args, "max_drop", None),
            skip_drop=getattr(args, "skip_drop", None),
            min_child_samples=getattr(args, "min_child_samples", 30),
            min_data_in_leaf=getattr(args, "min_data_in_leaf", 10),
            feature_batch_size=getattr(args, "feature_batch_size", 2000),
            pre_downsample_method=model_downsample_method,
            pre_downsample_factor=config.prep.ds_factor,
            signal_combo_map=signal_combo_map,
            augment_samples=augment_count,
            augment_target=augment_target,
            augment_method=augment_method,
            mixup_alpha=getattr(args, "mixup_alpha", 0.4),
            cutmix_ratio=getattr(args, "cutmix_ratio", 0.3),
            tta_samples=tta_count,
            tta_method=tta_method,
            feature_engineering=parse_bool_arg(
                getattr(args, "feature_engineering", False), default=False
            ),
        )

    return XGBoostClassifierSK(
        target_vals,
        use_val_in_train=parse_bool_arg(args.train_with_val, default=False),
        use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
        subsample_method=model_subsample_method,
        n_train_data_samples=model_subsample_n,
        max_depth=getattr(args, "max_depth", 6),
        colsample_bytree=getattr(args, "colsample_bytree", 0.8),
        pre_downsample_method=model_downsample_method,
        pre_downsample_factor=config.prep.ds_factor,
        signal_combo_map=signal_combo_map,
        augment_samples=augment_count,
        augment_target=augment_target,
        augment_method=augment_method,
        mixup_alpha=getattr(args, "mixup_alpha", 0.4),
        cutmix_ratio=getattr(args, "cutmix_ratio", 0.3),
        tta_samples=tta_count,
        tta_method=tta_method,
        feature_engineering=parse_bool_arg(
            getattr(args, "feature_engineering", False), default=False
        ),
    )
