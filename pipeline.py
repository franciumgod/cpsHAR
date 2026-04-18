import importlib
import math
from pathlib import Path

import numpy as np


MAINLINE_SPECTRUM_METHOD = "rfft"
MAINLINE_FEATURE_DOMAIN_CHOICES = ("time", "freq", "time_freq")
MAINLINE_MODEL_SPECS = {
    "lightgbm": {
        "display_name": "LightGBM",
        "module": "models.LGBM",
        "class_name": "LightGBMClassifierSK",
    },
    "xgboost": {
        "display_name": "XGBoost",
        "module": "models.XGB",
        "class_name": "XGBoostClassifierSK",
    },
    "catboost": {
        "display_name": "CatBoost",
        "module": "models.CatBoost",
        "class_name": "CatBoostClassifierSK",
    },
    "tabm": {
        "display_name": "TabM",
        "module": "models.TabM",
        "class_name": "TabMClassifierSK",
    },
    "tabicl": {
        "display_name": "TabICL",
        "module": "models.TabICL",
        "class_name": "TabICLClassifierSK",
    },
    "rgf": {
        "display_name": "RGF",
        "module": "models.RGF",
        "class_name": "RGFClassifierSK",
    },
}
MAINLINE_MODEL_CHOICES = tuple(spec["display_name"] for spec in MAINLINE_MODEL_SPECS.values())
MODEL_NAME_ALIASES = {
    "lgb": "lightgbm",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "cat": "catboost",
    "catboost": "catboost",
    "tabm": "tabm",
    "tabicl": "tabicl",
    "rgf": "rgf",
}


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


def normalize_model_name(value):
    raw = str(value).strip()
    compact = raw.lower().replace("-", "").replace("_", "")
    return MODEL_NAME_ALIASES.get(compact, compact)


def get_display_model_name(value):
    key = normalize_model_name(value)
    spec = MAINLINE_MODEL_SPECS.get(key)
    return spec["display_name"] if spec else str(value)


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


def _load_model_class(model_name):
    key = normalize_model_name(model_name)
    spec = MAINLINE_MODEL_SPECS.get(key)
    if spec is None:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available mainline models: {', '.join(MAINLINE_MODEL_CHOICES)}"
        )

    try:
        module = importlib.import_module(spec["module"])
        return getattr(module, spec["class_name"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {spec['display_name']} from {spec['module']}.{spec['class_name']}: {exc}"
        ) from exc


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

    use_signal_combo = parse_bool_arg(getattr(args, "signal_combo", True), default=True)
    feature_domain = str(getattr(args, "feature_domain", "time_freq")).strip().lower()
    if feature_domain not in MAINLINE_FEATURE_DOMAIN_CHOICES:
        feature_domain = "time_freq"
    signal_combo_map = (
        build_special_signal_combo_map(config.data.sensor_cols) if use_signal_combo else None
    )

    common_kwargs = {
        "classes": target_vals,
        "use_val_in_train": parse_bool_arg(getattr(args, "train_with_val", True), default=True),
        "random_state": int(getattr(args, "random_state", 42)),
        "subsample_method": model_subsample_method,
        "n_train_data_samples": model_subsample_n,
        "feature_batch_size": getattr(args, "feature_batch_size", 2000),
        "pre_downsample_method": "false",
        "pre_downsample_factor": config.prep.ds_factor,
        "signal_combo_map": signal_combo_map,
        "sensor_cols": config.data.sensor_cols,
        "feature_engineering": parse_bool_arg(
            getattr(args, "feature_engineering", True), default=True
        ),
        "feature_domain": feature_domain,
        "spectrum_method": MAINLINE_SPECTRUM_METHOD,
    }

    model_key = normalize_model_name(args.model if getattr(args, "model", None) else "xgboost")
    model_cls = _load_model_class(model_key)

    if model_key == "lightgbm":
        return model_cls(
            **common_kwargs,
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
        )

    if model_key == "xgboost":
        return model_cls(
            **common_kwargs,
            use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
            max_depth=getattr(args, "max_depth", 6),
            colsample_bytree=getattr(args, "colsample_bytree", 0.8),
            n_estimators=getattr(args, "n_estimators", 2000),
            learning_rate=getattr(args, "learning_rate", 0.05),
            subsample=getattr(args, "subsample", 0.8),
            min_child_weight=getattr(args, "min_child_weight", 1.0),
            reg_alpha=getattr(args, "reg_alpha", 0.0),
            reg_lambda=getattr(args, "reg_lambda", 1.0),
            early_stopping_rounds=getattr(args, "early_stopping_rounds", 100),
        )

    if model_key == "catboost":
        return model_cls(
            **common_kwargs,
            use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
            max_depth=getattr(args, "max_depth", 6),
            colsample_bytree=getattr(args, "colsample_bytree", 0.8),
            n_estimators=getattr(args, "n_estimators", 1000),
            learning_rate=getattr(args, "learning_rate", 0.05),
            subsample=getattr(args, "subsample", 0.8),
            reg_lambda=getattr(args, "reg_lambda", 3.0),
        )

    if model_key == "rgf":
        return model_cls(
            **common_kwargs,
            max_leaf=getattr(args, "rgf_max_leaf", 1000),
            algorithm=getattr(args, "rgf_algorithm", "RGF"),
            reg_depth=getattr(args, "rgf_reg_depth", 1.0),
            l2=getattr(args, "rgf_l2", 0.1),
            learning_rate=getattr(args, "rgf_learning_rate", 0.5),
            min_samples_leaf=getattr(args, "rgf_min_samples_leaf", 10),
        )

    if model_key == "tabicl":
        return model_cls(
            **common_kwargs,
            use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
            n_estimators=getattr(args, "tabicl_n_estimators", 8),
            batch_size=getattr(args, "tabicl_batch_size", 8),
            kv_cache=parse_bool_arg(getattr(args, "tabicl_kv_cache", False), default=False),
            model_path=getattr(args, "tabicl_model_path", None),
            allow_auto_download=parse_bool_arg(
                getattr(args, "tabicl_allow_auto_download", False),
                default=False,
            ),
            checkpoint_version=getattr(
                args,
                "tabicl_checkpoint_version",
                "tabicl-classifier-v2-20260212.ckpt",
            ),
            device=getattr(args, "tabicl_device", None),
            verbose=parse_bool_arg(getattr(args, "tabicl_verbose", False), default=False),
        )

    if model_key == "tabm":
        return model_cls(
            **common_kwargs,
            use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
            max_epochs=getattr(args, "tabm_max_epochs", 40),
            batch_size=getattr(args, "tabm_batch_size", 256),
            learning_rate=getattr(args, "tabm_learning_rate", 1e-3),
            weight_decay=getattr(args, "tabm_weight_decay", 1e-4),
            patience=getattr(args, "tabm_patience", 8),
            validation_fraction=getattr(args, "tabm_validation_fraction", 0.15),
            arch_type=getattr(args, "tabm_arch_type", "tabm"),
            k=getattr(args, "tabm_k", 32),
            d_block=getattr(args, "tabm_d_block", 512),
            n_blocks=getattr(args, "tabm_n_blocks", 3),
            dropout=getattr(args, "tabm_dropout", 0.1),
        )

    # Reserved slots keep the registry stable. Re-enable by replacing the model file.
    return model_cls(
        **common_kwargs,
        use_gpu=parse_bool_arg(getattr(args, "use_gpu", False), default=False),
    )
