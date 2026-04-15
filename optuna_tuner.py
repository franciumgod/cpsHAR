import argparse
import gc
from copy import deepcopy

import numpy as np

from pipeline import build_model
from utils.utils import calculate_mcc_multilabel


def _suggest_xgb_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
    }


def _suggest_lgbm_params(trial):
    max_depth = trial.suggest_int("max_depth", 3, 12)
    max_leaves = max(15, min(255, (2 ** max_depth) - 1))
    min_child_samples = trial.suggest_int("min_child_samples", 10, 120)
    min_data_upper = max(5, min(80, min_child_samples))

    return {
        "max_depth": max_depth,
        "num_leaves": trial.suggest_int("num_leaves", 15, max_leaves),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_samples": min_child_samples,
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, min_data_upper),
    }


def tune_hyperparameters_for_fold(
    args,
    target_vals,
    config,
    preprocess_order,
    train_data,
    val_data,
    scaler=None,
):
    try:
        import optuna
    except Exception as exc:
        raise RuntimeError(
            "Optuna is required for hyperparameter tuning. "
            "Please install it first: pip install optuna"
        ) from exc

    model_name = str(args.model).lower() if args.model else "xgboost"
    if model_name not in {"xgboost", "lightgbm"}:
        raise ValueError(f"Unsupported model for Optuna tuning: {model_name}")

    n_trials = int(getattr(args, "optuna_trials", 0))
    timeout = getattr(args, "optuna_timeout", None)
    timeout = None if timeout is None else int(timeout)
    seed = int(getattr(args, "optuna_seed", 42))

    val_x, val_y = val_data
    if val_x is None or len(val_x) == 0:
        raise ValueError("Validation split is empty; Optuna tuning requires validation data.")

    def objective(trial):
        trial_args = argparse.Namespace(**vars(deepcopy(args)))
        trial_args.train_with_val = False

        if model_name == "xgboost":
            sampled = _suggest_xgb_params(trial)
        else:
            sampled = _suggest_lgbm_params(trial)

        for k, v in sampled.items():
            setattr(trial_args, k, v)

        model = build_model(
            trial_args,
            target_vals=target_vals,
            config=config,
            preprocess_order=preprocess_order,
        )
        if hasattr(model, "set_input_scaler"):
            model.set_input_scaler(scaler)

        model.train(train_data, val_data)
        pred_val = model.predict(val_x)
        score = float(calculate_mcc_multilabel(np.asarray(val_y), np.asarray(pred_val)))

        del model, pred_val
        gc.collect()
        if not np.isfinite(score):
            return -1.0
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    return {
        "model": model_name,
        "n_trials": int(len(study.trials)),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
    }
