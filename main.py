import gc
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from data_handler import DataHandler
from pipeline import (
    MAINLINE_FEATURE_DOMAIN_CHOICES,
    MAINLINE_MODEL_CHOICES,
    MAINLINE_SPECTRUM_METHOD,
    build_model,
    get_display_model_name,
    parse_bool_arg,
    parse_fold_list,
    resolve_dataset_file,
    to_jsonable,
)
from sample_stats import (
    build_fold_classifier_input_stats,
    compute_dataset_label_ratio_stats,
    print_dataset_label_ratio_stats,
    print_fold_classifier_input_stats,
)
from utils.config import Config
from utils.utils import evaluate_and_print_multilabel_metrics, plot_confusion_and_timeline

import warnings
import sys
import numpy.core.numeric

sys.modules["numpy._core.numeric"] = numpy.core.numeric
warnings.filterwarnings("ignore")


def _split_label_stats(y, class_names):
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    n_samples = int(len(y_arr))
    n_classes = int(y_arr.shape[1]) if y_arr.ndim == 2 else 0
    pos_counts = np.sum(y_arr, axis=0).astype(int) if n_samples > 0 else np.zeros((n_classes,), dtype=int)
    multi_label_count = int(np.sum(np.sum(y_arr, axis=1) > 1)) if n_samples > 0 else 0

    rows = []
    max_len = max(len(class_names), n_classes)
    for i in range(max_len):
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        class_pos = int(pos_counts[i]) if i < len(pos_counts) else 0
        rows.append((class_name, class_pos))

    return n_samples, rows, multi_label_count


def _safe_ratio_matrix(ratio, y):
    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    if ratio is None:
        return y_arr.astype(np.float32, copy=False)

    ratio_arr = np.asarray(ratio, dtype=np.float32)
    if ratio_arr.ndim == 1:
        ratio_arr = ratio_arr.reshape(-1, 1)
    if ratio_arr.shape != y_arr.shape:
        return y_arr.astype(np.float32, copy=False)
    return ratio_arr


def _build_split_overview(split_data, class_names):
    y = split_data[1] if split_data and len(split_data) > 1 else np.empty((0, len(class_names)), dtype=np.int8)
    n_samples, rows, multi_label_count = _split_label_stats(y, class_names)
    return {
        "samples": n_samples,
        "per_class_positive": rows,
        "multi_label_count": multi_label_count,
    }


def print_key_run_params(args, resolved_dataset_file):
    def on_off(v, default=False):
        return "ON" if parse_bool_arg(v, default=default) else "OFF"

    print("\n=== Mainline Params ===")
    print(f"model          : {get_display_model_name(args.model)}")
    print(f"data           : {args.data} -> {resolved_dataset_file}")
    print(f"folds          : {args.folds}")
    print(f"train_with_val : {on_off(args.train_with_val, default=True)}")
    print(f"signal_combo   : {on_off(args.signal_combo, default=True)}")
    print(f"synthetic_axes : {on_off(args.synthetic_axes, default=True)}")
    print(f"feature_engine : {on_off(args.feature_engineering, default=True)}")
    print(f"feature_domain : {str(getattr(args, 'feature_domain', 'time_freq')).strip().lower()}")
    print(f"spectrum_method: {MAINLINE_SPECTRUM_METHOD}")
    print("=======================\n")


def print_fold_preview(preview, class_names, include_sample_stats):
    print(
        f"[Fold {preview['fold']}] "
        f"test_experiment={preview['fold']} | validation_experiment={preview['validation_experiment_id']}"
    )
    print(
        f"  split samples: "
        f"train={preview['train_samples']} | val={preview['val_samples']} | test={preview['test_samples']}"
    )
    for split_name in ("train", "val", "test"):
        split_info = preview["splits"][split_name]
        print(f"  [{split_name}] samples={split_info['samples']}")
        for class_name, class_pos in split_info["per_class_positive"]:
            print(f"    - {class_name}: {class_pos}")
        print(f"    - Multilabel(>1 active labels): {split_info['multi_label_count']}")

    if include_sample_stats and preview.get("sample_stats") is not None:
        print_fold_classifier_input_stats(preview["fold"], preview["sample_stats"], class_names)


def collect_fold_previews(datahandler, fold_ids, args):
    previews = []
    dataset_level_sample_stats = None
    class_names = None
    include_val_in_train = parse_bool_arg(args.train_with_val, default=True)
    enable_sample_stats = parse_bool_arg(args.sample_stats, default=True)

    print("\n=== Preflight Fold Summary ===")
    for fold in fold_ids:
        val_id = fold + 1 if fold < 4 else 1
        datahandler.config.data.test_experiment_id = fold
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()
        class_names = target_vals

        split_ratio_pre_ds = datahandler.get_last_split_ratio_pre_ds()
        train_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("train"), train[1])
        val_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("val"), val[1])

        if enable_sample_stats and dataset_level_sample_stats is None:
            dataset_level_sample_stats = compute_dataset_label_ratio_stats(datahandler, target_vals)
            print_dataset_label_ratio_stats(dataset_level_sample_stats)

        fold_sample_stats = None
        if enable_sample_stats:
            if include_val_in_train:
                y_for_stats = np.concatenate([np.asarray(train[1]), np.asarray(val[1])], axis=0)
                ratio_for_stats = np.concatenate([train_ratio, val_ratio], axis=0)
            else:
                y_for_stats = np.asarray(train[1])
                ratio_for_stats = train_ratio

            fold_sample_stats = build_fold_classifier_input_stats(
                y_train=y_for_stats,
                ratio_pre_ds_train=ratio_for_stats,
                class_names=target_vals,
            )

        preview = {
            "fold": int(fold),
            "validation_experiment_id": int(val_id),
            "train_samples": int(train[0].shape[0]) if train and train[0] is not None else 0,
            "val_samples": int(val[0].shape[0]) if val and val[0] is not None else 0,
            "test_samples": int(test[0].shape[0]) if test and test[0] is not None else 0,
            "splits": {
                "train": _build_split_overview(train, target_vals),
                "val": _build_split_overview(val, target_vals),
                "test": _build_split_overview(test, target_vals),
            },
            "sample_stats": fold_sample_stats,
        }
        previews.append(preview)
        print_fold_preview(preview, target_vals, include_sample_stats=enable_sample_stats)

        del train, val, test
        gc.collect()

    print("=== End Preflight Summary ===\n")
    return previews, dataset_level_sample_stats, class_names


def collect_selected_hparams(args):
    model_name = get_display_model_name(args.model)
    common = {
        "random_state": getattr(args, "random_state", 42),
        "feature_domain": str(args.feature_domain).strip().lower(),
        "feature_engineering": parse_bool_arg(args.feature_engineering, default=True),
        "signal_combo": parse_bool_arg(args.signal_combo, default=True),
        "synthetic_axes": parse_bool_arg(args.synthetic_axes, default=True),
        "train_with_val": parse_bool_arg(args.train_with_val, default=True),
        "spectrum_method": MAINLINE_SPECTRUM_METHOD,
    }

    if model_name == "LightGBM":
        common.update(
            {
                "boosting_type": getattr(args, "boosting_type", "lgbt"),
                "max_depth": getattr(args, "max_depth", None),
                "num_leaves": getattr(args, "num_leaves", None),
                "n_estimators": getattr(args, "n_estimators", None),
                "learning_rate": getattr(args, "learning_rate", None),
                "subsample": getattr(args, "subsample", None),
                "colsample_bytree": getattr(args, "colsample_bytree", None),
                "reg_alpha": getattr(args, "reg_alpha", None),
                "reg_lambda": getattr(args, "reg_lambda", None),
            }
        )
    elif model_name == "XGBoost":
        common.update(
            {
                "max_depth": getattr(args, "max_depth", None),
                "n_estimators": getattr(args, "n_estimators", None),
                "learning_rate": getattr(args, "learning_rate", None),
                "subsample": getattr(args, "subsample", None),
                "colsample_bytree": getattr(args, "colsample_bytree", None),
                "min_child_weight": getattr(args, "min_child_weight", None),
                "reg_alpha": getattr(args, "reg_alpha", None),
                "reg_lambda": getattr(args, "reg_lambda", None),
            }
        )
    elif model_name == "CatBoost":
        common.update(
            {
                "max_depth": getattr(args, "max_depth", None),
                "n_estimators": getattr(args, "n_estimators", None),
                "learning_rate": getattr(args, "learning_rate", None),
                "subsample": getattr(args, "subsample", None),
                "colsample_bytree": getattr(args, "colsample_bytree", None),
                "reg_lambda": getattr(args, "reg_lambda", None),
            }
        )
    elif model_name == "RGF":
        common.update(
            {
                "rgf_max_leaf": getattr(args, "rgf_max_leaf", None),
                "rgf_algorithm": getattr(args, "rgf_algorithm", None),
                "rgf_reg_depth": getattr(args, "rgf_reg_depth", None),
                "rgf_l2": getattr(args, "rgf_l2", None),
                "rgf_learning_rate": getattr(args, "rgf_learning_rate", None),
                "rgf_min_samples_leaf": getattr(args, "rgf_min_samples_leaf", None),
            }
        )
    elif model_name == "TabM":
        common.update(
            {
                "tabm_max_epochs": getattr(args, "tabm_max_epochs", None),
                "tabm_batch_size": getattr(args, "tabm_batch_size", None),
                "tabm_learning_rate": getattr(args, "tabm_learning_rate", None),
                "tabm_weight_decay": getattr(args, "tabm_weight_decay", None),
                "tabm_patience": getattr(args, "tabm_patience", None),
                "tabm_validation_fraction": getattr(args, "tabm_validation_fraction", None),
                "tabm_arch_type": getattr(args, "tabm_arch_type", None),
                "tabm_k": getattr(args, "tabm_k", None),
                "tabm_d_block": getattr(args, "tabm_d_block", None),
                "tabm_n_blocks": getattr(args, "tabm_n_blocks", None),
                "tabm_dropout": getattr(args, "tabm_dropout", None),
            }
        )
    elif model_name == "TabICL":
        common.update(
            {
                "tabicl_n_estimators": getattr(args, "tabicl_n_estimators", None),
                "tabicl_batch_size": getattr(args, "tabicl_batch_size", None),
                "tabicl_kv_cache": parse_bool_arg(getattr(args, "tabicl_kv_cache", False), default=False),
                "tabicl_model_path": getattr(args, "tabicl_model_path", None),
                "tabicl_allow_auto_download": parse_bool_arg(
                    getattr(args, "tabicl_allow_auto_download", False),
                    default=False,
                ),
                "tabicl_checkpoint_version": getattr(args, "tabicl_checkpoint_version", None),
                "tabicl_device": getattr(args, "tabicl_device", None),
            }
        )
    return common


if __name__ == "__main__":
    import argparse

    def _normalize_legacy_model_argv(argv):
        # Backward-compatibility:
        # allow legacy forms like `python main.py --data 500 model RGF`
        # by stripping the literal token `model`.
        normalized = list(argv)
        if "model" in normalized:
            idx = normalized.index("model")
            if idx + 1 < len(normalized):
                normalized.pop(idx)
        return normalized

    parser = argparse.ArgumentParser(
        description="Mainline cpsHAR runner: fixed train-with-val protocol, default handcrafted features, and RFFT-only frequency branch."
    )

    parser.add_argument(
        "model",
        nargs="?",
        default="XGBoost",
        help=f"Mainline model. Recommended choices: {', '.join(MAINLINE_MODEL_CHOICES)}",
    )
    parser.add_argument(
        "--data",
        default="500",
        help="Dataset selector: raw, a step number like 200/500, or a custom filename under data/.",
    )
    parser.add_argument(
        "--output",
        default="output/mainline",
        help="Output directory for run summary and confusion plots.",
    )
    parser.add_argument(
        "--train_with_val",
        default=True,
        help="Mainline default is True: merge train+val so each fold uses 3 experiments for training and 1 for test.",
    )
    parser.add_argument(
        "--train_sample_num",
        type=int,
        default=100_000,
        help="Max number of training windows after subsampling.",
    )
    parser.add_argument(
        "--subsample_method",
        default="interval",
        help="Training-window subsampling strategy: false, random, interval.",
    )
    parser.add_argument(
        "--downsample_method",
        default="sliding_window",
        help="Raw-to-target downsampling strategy: false, interval, sliding_window.",
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4",
        help="Comma-separated test experiment ids, e.g. 1,2,3,4.",
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        help="Whether to use GPU for models that support it.",
    )
    parser.add_argument(
        "--show_images",
        default=False,
        help="Show plots while saving.",
    )
    parser.add_argument(
        "--signal_combo",
        default=True,
        help="Keep per-class special signal combinations on/off.",
    )
    parser.add_argument(
        "--synthetic_axes",
        default=True,
        help="Keep the extra synthetic axes (Acc.norm and Gyro.norm) on/off.",
    )
    parser.add_argument(
        "--feature_engineering",
        default=True,
        help="Use the current default handcrafted feature block.",
    )
    parser.add_argument(
        "--feature_domain",
        type=str,
        default="time_freq",
        choices=MAINLINE_FEATURE_DOMAIN_CHOICES,
        help="Feature domain: time, freq, or time_freq.",
    )
    parser.add_argument(
        "--spectrum_method",
        type=str,
        default=MAINLINE_SPECTRUM_METHOD,
        choices=[MAINLINE_SPECTRUM_METHOD],
        help="Mainline locks frequency features to RFFT. Re-enable legacy methods by expanding this choice list later.",
    )
    parser.add_argument(
        "--single_label_only",
        default=False,
        help="Optional post-window filter to remove multi-label training samples.",
    )
    parser.add_argument(
        "--sample_stats",
        default=True,
        help="Print and store dataset/fold label-ratio summaries.",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed used by model wrappers.")

    parser.add_argument("--max_depth", type=int, default=6, help="Tree depth for LightGBM/XGBoost.")
    parser.add_argument("--colsample_bytree", type=float, default=0.8, help="Feature subsampling ratio.")
    parser.add_argument("--n_estimators", type=int, default=1000, help="Boosting rounds for tree models.")
    parser.add_argument("--learning_rate", type=float, default=0.05, help="Learning rate for tree models.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Row subsampling ratio for tree models.")
    parser.add_argument("--reg_alpha", type=float, default=0.0, help="L1 regularization for tree models.")
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="L2 regularization for tree models.")
    parser.add_argument("--feature_batch_size", type=int, default=2000, help="Feature extraction batch size.")

    parser.add_argument("--boosting_type", type=str, default="lgbt", choices=["lgbt", "gbdt", "dart"])
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--max_drop", type=int, default=None)
    parser.add_argument("--skip_drop", type=float, default=None)
    parser.add_argument("--min_child_samples", type=int, default=30)
    parser.add_argument("--min_data_in_leaf", type=int, default=10)

    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--rgf_max_leaf", type=int, default=1000)
    parser.add_argument("--rgf_algorithm", type=str, default="RGF")
    parser.add_argument("--rgf_reg_depth", type=float, default=1.0)
    parser.add_argument("--rgf_l2", type=float, default=0.1)
    parser.add_argument("--rgf_learning_rate", type=float, default=0.5)
    parser.add_argument("--rgf_min_samples_leaf", type=int, default=10)
    parser.add_argument("--tabm_max_epochs", type=int, default=40)
    parser.add_argument("--tabm_batch_size", type=int, default=256)
    parser.add_argument("--tabm_learning_rate", type=float, default=1e-3)
    parser.add_argument("--tabm_weight_decay", type=float, default=1e-4)
    parser.add_argument("--tabm_patience", type=int, default=8)
    parser.add_argument("--tabm_validation_fraction", type=float, default=0.15)
    parser.add_argument("--tabm_arch_type", type=str, default="tabm", choices=["tabm", "tabm-mini", "tabm-packed"])
    parser.add_argument("--tabm_k", type=int, default=32)
    parser.add_argument("--tabm_d_block", type=int, default=512)
    parser.add_argument("--tabm_n_blocks", type=int, default=3)
    parser.add_argument("--tabm_dropout", type=float, default=0.1)
    parser.add_argument("--tabicl_n_estimators", type=int, default=8)
    parser.add_argument("--tabicl_batch_size", type=int, default=8)
    parser.add_argument("--tabicl_kv_cache", default=False)
    parser.add_argument("--tabicl_model_path", default=None)
    parser.add_argument("--tabicl_allow_auto_download", default=False)
    parser.add_argument("--tabicl_checkpoint_version", default="tabicl-classifier-v2-20260212.ckpt")
    parser.add_argument("--tabicl_device", default=None)
    parser.add_argument("--tabicl_verbose", default=False)

    args = parser.parse_args(_normalize_legacy_model_argv(sys.argv[1:]))
    args.model = get_display_model_name(args.model)

    config = Config()
    config.data.dataset_file = resolve_dataset_file(args.data)
    config.data.set_sensor_cols(include_synthetic_axes=parse_bool_arg(args.synthetic_axes, default=True))

    show_images = parse_bool_arg(args.show_images, default=False)
    enable_sample_stats = parse_bool_arg(args.sample_stats, default=True)
    preprocess_order = "subsample_first"

    print("Preprocess order: subsample_first")
    print_key_run_params(args, config.data.dataset_file)

    seed = int(getattr(args, "random_state", 42))
    random.seed(seed)
    np.random.seed(seed)

    fold_ids = parse_fold_list(args.folds)
    datahandler = DataHandler(
        config=config,
        downsample_method=args.downsample_method,
        subsample_method=args.subsample_method,
        n_train_data_samples=args.train_sample_num,
        preprocess_order=preprocess_order,
        single_label_only=args.single_label_only,
    )

    fold_previews, dataset_level_sample_stats, target_vals = collect_fold_previews(
        datahandler=datahandler,
        fold_ids=fold_ids,
        args=args,
    )

    test_mccs = []
    macro_f1s = []
    macro_pr_aucs = []
    macro_briers = []
    all_true = []
    all_pred = []

    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "main.py",
        "model": get_display_model_name(args.model),
        "resolved_dataset_file": config.data.dataset_file,
        "preprocess_order": preprocess_order,
        "arguments": vars(args),
        "preflight": {
            "dataset_level_sample_stats": dataset_level_sample_stats if enable_sample_stats else None,
            "folds": fold_previews,
        },
        "folds": [],
        "aggregate": {},
    }

    for fold_idx, fold in enumerate(fold_ids, start=1):
        print(f"\n--- Train Fold {fold_idx}/{len(fold_ids)} | test_experiment={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()
        split_ratio_pre_ds = datahandler.get_last_split_ratio_pre_ds()
        train_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("train"), train[1])
        val_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("val"), val[1])
        train_for_model = (train[0], train[1], train_ratio)
        val_for_model = (val[0], val[1], val_ratio)

        fold_sample_stats = None
        if enable_sample_stats:
            include_val_in_train = parse_bool_arg(args.train_with_val, default=True)
            if include_val_in_train:
                y_for_stats = np.concatenate([np.asarray(train[1]), np.asarray(val[1])], axis=0)
                ratio_for_stats = np.concatenate([train_ratio, val_ratio], axis=0)
            else:
                y_for_stats = np.asarray(train[1])
                ratio_for_stats = train_ratio
            fold_sample_stats = build_fold_classifier_input_stats(
                y_train=y_for_stats,
                ratio_pre_ds_train=ratio_for_stats,
                class_names=target_vals,
            )

        try:
            model = build_model(args, target_vals, config, preprocess_order)
            if hasattr(model, "set_input_scaler"):
                model.set_input_scaler(datahandler.scaler)

            print("Training model...")
            model.train(train_for_model, val_for_model)
            print("Evaluating model...")

            eval_y_true = test[1]
            eval_y_pred = model.predict(test[0])
            eval_y_prob = model.predict_proba(test[0]) if hasattr(model, "predict_proba") else None

            fold_metrics = evaluate_and_print_multilabel_metrics(
                y_true=eval_y_true,
                y_pred=eval_y_pred,
                y_prob=eval_y_prob,
                class_names=target_vals,
                fold_idx=fold,
                split_name="Test",
            )

            macro_f1s.append(fold_metrics["macro_f1"])
            macro_pr_aucs.append(fold_metrics["macro_pr_auc"])
            macro_briers.append(fold_metrics["macro_brier"])
            test_mccs.append(fold_metrics["macro_mcc"])
            all_true.append(eval_y_true)
            all_pred.append(eval_y_pred)

            plot_paths = plot_confusion_and_timeline(
                y_true=eval_y_true,
                y_pred=eval_y_pred,
                class_names=target_vals,
                fold_label=f"fold_{fold}",
                X_for_timeline=None,
                save_dir=args.output,
                show_plots=show_images,
                save_timeline=False,
                save_binary_confusion=False,
            )
            print(f"Saved fold confusion plot: {plot_paths['confusion_path']}")

            run_summary["folds"].append(
                {
                    "fold": int(fold),
                    "validation_experiment_id": int(val_id),
                    "train_samples": int(train[0].shape[0]) if train and train[0] is not None else 0,
                    "val_samples": int(val[0].shape[0]) if val and val[0] is not None else 0,
                    "test_samples": int(eval_y_true.shape[0]) if eval_y_true is not None else 0,
                    "metrics": fold_metrics,
                    "plots": plot_paths,
                    "selected_hparams": collect_selected_hparams(args),
                    "sample_stats": fold_sample_stats if enable_sample_stats else None,
                }
            )

        except Exception as exc:
            print(f"Fold {fold} failed with error: {exc}")
            raise

        del train, val, test, model
        gc.collect()
        print("--- End of Fold ---")

    avg_mcc = float(np.mean(test_mccs)) if test_mccs else None

    if all_true and all_pred:
        overall_true = np.concatenate(all_true, axis=0)
        overall_pred = np.concatenate(all_pred, axis=0)
        overall_plot_paths = plot_confusion_and_timeline(
            y_true=overall_true,
            y_pred=overall_pred,
            class_names=target_vals,
            fold_label="overall",
            X_for_timeline=None,
            save_dir=args.output,
            show_plots=show_images,
            save_timeline=False,
            save_binary_confusion=True,
            binary_n_cols=1,
            binary_file_prefix="vertical_binary_confusion",
        )
        print(
            "Saved overall plots: "
            f"{overall_plot_paths['confusion_path']} | "
            f"{overall_plot_paths.get('binary_confusion_path')}"
        )
    else:
        overall_plot_paths = {}

    run_summary["aggregate"] = {
        "folds": [int(x) for x in fold_ids],
        "mcc_per_fold": test_mccs,
        "macro_f1_per_fold": macro_f1s,
        "macro_pr_auc_per_fold": macro_pr_aucs,
        "macro_brier_per_fold": macro_briers,
        "avg_mcc": avg_mcc,
        "avg_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else None,
        "avg_macro_pr_auc": float(np.nanmean(macro_pr_aucs)) if macro_pr_aucs else None,
        "avg_macro_brier": float(np.mean(macro_briers)) if macro_briers else None,
        "overall_plots": overall_plot_paths,
        "sample_stats_enabled": enable_sample_stats,
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_summary), f, ensure_ascii=False, indent=2)
    print(f"Saved run summary: {summary_path}")

    print("Scores for each run: ", test_mccs)
    if macro_f1s:
        print(f"Macro-F1 per fold: {macro_f1s} | avg={np.mean(macro_f1s):.4f}")
    if macro_pr_aucs:
        print(f"Macro PR-AUC per fold: {macro_pr_aucs} | avg={np.nanmean(macro_pr_aucs):.4f}")
    if macro_briers:
        print(f"Macro Brier per fold: {macro_briers} | avg={np.mean(macro_briers):.4f}")
    print("\nTotal score:", avg_mcc)
