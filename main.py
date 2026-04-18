import random
import gc
import copy
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from utils.utils import (
    evaluate_and_print_multilabel_metrics,
    plot_confusion_and_timeline,
)
from utils.config import Config
from data_handler import DataHandler
from optuna_tuner import tune_hyperparameters_for_fold
from sample_stats import (
    compute_dataset_label_ratio_stats,
    build_fold_classifier_input_stats,
    print_dataset_label_ratio_stats,
    print_fold_classifier_input_stats,
)
from pipeline import (
    resolve_dataset_file,
    parse_bool_arg,
    parse_fold_list,
    sample_windows_for_timeline,
    to_jsonable,
    build_model,
)
import warnings
import sys
import numpy.core.numeric
sys.modules['numpy._core.numeric'] = numpy.core.numeric

# import sys
# import numpy.core.numeric
# sys.modules['numpy._core.numeric'] = numpy.core.numeric


warnings.filterwarnings("ignore")


def align_test_scale_with_train(test_x, source_scaler, target_scaler):
    if (
        test_x is None
        or not isinstance(test_x, np.ndarray)
        or test_x.size == 0
        or source_scaler is None
        or target_scaler is None
    ):
        return test_x

    n_channels = test_x.shape[-1]
    flat = test_x.reshape(-1, n_channels)
    raw_flat = source_scaler.inverse_transform(flat)
    train_scaled_flat = target_scaler.transform(raw_flat)
    return train_scaled_flat.reshape(test_x.shape).astype(np.float32, copy=False)


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


def print_fold_data_overview(train, val, test, class_names):
    train_n = int(train[0].shape[0]) if train and train[0] is not None else 0
    val_n = int(val[0].shape[0]) if val and val[0] is not None else 0
    test_n = int(test[0].shape[0]) if test and test[0] is not None else 0
    print(
        f"Split samples before training: "
        f"train={train_n}, val={val_n}, test={test_n}"
    )

    for split_name, split_data in (("train", train), ("val", val), ("test", test)):
        y = split_data[1] if split_data and len(split_data) > 1 else np.empty((0, len(class_names)), dtype=np.int8)
        n_samples, rows, multi_label_count = _split_label_stats(y, class_names)
        print(f"[{split_name}] samples={n_samples}")
        for class_name, class_pos in rows:
            print(f"  - {class_name}: {class_pos}")
        print(f"  - Multilabel(>1 active labels): {multi_label_count}")


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


def print_key_run_params(args, resolved_dataset_file):
    def on_off(v):
        return "ON" if parse_bool_arg(v, default=False) else "OFF"

    print("\n=== Key Params ===")
    print(f"data           : {args.data} -> {resolved_dataset_file}")
    print(f"train_with_val : {on_off(args.train_with_val)}")
    print(f"signal_combo   : {on_off(args.signal_combo)}")
    print(f"feature_engine : {on_off(args.feature_engineering)}")
    print(f"feature_domain : {str(getattr(args, 'feature_domain', 'time')).strip().lower()}")
    print(f"spectrum_method: {str(getattr(args, 'spectrum_method', 'rfft')).strip().lower()}")
    print(f"pos_threshold  : {str(getattr(args, 'pos_threshold', '')).strip() or 'NONE'}")
    print(f"ovr_neg_balance: {on_off(getattr(args, 'ovr_neg_balance', False))}")
    print(f"ovr_pos_neg_rt : {float(getattr(args, 'ovr_pos_neg_ratio', 1.0))}")
    print(f"ovr_neg_target : {str(getattr(args, 'ovr_neg_target_ratio', '')).strip() or 'NONE'}")
    print("==================\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Training and evaluation pipeline for multi-label activity recognition."
    )

    parser.add_argument(
        "--data",
        default="raw",
        help="Dataset selector: raw (default), or a step number like 100/200/400/500/800/1000, or a custom filename under data/."
    )
    parser.add_argument(
        "--output",
        default="output/plot",
        help="Output directory for generated plots and evaluation results."
    )
    parser.add_argument(
        "--train_with_val",
        default=False,
        help="Whether to merge training and validation sets for model training (default: False)."
    )
    parser.add_argument(
        "--train_sample_num",
        type=int,
        default=100_000,
        help="Number of training samples to use for model training."
    )
    parser.add_argument(
        "--subsample_method",
        default="interval",
        help="Strategy for subsampling training data:\n"
             "  - False    : Use all available windows without subsampling.\n"
             "  - random   : Randomly select samples from the training set (or combined train/val if --train_with_val=True).\n"
             "  - interval : Select samples at a fixed interval to preserve temporal distribution."
    )
    parser.add_argument(
        "--downsample_method",
        default="sliding_window",
        help="Strategy for reducing the raw 2000Hz signal:\n"
             "  - False           : Keep original data at 2000Hz (no downsampling applied).\n"
             "  - interval        : Simple decimation. Selects every N-th sample to achieve the target sampling rate.\n"
             "  - sliding_window  : Computes channel-wise moving average over 'window_size' samples, "
             "then advances by 'window_step'. The resulting signal is downsampled to the target frequency."
    )
    parser.add_argument(
        "--folds",
        default="1,2,3,4",
        help="Idx of training and testing fold"
    )
    parser.add_argument(
        "--use_gpu",
        default=False,
        help="Whether to use GPU or not."
    )
    parser.add_argument(
        "--show_images",
        default=False,
        help="Drawing picture while saving."
    )
    parser.add_argument(
        "--signal_combo",
        default=True,
        help="Whether to use the specific signal combinations for training (default: False)."
    )
    parser.add_argument(
        "--feature_engineering",
        default=True,
        help="Enable additional engineered features: RMS, RMSE, first-order diff, and lags (1,3,5,10)."
    )
    parser.add_argument(
        "--use_tsfresh",
        default=False,
        help="Enable tsfresh feature extraction and append tsfresh features to model input."
    )
    parser.add_argument(
        "--feature_domain",
        type=str,
        default="time",
        choices=["time", "freq", "time_freq"],
        help="Feature domain for handcrafted features: time (default), freq, or time_freq."
    )
    parser.add_argument(
        "--spectrum_method",
        type=str,
        default="welch_psd",
        help="Frequency-spectrum generator when feature_domain includes freq. "
             "Supported: rfft, welch_psd, stft, dwt."
    )
    parser.add_argument(
        "--sample_augment",
        default="false",
        help="Sample perturbation augmentation for training only. "
             "Use False/0 to disable (default), True to add 1 augmented sample per matched sample, "
             "or pass an integer N to add N augmented samples per matched sample."
    )
    parser.add_argument(
        "--augment_target",
        default="multilabel",
        help="Augmentation target selector. Default 'multilabel' (samples with >1 active labels). "
             "Supports comma-separated multiple targets (union), e.g. "
             "'Lifting(lowering),Driving(Straight)' or 'multilabel,Driving(curve)'."
    )
    parser.add_argument(
        "--augment_method",
        default="jitter",
        help="Augmentation method(s), comma-separated. "
             "Supported: jitter,scaling,rotation,mixup,cutmix,smote,basic."
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.4,
        help="Beta(alpha, alpha) parameter used by mixup."
    )
    parser.add_argument(
        "--cutmix_ratio",
        type=float,
        default=0.3,
        help="Time-axis cut ratio used by cutmix, in [0,1]."
    )
    parser.add_argument(
        "--rotation_plane",
        type=str,
        default="xyz",
        choices=["xyz", "xy", "xz", "yz"],
        help="Rotation augmentation plane. xyz=3D random, xy/xz/yz=single-plane rotation."
    )
    parser.add_argument(
        "--rotation_max_degrees",
        type=float,
        default=15.0,
        help="Maximum absolute rotation angle in degrees."
    )
    parser.add_argument(
        "--tta",
        default="false",
        help="Test-Time Augmentation count. False/0 disables; True enables 1 view; integer N enables N views."
    )
    parser.add_argument(
        "--tta_method",
        default="jitter",
        help="TTA method(s), comma-separated. "
             "Supported: jitter,scaling,rotation,mixup,cutmix,smote,basic."
    )
    parser.add_argument(
        "--test_on_step1_full",
        default=False,
        help="Use full step=1 sliding windows for test split while keeping train/val from --data."
    )
    parser.add_argument(
        "--optuna_trials",
        type=int,
        default=0,
        help="Number of Optuna trials per fold. 0 disables tuning (default)."
    )
    parser.add_argument(
        "--optuna_timeout",
        type=int,
        default=None,
        help="Optional Optuna timeout (seconds) per fold."
    )
    parser.add_argument(
        "--optuna_seed",
        type=int,
        default=42,
        help="Random seed for Optuna TPE sampler."
    )
    parser.add_argument(
        "--single_label_only",
        default=False,
        help="Remove multi-label samples from training set (keep samples with <=1 active label)."
    )
    parser.add_argument(
        "--sample_stats",
        default=False,
        help="Enable sample statistics collection and printing. "
             "Stats are written into run_summary.json together with training/testing results."
    )
    parser.add_argument(
        "--pos_threshold",
        default="",
        help="Per-classifier positive ratio threshold. "
             "Format: 'LabelA:0.3,LabelB:0.2' (ratio uses pre-downsample window label ratio)."
    )
    parser.add_argument(
        "--ovr_neg_balance",
        default=False,
        help="Enable OvR negative-sample balancing per classifier."
    )
    parser.add_argument(
        "--ovr_pos_neg_ratio",
        type=float,
        default=1.0,
        help="Target pos/neg ratio per OvR classifier when --ovr_neg_balance=True. "
             "1 means equal, 2 means positives are 2x negatives."
    )
    parser.add_argument(
        "--ovr_neg_target_ratio",
        default="",
        help="Optional per-classifier negative-bucket target ratio. "
             "Format: 'TargetLabel:NegBucketLabel:Percent;Target2:Neg2:Percent'. "
             "Example: 'Lifting(raising):Driving(curve):15'."
    )
    subparsers = parser.add_subparsers(dest="model", help="Select the model type.")
    parser_XGB = subparsers.add_parser(
        "XGBoost",
        help="Configuration for XGBoost model."
    )
    parser_XGB.add_argument("--max_depth", type=int, default=6, help="Maximum depth for XGBoost.")
    parser_XGB.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="Fraction of features (columns) to randomly sample when constructing each tree (XGBoost parameter)."
    )
    parser_LGBM = subparsers.add_parser(
        "LightGBM",
        help="Configuration for LightGBM model."
    )
    parser_LGBM.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="Fraction of features (columns) to randomly sample when constructing each tree (LightGBM parameter)."
    )
    parser_LGBM.add_argument(
        "--boosting_type",
        type=str,
        default="lgbt",
        choices=["lgbt", "gbdt", "dart"],
        help="Boosting type. 'lgbt' is kept as alias and internally mapped to LightGBM 'gbdt'."
    )
    parser_LGBM.add_argument(
        "--max_depth",
        type=int,
        default=6,
        help="Maximum tree depth for LightGBM."
    )
    parser_LGBM.add_argument(
        "--num_leaves",
        type=int,
        default=63,
        help="Maximum number of leaves in one tree."
    )
    parser_LGBM.add_argument(
        "--n_estimators",
        type=int,
        default=1000,
        help="Number of boosting trees."
    )
    parser_LGBM.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Boosting learning rate."
    )
    parser_LGBM.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="Row sampling ratio for each tree."
    )
    parser_LGBM.add_argument(
        "--reg_alpha",
        type=float,
        default=0.0,
        help="L1 regularization term."
    )
    parser_LGBM.add_argument(
        "--reg_lambda",
        type=float,
        default=0.0,
        help="L2 regularization term."
    )
    parser_LGBM.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for DART boosting."
    )
    parser_LGBM.add_argument(
        "--max_drop",
        type=int,
        default=None,
        help="Max dropped trees for DART. Ignored unless --boosting_type dart."
    )
    parser_LGBM.add_argument(
        "--skip_drop",
        type=float,
        default=None,
        help="Probability to skip dropout for DART. Ignored unless --boosting_type dart."
    )
    parser_LGBM.add_argument(
        "--feature_batch_size",
        type=int,
        default=2000,
        help="Batch size used during feature extraction."
    )
    parser_LGBM.add_argument(
        "--min_child_samples",
        type=int,
        default=30,
        help="Minimum samples needed in a child node."
    )
    parser_LGBM.add_argument(
        "--min_data_in_leaf",
        type=int,
        default=10,
        help="Minimum data points in one leaf."
    )

    args = parser.parse_args()
    config = Config()
    config.data.dataset_file = resolve_dataset_file(args.data)
    show_images = parse_bool_arg(args.show_images, default=False)
    enable_sample_stats = parse_bool_arg(args.sample_stats, default=False)
    preprocess_order = "subsample_first"
    print("Preprocess order: subsample_first")
    print_key_run_params(args, config.data.dataset_file)

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    test_mccs = []

    fold_ids = parse_fold_list(args.folds)
    macro_f1s = []
    macro_pr_aucs = []
    macro_briers = []
    all_true = []
    all_pred = []
    all_test_x_timeline = []

    overall_timeline_max_windows = 12000
    per_fold_timeline_windows = max(500, overall_timeline_max_windows // max(1, len(fold_ids)))
    run_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "main.py",
        "model": args.model if args.model else "XGBoost",
        "resolved_dataset_file": config.data.dataset_file,
        "preprocess_order": preprocess_order,
        "arguments": vars(args),
        "sample_stats": {"enabled": enable_sample_stats},
        "folds": [],
        "aggregate": {},
    }
    dataset_level_sample_stats = None

    # load data
    datahandler = DataHandler(
        config=config,
        downsample_method=args.downsample_method,
        subsample_method=args.subsample_method,
        n_train_data_samples=args.train_sample_num,
        preprocess_order=preprocess_order,
        single_label_only=args.single_label_only,
    )
    test_on_step1_full = parse_bool_arg(args.test_on_step1_full, default=False)
    external_test_handler = None
    if test_on_step1_full:
        external_test_config = copy.deepcopy(config)
        external_test_config.data.dataset_file = resolve_dataset_file("raw")
        external_test_handler = DataHandler(
            config=external_test_config,
            downsample_method=args.downsample_method,
            subsample_method="false",
            n_train_data_samples=args.train_sample_num,
            preprocess_order=preprocess_order,
            single_label_only=False,
        )
        print(
            "External test mode enabled: test split will use full step=1 windows "
            "from raw data."
        )

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold_idx, fold in enumerate(fold_ids, start=1):
        print(f"\n--- Fold {fold_idx}/{len(fold_ids)} | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()
        print_fold_data_overview(train, val, test, target_vals)
        split_ratio_pre_ds = datahandler.get_last_split_ratio_pre_ds()
        train_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("train"), train[1])
        val_ratio = _safe_ratio_matrix(split_ratio_pre_ds.get("val"), val[1])
        train_for_model = (train[0], train[1], train_ratio)
        val_for_model = (val[0], val[1], val_ratio)
        fold_sample_stats = None

        if enable_sample_stats and dataset_level_sample_stats is None:
            dataset_level_sample_stats = compute_dataset_label_ratio_stats(datahandler, target_vals)
            run_summary["sample_stats"]["dataset_level"] = dataset_level_sample_stats
            print_dataset_label_ratio_stats(dataset_level_sample_stats)

        if enable_sample_stats:
            include_val_in_train = parse_bool_arg(args.train_with_val, default=False)
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
            print_fold_classifier_input_stats(fold, fold_sample_stats, target_vals)

        try:

            fold_args = copy.deepcopy(args)
            optuna_result = None
            if int(getattr(args, "optuna_trials", 0)) > 0:
                print(
                    f"Running Optuna tuning for fold={fold} | model={fold_args.model} | "
                    f"trials={fold_args.optuna_trials}..."
                )
                optuna_result = tune_hyperparameters_for_fold(
                    args=fold_args,
                    target_vals=target_vals,
                    config=config,
                    preprocess_order=preprocess_order,
                    train_data=train_for_model,
                    val_data=val_for_model,
                    scaler=datahandler.scaler,
                )
                for k, v in optuna_result["best_params"].items():
                    setattr(fold_args, k, v)
                print(
                    f"Optuna best (fold={fold}): value={optuna_result['best_value']:.4f}, "
                    f"params={optuna_result['best_params']}"
                )

            model = build_model(fold_args, target_vals, config, preprocess_order)
            if hasattr(model, "set_input_scaler"):
                model.set_input_scaler(datahandler.scaler)
            print("Training model...")
            model.train(train_for_model, val_for_model)
            print("Evaluating model...")

            if external_test_handler is not None:
                external_test_handler.config.data.test_experiment_id = fold
                external_test_handler.config.data.validation_experiment_id = val_id
                _, _, external_test, _ = external_test_handler.get_data_loaders()

                ext_x = align_test_scale_with_train(
                    external_test[0],
                    source_scaler=external_test_handler.scaler,
                    target_scaler=datahandler.scaler,
                )
                eval_y_true = external_test[1]
                eval_y_pred = model.predict(ext_x)
                eval_y_prob = model.predict_proba(ext_x) if hasattr(model, "predict_proba") else None
                eval_timeline_x = sample_windows_for_timeline(ext_x, per_fold_timeline_windows)
                print(f"External test(step=1 full) samples: {len(eval_y_true)}")
            else:
                eval_y_true = test[1]
                eval_y_pred = model.predict(test[0])
                eval_y_prob = model.predict_proba(test[0]) if hasattr(model, "predict_proba") else None
                eval_timeline_x = test[0]

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

            plot_paths = plot_confusion_and_timeline(
                y_true=eval_y_true,
                y_pred=eval_y_pred,
                class_names=target_vals,
                fold_label=f"fold_{fold}",
                X_for_timeline=eval_timeline_x,
                save_dir=args.output,
                show_plots=show_images,
            )
            print(
                f"Saved fold plots: {plot_paths['confusion_path']} | "
                f"{plot_paths['timeline_path']} | {plot_paths.get('binary_confusion_path')}"
            )

            all_true.append(eval_y_true)
            all_pred.append(eval_y_pred)
            sampled_timeline_x = sample_windows_for_timeline(eval_timeline_x, per_fold_timeline_windows)
            if sampled_timeline_x is not None and len(sampled_timeline_x) > 0:
                all_test_x_timeline.append(sampled_timeline_x)

            test_mcc = fold_metrics["macro_mcc"]
            test_mccs.append(test_mcc)

            run_summary["folds"].append(
                {
                    "fold": int(fold),
                    "validation_experiment_id": int(val_id),
                    "train_samples": int(train[0].shape[0]) if train and train[0] is not None else 0,
                    "val_samples": int(val[0].shape[0]) if val and val[0] is not None else 0,
                    "test_samples": int(eval_y_true.shape[0]) if eval_y_true is not None else 0,
                    "metrics": fold_metrics,
                    "plots": plot_paths,
                    "selected_hparams": {
                        "max_depth": getattr(fold_args, "max_depth", None),
                        "colsample_bytree": getattr(fold_args, "colsample_bytree", None),
                        "num_leaves": getattr(fold_args, "num_leaves", None),
                        "subsample": getattr(fold_args, "subsample", None),
                        "min_child_samples": getattr(fold_args, "min_child_samples", None),
                        "min_data_in_leaf": getattr(fold_args, "min_data_in_leaf", None),
                    },
                    "optuna": optuna_result,
                    "sample_stats": fold_sample_stats if enable_sample_stats else None,
                }
            )


        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            raise e

        del train, val, test, model
        gc.collect()
        print("--- End of Fold ---")

    avg_mcc = sum(test_mccs) / len(test_mccs)

    if all_true and all_pred:
        overall_true = np.concatenate(all_true, axis=0)
        overall_pred = np.concatenate(all_pred, axis=0)
        overall_x = np.concatenate(all_test_x_timeline, axis=0) if all_test_x_timeline else None
        overall_plot_paths = plot_confusion_and_timeline(
            y_true=overall_true,
            y_pred=overall_pred,
            class_names=target_vals,
            fold_label="overall",
            X_for_timeline=overall_x,
            save_dir=args.output,
            show_plots=show_images,
        )
        print(
            f"Saved overall plots: "
            f"{overall_plot_paths['confusion_path']} | "
            f"{overall_plot_paths['timeline_path']} | "
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
    }
    if enable_sample_stats:
        run_summary["aggregate"]["sample_stats"] = {
            "enabled": True,
            "dataset_level_available": dataset_level_sample_stats is not None,
            "fold_stats_count": int(len(run_summary["folds"])),
        }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(run_summary), f, ensure_ascii=False, indent=2)
    print(f"Saved run summary: {summary_path}")

    print("Scores for each run: ", test_mccs)
    print(f"Macro-F1 per fold: {macro_f1s} | avg={np.mean(macro_f1s):.4f}")
    print(f"Macro PR-AUC per fold: {macro_pr_aucs} | avg={np.nanmean(macro_pr_aucs):.4f}")
    print(f"Macro Brier per fold: {macro_briers} | avg={np.mean(macro_briers):.4f}")
    print("\nTotal score:", avg_mcc)
