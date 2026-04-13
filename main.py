import random
import gc
import sys
import numpy as np
from utils.utils import (
    evaluate_and_print_multilabel_metrics,
    plot_confusion_and_timeline,
)
from utils.config import Config
from data_handler import DataHandler
from models.XGB import XGBoostClassifierSK
from models.LGBM import LightGBMClassifierSK
import warnings
warnings.filterwarnings("ignore")


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


def detect_preprocess_order(argv_tokens):
    subsample_flags = {"--subsample_method", "--train_sample_num"}
    downsample_flag = "--downsample_method"

    subsample_pos = min(
        [idx for idx, token in enumerate(argv_tokens) if token in subsample_flags],
        default=None,
    )
    downsample_pos = next(
        (idx for idx, token in enumerate(argv_tokens) if token == downsample_flag),
        None,
    )

    if subsample_pos is not None and downsample_pos is not None:
        if subsample_pos < downsample_pos:
            return "subsample_first"
        return "downsample_first"
    return "downsample_first"


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

def build_model(args, target_vals, config, preprocess_order):
    model_name = str(args.model).lower() if args.model else "xgboost"
    model_downsample_method = args.downsample_method if preprocess_order == "subsample_first" else "false"

    if model_name == "xgboost":
        return XGBoostClassifierSK(
            target_vals,
            use_val_in_train=parse_bool_arg(args.train_with_val, default=False),
            use_gpu=parse_bool_arg(args.use_gpu, default=False),
            subsample_method=args.subsample_method,
            n_train_data_samples=args.train_sample_num,
            max_depth=getattr(args, "max_depth", 6),
            colsample_bytree=getattr(args, "colsample_bytree", 0.8),
            pre_downsample_method=model_downsample_method,
            pre_downsample_factor=config.prep.ds_factor,
        )

    if model_name == "lightgbm":
        return LightGBMClassifierSK(
            target_vals,
            use_val_in_train=parse_bool_arg(args.train_with_val, default=False),
            subsample_method=args.subsample_method,
            n_train_data_samples=args.train_sample_num,
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
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Training and evaluation pipeline for multi-label activity recognition."
    )

    parser.add_argument(
        "--data",
        default="raw",
        help="Dataset selector: raw (default), or a step number like 100/200/400/500, or a custom filename under data/."
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
        "--balance",
        default=False,
        help="Balance class distribution in the training set."
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
        default=False,
        help="Whether to use the specific signal combinations for training (default: False)."
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
    preprocess_order = detect_preprocess_order(sys.argv[1:])
    print(f"Preprocess order: {preprocess_order}")

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    val_mccs = []
    test_mccs = []

    fold_ids = parse_fold_list(args.folds)
    macro_f1s = []
    macro_pr_aucs = []
    macro_briers = []
    all_true = []
    all_pred = []
    all_test_x = []

    lr_histories_by_fold = {}

    # load data
    datahandler_downsample_method = (
        args.downsample_method if preprocess_order == "downsample_first" else "false"
    )
    datahandler = DataHandler(config=config, downsample_method=datahandler_downsample_method)

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold_idx, fold in enumerate(fold_ids, start=1):
        print(f"\n--- Fold {fold_idx}/{len(fold_ids)} | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()

        try:

            model = build_model(args, target_vals, config, preprocess_order)
            print("Training model...")
            model.train(train, val)
            print("Evaluating model...")
            predicted_y = model.predict(test[0])

            predicted_prob = model.predict_proba(test[0]) if hasattr(model, "predict_proba") else None

            fold_metrics = evaluate_and_print_multilabel_metrics(
                y_true=test[1],
                y_pred=predicted_y,
                y_prob=predicted_prob,
                class_names=target_vals,
                fold_idx=fold,
                split_name="Test",
            )

            macro_f1s.append(fold_metrics["macro_f1"])
            macro_pr_aucs.append(fold_metrics["macro_pr_auc"])
            macro_briers.append(fold_metrics["macro_brier"])

            plot_paths = plot_confusion_and_timeline(
                y_true=test[1],
                y_pred=predicted_y,
                class_names=target_vals,
                fold_label=f"fold_{fold}",
                X_for_timeline=test[0],
                save_dir=args.output,
                show_plots=show_images,
            )
            print(f"Saved fold plots: {plot_paths['confusion_path']} | {plot_paths['timeline_path']}")

            all_true.append(test[1])
            all_pred.append(predicted_y)
            all_test_x.append(test[0])

            test_mcc = fold_metrics["macro_mcc"]
            test_mccs.append(test_mcc)


        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            raise e

        del train, val, test, predicted_y, predicted_prob, model
        gc.collect()
        print("--- End of Fold ---")

    avg_mcc = sum(test_mccs) / len(test_mccs)

    if all_true and all_pred:
        overall_true = np.concatenate(all_true, axis=0)
        overall_pred = np.concatenate(all_pred, axis=0)
        overall_x = np.concatenate(all_test_x, axis=0) if all_test_x else None
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
            f"{overall_plot_paths['confusion_path']} | {overall_plot_paths['timeline_path']}"
        )

    print("Scores for each run: ", test_mccs)
    print(f"Macro-F1 per fold: {macro_f1s} | avg={np.mean(macro_f1s):.4f}")
    print(f"Macro PR-AUC per fold: {macro_pr_aucs} | avg={np.nanmean(macro_pr_aucs):.4f}")
    print(f"Macro Brier per fold: {macro_briers} | avg={np.mean(macro_briers):.4f}")
    print("\nTotal score:", avg_mcc)
