import random
import gc

import numpy as np
from utils.utils import (
    plot_per_class_confusion,
    evaluate_and_print_multilabel_metrics,
    plot_confusion_and_timeline,
)
from utils.config import Config
from data_handler import DataHandler
from models.RF import RandomForestClassifierSK
from models.XGB import XGBoostClassifierSK

if __name__ == '__main__':

    config = Config()

    # Seeding
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # if you use any other libraries that require seeding, set it here as well (e.g., torch.manual_seed(SEED) for PyTorch)
    # -> your results should be reproducible across runs with the same seed


    val_mccs = []
    test_mccs = []

    ###########################################################
    # 新增一些指标，多角度检查模型性能，后续可作为模型优化的综合指标
    # val_marco_f1s = []
    # test_marco_f1s = []
    # val_macro_pr_aucs = []
    # test_macro_pr_aucs = []
    # val_macro_briers = []
    # test_macro_briers = []
    macro_f1s = []
    macro_pr_aucs = []
    macro_briers = []
    all_true = []
    all_pred = []
    all_test_x = []

    lr_histories_by_fold = {}

    # load data
    datahandler = DataHandler(config=config)

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold in range(1, 5):
        print(f"\n--- Fold {fold}/4 | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        # validation hat to be different from test
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()

        # just to get an insight into the data
        plot_per_class_confusion(train[1], target_vals)
        plot_per_class_confusion(val[1], target_vals)
        plot_per_class_confusion(test[1], target_vals)

        try:

            #model = DummyClassifier(target_vals)
            # model = RandomForestClassifierSK(target_vals)
            ### INSERT YOUR MODEL HERE ###
            model = XGBoostClassifierSK(target_vals)
            # model = MyAwesomeModel(...)

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
                save_dir="outputs/plots",
            )
            print(f"Saved fold plots: {plot_paths['confusion_path']} | {plot_paths['timeline_path']}")

            all_true.append(test[1])
            all_pred.append(predicted_y)
            all_test_x.append(test[0])

            # optional, for more insight, plot the per-class-confusion-matrix for the test set
            # plot_per_class_confusion(test[1], predicted_y, target_vals)

            # Note: The MCC might be negative for a fold since its a correlation coefficient -> Range -1 to +1
            # +1 indicates perfect correlation, 0 indicates no correlation, and -1 indicates perfect inverse correlation.
            # Since we average over classes, it can happen that some classes have a
            # negative MCC while others have a positive MCC, resulting that scores balance each other out
            # Check your individual scores to see if they are reasonable
            test_mcc = fold_metrics["macro_mcc"]
            test_mccs.append(test_mcc)


        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            raise e

        del train, val, test, predicted_y, model
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
            save_dir="outputs/plots",
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
