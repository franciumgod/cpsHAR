import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import matthews_corrcoef, multilabel_confusion_matrix
from sklearn.metrics import (
    matthews_corrcoef,
    multilabel_confusion_matrix,
    f1_score,
    recall_score,
    average_precision_score,
    brier_score_loss,
)

def plot_class_distribution(labels, label_names):
    counts = labels.sum(axis=0)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_names, counts, color='skyblue', edgecolor='black')
    plt.title("Class Distribution", fontsize=16)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def plot_per_class_confusion(y_true, y_pred, class_names):
    # Gibt ein Array der Form (N_classes, 2, 2) zurück
    cms = multilabel_confusion_matrix(y_true, y_pred)

    n_classes = len(class_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Anpassen je nach Klassenanzahl (hier 2x3 für 6 Klassen)
    axes = axes.flatten()

    for i, (cm, name) in enumerate(zip(cms, class_names)):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i], cbar=False,
                    xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
        axes[i].set_title(f"Class: {name}")
        axes[i].set_ylabel("True")
        axes[i].set_xlabel("Predicted")

    plt.tight_layout()
    plt.show()

def calculate_mcc_multilabel(y_true, y_pred):
    n_classes = y_true.shape[1]
    mcc_scores = []

    for i in range(n_classes):
        # MCC for each class treated as binary classification
        score = matthews_corrcoef(y_true[:, i], y_pred[:, i])
        mcc_scores.append(score)
    return np.mean(mcc_scores)

def _normalize_multilabel_proba(y_prob):
    """
    Normalize predict_proba output to shape (n_samples, n_classes).
    Handles:
    - ndarray: already (n_samples, n_classes) or (n_samples,)
    - list of ndarrays from sklearn multi-output estimators
    """
    if y_prob is None:
        return None

    if isinstance(y_prob, list):
        # e.g. [arr_cls0, arr_cls1, ...], each arr shape (n_samples, 2)
        cols = []
        for arr in y_prob:
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                cols.append(arr[:, 1])  # positive class prob
            else:
                cols.append(arr.ravel())
        return np.column_stack(cols)

    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        return y_prob.reshape(-1, 1)
    return y_prob


def evaluate_and_print_multilabel_metrics(y_true, y_pred, y_prob, class_names, fold_idx, split_name="Test"):
    """
    Prints:
    - Macro-F1
    - Macro PR-AUC
    - Macro Brier Score
    - Per-class recall (+ F1/PR-AUC/Brier for each class)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = _normalize_multilabel_proba(y_prob)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_prob is None:
        # fallback if model has no predict_proba
        y_prob = y_pred.astype(float)

    n_classes = y_true.shape[1]
    per_class = []

    for i in range(n_classes):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        y_s = y_prob[:, i]

        f1_i = f1_score(y_t, y_p, zero_division=0)
        rec_i = recall_score(y_t, y_p, zero_division=0)

        try:
            pr_auc_i = average_precision_score(y_t, y_s)
        except ValueError:
            # happens when a fold has no positive samples for this class
            pr_auc_i = np.nan

        brier_i = brier_score_loss(y_t, y_s)

        per_class.append(
            {
                "name": class_names[i] if i < len(class_names) else f"class_{i}",
                "f1": f1_i,
                "recall": rec_i,
                "pr_auc": pr_auc_i,
                "brier": brier_i,
            }
        )

    macro_f1 = float(np.mean([x["f1"] for x in per_class]))
    macro_recall = float(np.mean([x["recall"] for x in per_class]))
    macro_pr_auc = float(np.nanmean([x["pr_auc"] for x in per_class]))
    macro_brier = float(np.mean([x["brier"] for x in per_class]))

    print(f"\n[{split_name}] Fold {fold_idx} metrics")
    print(
        f"Macro-F1={macro_f1:.4f} | "
        f"Macro-Recall={macro_recall:.4f} | "
        f"Macro PR-AUC={macro_pr_auc:.4f} | "
        f"Macro Brier={macro_brier:.4f}"
    )
    print("Per-class metrics:")
    print(f"{'Class':20s} {'Recall':>8s} {'F1':>8s} {'PR-AUC':>8s} {'Brier':>8s}")
    for row in per_class:
        pr_auc_txt = f"{row['pr_auc']:.4f}" if not np.isnan(row["pr_auc"]) else "nan"
        print(
            f"{row['name'][:20]:20s} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:8.4f} "
            f"{pr_auc_txt:>8s} "
            f"{row['brier']:8.4f}"
        )

    return {
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_pr_auc": macro_pr_auc,
        "macro_brier": macro_brier,
        "per_class": per_class,
    }