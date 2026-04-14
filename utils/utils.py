import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import math
from sklearn.metrics import confusion_matrix
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


def plot_per_class_binary_confusions(
    y_true,
    y_pred,
    class_names,
    fold_label,
    save_dir="outputs/train_100W_without_val",
    show_plots=True,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    cms = multilabel_confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    if n_classes == 0:
        return None

    n_cols = min(3, max(1, n_classes))
    n_rows = int(math.ceil(n_classes / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.2 * n_rows))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = np.asarray([axes])

    for i, (cm, name) in enumerate(zip(cms, class_names)):
        ax = axes[i]
        tn, fp, fn, tp = cm.ravel()
        total = max(1, int(cm.sum()))

        ann = np.array(
            [
                [f"TN\n{tn}\n{tn / total:.1%}", f"FP\n{fp}\n{fp / total:.1%}"],
                [f"FN\n{fn}\n{fn / total:.1%}", f"TP\n{tp}\n{tp / total:.1%}"],
            ]
        )

        sns.heatmap(
            cm,
            annot=ann,
            fmt="",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    out_file = save_path / f"binary_confusion_{fold_label}.png"
    fig.savefig(out_file, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)

    return str(out_file)

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
    - Macro-MCC
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

        mcc_i = matthews_corrcoef(y_t, y_p)
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
                "mcc": mcc_i,
                "f1": f1_i,
                "recall": rec_i,
                "pr_auc": pr_auc_i,
                "brier": brier_i,
            }
        )

    macro_mcc = float(calculate_mcc_multilabel(y_true, y_pred))
    macro_f1 = float(np.mean([x["f1"] for x in per_class]))
    macro_recall = float(np.mean([x["recall"] for x in per_class]))
    macro_pr_auc = float(np.nanmean([x["pr_auc"] for x in per_class]))
    macro_brier = float(np.mean([x["brier"] for x in per_class]))

    print(f"\n[{split_name}] Fold {fold_idx} metrics")
    print(
        f"Macro-MCC={macro_mcc:.4f} | "
        f"Macro-F1={macro_f1:.4f} | "
        f"Macro-Recall={macro_recall:.4f} | "
        f"Macro PR-AUC={macro_pr_auc:.4f} | "
        f"Macro Brier={macro_brier:.4f}"
    )
    print("Per-class metrics:")
    print(f"{'Class':20s} {'MCC':>8s} {'Recall':>8s} {'F1':>8s} {'PR-AUC':>8s} {'Brier':>8s}")
    for row in per_class:
        pr_auc_txt = f"{row['pr_auc']:.4f}" if not np.isnan(row["pr_auc"]) else "nan"
        print(
            f"{row['name'][:20]:20s} "
            f"{row['mcc']:8.4f} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:8.4f} "
            f"{pr_auc_txt:>8s} "
            f"{row['brier']:8.4f}"
        )

    return {
        "macro_mcc": macro_mcc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "macro_pr_auc": macro_pr_auc,
        "macro_brier": macro_brier,
        "per_class": per_class,
    }


def _multilabel_to_single_index(y):
    y = np.asarray(y)
    if y.ndim == 1:
        return y.astype(int)
    return np.argmax(y, axis=1).astype(int)


def _extract_plot_signal_from_windows(X):
    X = np.asarray(X)
    if X.ndim != 3:
        return np.arange(len(X)), np.asarray(X).reshape(-1)

    # Support both (n, time, ch) and (n, ch, time)
    if X.shape[1] >= X.shape[2]:
        # (n, time, ch): use last timestep of first channel
        signal = X[:, -1, 0]
    else:
        # (n, ch, time): use first channel last timestep
        signal = X[:, 0, -1]

    t = np.arange(len(signal))
    return t, signal


def plot_confusion_and_timeline(
    y_true,
    y_pred,
    class_names,
    fold_label,
    X_for_timeline=None,
    save_dir="outputs/train_100W_without_val",
    max_timeline_points=8000,
    show_plots=True,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_true_idx = _multilabel_to_single_index(y_true_arr)
    y_pred_idx = _multilabel_to_single_index(y_pred_arr)
    n_classes = len(class_names)
    labels = list(range(n_classes))

    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig_cm, ax_cm = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_cm,
    )
    ax_cm.set_xlabel("Predicted class")
    ax_cm.set_ylabel("True class")
    ax_cm.set_title(f"Normalized Confusion Matrix ({fold_label})")
    plt.tight_layout()
    cm_file = save_path / f"confusion_{fold_label}.png"
    fig_cm.savefig(cm_file, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig_cm)

    # Timeline style plot (true vs predicted)
    if X_for_timeline is None:
        t = np.arange(len(y_true_idx))
        signal = np.zeros_like(t, dtype=float)
    else:
        t, signal = _extract_plot_signal_from_windows(X_for_timeline)

    n = min(len(t), len(y_true_idx), len(y_pred_idx))
    t = t[:n]
    signal = signal[:n]
    y_true_idx = y_true_idx[:n]
    y_pred_idx = y_pred_idx[:n]

    if n > max_timeline_points:
        idx = np.linspace(0, n - 1, max_timeline_points, dtype=int)
        t = t[idx]
        signal = signal[idx]
        y_true_idx = y_true_idx[idx]
        y_pred_idx = y_pred_idx[idx]

    cmap = plt.get_cmap("tab10", max(10, n_classes))
    class_colors = [cmap(i) for i in range(n_classes)]

    fig_tl, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for cls in range(n_classes):
        m_true = y_true_idx == cls
        m_pred = y_pred_idx == cls
        if np.any(m_true):
            axes[0].scatter(t[m_true], signal[m_true], s=4, color=class_colors[cls], label=class_names[cls], alpha=0.9)
        if np.any(m_pred):
            axes[1].scatter(t[m_pred], signal[m_pred], s=4, color=class_colors[cls], label=class_names[cls], alpha=0.9)

    axes[0].set_title(f"TRUE class timeline ({fold_label})")
    axes[1].set_title(f"PREDICTED class timeline ({fold_label})")
    axes[0].set_ylabel("Signal")
    axes[1].set_ylabel("Signal")
    axes[1].set_xlabel("Sample index")

    handles, labels_text = axes[1].get_legend_handles_labels()
    uniq = dict(zip(labels_text, handles))
    axes[1].legend(uniq.values(), uniq.keys(), loc="upper right", fontsize=8)

    plt.tight_layout()
    tl_file = save_path / f"timeline_{fold_label}.png"
    fig_tl.savefig(tl_file, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig_tl)

    binary_cm_file = plot_per_class_binary_confusions(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        class_names=class_names,
        fold_label=fold_label,
        save_dir=save_dir,
        show_plots=show_plots,
    )

    return {
        "confusion_path": str(cm_file),
        "timeline_path": str(tl_file),
        "binary_confusion_path": binary_cm_file,
    }
