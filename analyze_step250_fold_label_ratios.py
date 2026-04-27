from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_PATH = Path("data") / "cps_windows_2s_2000hz_step_250.pkl"
OUT_DIR = Path("output") / "step250_fold_label_ratio_analysis"


LABEL_ORDER = [
    "Driving(curve)",
    "Driving(straight)",
    "Lifting(lowering)",
    "Lifting(raising)",
    "Stationary processes",
    "Turntable wrapping",
    "Driving(curve) + Lifting(lowering)",
    "Driving(curve) + Lifting(raising)",
    "Driving(straight) + Lifting(lowering)",
    "Driving(straight) + Lifting(raising)",
]


def state_name_from_vector(vec: np.ndarray, label_cols: List[str]) -> str:
    active = [label_cols[i] for i, v in enumerate(vec) if int(v) == 1]
    return " + ".join(active) if active else "0-label"


def fold_split_map(fold: int) -> Dict[str, List[int]]:
    val_id = fold + 1 if fold < 4 else 1
    test_id = fold
    train_ids = [exp for exp in [1, 2, 3, 4] if exp not in {test_id, val_id}]
    return {
        "train": train_ids,
        "val": [val_id],
        "test": [test_id],
    }


def build_counts_df(payload: Dict) -> pd.DataFrame:
    label_cols = list(payload["label_cols"])
    experiments = np.asarray(payload["experiment"], dtype=np.int64)
    y = np.asarray(payload["y"], dtype=np.int8)
    states = [state_name_from_vector(row, label_cols) for row in y]

    rows = []
    all_states = LABEL_ORDER.copy()
    if "0-label" in states and "0-label" not in all_states:
        all_states.append("0-label")

    for fold in [1, 2, 3, 4]:
        split_cfg = fold_split_map(fold)
        for split_name, split_exps in split_cfg.items():
            mask = np.isin(experiments, split_exps)
            split_states = [states[i] for i in np.flatnonzero(mask)]
            total = len(split_states)
            counts = pd.Series(split_states).value_counts() if total > 0 else pd.Series(dtype=np.int64)
            for label in all_states:
                count = int(counts.get(label, 0))
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "experiments": ",".join(map(str, split_exps)),
                        "label": label,
                        "count": count,
                        "ratio": (count / total) if total > 0 else 0.0,
                        "total_samples": total,
                    }
                )

    return pd.DataFrame(rows)


def build_exp_counts_df(payload: Dict) -> pd.DataFrame:
    label_cols = list(payload["label_cols"])
    experiments = np.asarray(payload["experiment"], dtype=np.int64)
    y = np.asarray(payload["y"], dtype=np.int8)
    states = [state_name_from_vector(row, label_cols) for row in y]

    rows = []
    all_states = LABEL_ORDER.copy()
    if "0-label" in states and "0-label" not in all_states:
        all_states.append("0-label")

    for exp in [1, 2, 3, 4]:
        exp_states = [states[i] for i in np.flatnonzero(experiments == exp)]
        total = len(exp_states)
        counts = pd.Series(exp_states).value_counts() if total > 0 else pd.Series(dtype=np.int64)
        for label in all_states:
            count = int(counts.get(label, 0))
            rows.append(
                {
                    "experiment": exp,
                    "label": label,
                    "count": count,
                    "ratio": (count / total) if total > 0 else 0.0,
                    "total_samples": total,
                }
            )
    return pd.DataFrame(rows)


def plot_fold_ratio_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    for split_name in ["train", "test"]:
        sub = df[df["split"] == split_name].copy()
        mat = (
            sub.pivot(index="label", columns="fold", values="ratio")
            .reindex(LABEL_ORDER)
            .fillna(0.0)
        )
        count_mat = (
            sub.pivot(index="label", columns="fold", values="count")
            .reindex(LABEL_ORDER)
            .fillna(0)
        )

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        arr = mat.to_numpy(dtype=np.float64)
        im = ax.imshow(arr, cmap="Blues", vmin=0.0, vmax=max(0.01, float(arr.max())))
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ratio = arr[i, j]
                count = int(count_mat.iloc[i, j])
                txt = f"{ratio*100:.2f}%\n({count})"
                color = "white" if ratio > 0.45 * arr.max() and arr.max() > 0 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"fold_{k}" for k in [1, 2, 3, 4]])
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, fontsize=9)
        ax.set_xlabel("Fold")
        ax.set_ylabel("Exact label state")
        ax.set_title(f"step=250 {split_name} label ratio by fold")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="ratio")
        fig.tight_layout()
        fig.savefig(out_dir / f"{split_name}_ratio_heatmap.png", dpi=180)
        plt.close(fig)


def plot_curve_multilabel_focus(df: pd.DataFrame, out_dir: Path) -> None:
    focus_labels = [
        "Driving(curve) + Lifting(lowering)",
        "Driving(curve) + Lifting(raising)",
    ]
    sub = df[(df["split"].isin(["train", "test"])) & (df["label"].isin(focus_labels))].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=False)

    for ax, metric in zip(axes, ["count", "ratio"]):
        width = 0.18
        x = np.arange(4, dtype=float)
        offsets = {
            ("train", focus_labels[0]): -1.5 * width,
            ("train", focus_labels[1]): -0.5 * width,
            ("test", focus_labels[0]): 0.5 * width,
            ("test", focus_labels[1]): 1.5 * width,
        }
        colors = {
            ("train", focus_labels[0]): "#1f77b4",
            ("train", focus_labels[1]): "#4c78a8",
            ("test", focus_labels[0]): "#d62728",
            ("test", focus_labels[1]): "#f58518",
        }
        for split_name in ["train", "test"]:
            for label in focus_labels:
                vals = []
                for fold in [1, 2, 3, 4]:
                    row = sub[(sub["fold"] == fold) & (sub["split"] == split_name) & (sub["label"] == label)].iloc[0]
                    vals.append(float(row[metric]))
                ax.bar(
                    x + offsets[(split_name, label)],
                    vals,
                    width=width,
                    color=colors[(split_name, label)],
                    label=f"{split_name} | {label}",
                )
        ax.set_xticks(x)
        ax.set_xticklabels([f"fold_{k}" for k in [1, 2, 3, 4]])
        ax.set_title(f"curve multi-label {metric} by fold")
        ax.grid(axis="y", alpha=0.25)
        if metric == "ratio":
            ax.set_ylabel("ratio")
        else:
            ax.set_ylabel("count")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(out_dir / "curve_multilabel_focus.png", dpi=180)
    plt.close(fig)


def write_report(
    fold_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    out_path: Path,
) -> None:
    def table_md(df: pd.DataFrame) -> str:
        cols = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for row in df.itertuples(index=False, name=None):
            vals = [str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    exp_pivot = (
        exp_df.pivot(index="label", columns="experiment", values="count")
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
    )
    exp_ratio_pivot = (
        exp_df.pivot(index="label", columns="experiment", values="ratio")
        .reindex(LABEL_ORDER)
        .fillna(0.0)
    )
    train_pivot = (
        fold_df[fold_df["split"] == "train"]
        .pivot(index="label", columns="fold", values="count")
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
    )
    test_pivot = (
        fold_df[fold_df["split"] == "test"]
        .pivot(index="label", columns="fold", values="count")
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
    )
    train_ratio_pivot = (
        fold_df[fold_df["split"] == "train"]
        .pivot(index="label", columns="fold", values="ratio")
        .reindex(LABEL_ORDER)
        .fillna(0.0)
        .mul(100)
        .round(3)
    )
    test_ratio_pivot = (
        fold_df[fold_df["split"] == "test"]
        .pivot(index="label", columns="fold", values="ratio")
        .reindex(LABEL_ORDER)
        .fillna(0.0)
        .mul(100)
        .round(3)
    )

    curve_lower_test = fold_df[
        (fold_df["split"] == "test") & (fold_df["label"] == "Driving(curve) + Lifting(lowering)")
    ][["fold", "experiments", "count", "ratio", "total_samples"]].copy()
    curve_raise_test = fold_df[
        (fold_df["split"] == "test") & (fold_df["label"] == "Driving(curve) + Lifting(raising)")
    ][["fold", "experiments", "count", "ratio", "total_samples"]].copy()
    curve_lower_test["ratio"] = (curve_lower_test["ratio"] * 100).round(4)
    curve_raise_test["ratio"] = (curve_raise_test["ratio"] * 100).round(4)

    report = f"""# step=250 Fold Label Ratio Analysis

## 结论

1. 你怀疑的方向是对的，`Driving(curve) + Lifting(lowering)` 和 `Driving(curve) + Lifting(raising)` 在 step=250 窗口里本来就很少，而且在不同 `exp`/fold 之间分布很不均衡。
2. `Driving(curve) + Lifting(raising)` 最极端：`exp3` 的测试集里是 **0 个样本**。这会直接解释你给的 `fold_3` 混淆矩阵里这一行整行是 0 的现象，本质上不是“模型完全不会”，而是 `fold_3 test` 根本没有这个真值类别。
3. `Driving(curve) + Lifting(lowering)` 虽然不是 0，但在各 fold 的测试集里也都偏少，尤其和单标签 `Driving(curve)` 相比差了一个数量级，因此它非常容易被模型吸到 `Driving(curve)` 上。
4. 训练集里这两个 `curve` 多标签也不稳定。由于 leave-one-exp-out 下 train 只来自两个 `exp`，一旦某个多标签主要集中在被留作 `val/test` 的 `exp`，该 fold 的 train 支持就会进一步变弱。

## fold 划分规则

- `fold_1`: test=`exp1`, val=`exp2`, train=`exp3+exp4`
- `fold_2`: test=`exp2`, val=`exp3`, train=`exp1+exp4`
- `fold_3`: test=`exp3`, val=`exp4`, train=`exp1+exp2`
- `fold_4`: test=`exp4`, val=`exp1`, train=`exp2+exp3`

## 各 experiment 原始样本计数

{table_md(exp_pivot.reset_index())}

## 各 experiment 原始样本比例（%）

{table_md(exp_ratio_pivot.mul(100).round(3).reset_index())}

## 各 fold 训练集计数

{table_md(train_pivot.reset_index())}

## 各 fold 训练集比例（%）

{table_md(train_ratio_pivot.reset_index())}

## 各 fold 测试集计数

{table_md(test_pivot.reset_index())}

## 各 fold 测试集比例（%）

{table_md(test_ratio_pivot.reset_index())}

## 两个 curve 多标签在测试集里的支持度

### `Driving(curve) + Lifting(lowering)`

{table_md(curve_lower_test)}

### `Driving(curve) + Lifting(raising)`

{table_md(curve_raise_test)}

## 对你这张 `fold_3` 混淆矩阵的直接解释

1. `fold_3` 的测试集就是 `exp3`。
2. `exp3` 里 `Driving(curve) + Lifting(raising)` 是 0 个，所以混淆矩阵这一行没有有效支持。
3. `exp3` 里 `Driving(curve) + Lifting(lowering)` 也只有很少的样本，因此它被压到 `Driving(curve)` 上是符合样本分布的。
4. 这两个 `curve` 多标签和单标签 `Driving(curve)` 在时序上又强相关，所以当类别先验本来就弱时，模型会优先学到更稳定、更高占比的 `Driving(curve)`。

## 图

- `train_ratio_heatmap.png`: 4 个 fold 训练集的 exact label ratio 热力图，单元格内同时标了比例和计数。
- `test_ratio_heatmap.png`: 4 个 fold 测试集的 exact label ratio 热力图。
- `curve_multilabel_focus.png`: 两个 `curve` 多标签在 train/test 中的计数和比例对比。
"""
    out_path.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = pd.read_pickle(DATA_PATH)
    fold_df = build_counts_df(payload)
    exp_df = build_exp_counts_df(payload)

    fold_df.to_csv(OUT_DIR / "fold_label_counts.csv", index=False, encoding="utf-8-sig")
    exp_df.to_csv(OUT_DIR / "experiment_label_counts.csv", index=False, encoding="utf-8-sig")

    plot_fold_ratio_heatmaps(fold_df, OUT_DIR)
    plot_curve_multilabel_focus(fold_df, OUT_DIR)
    write_report(fold_df, exp_df, OUT_DIR / "REPORT.md")
    print(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
