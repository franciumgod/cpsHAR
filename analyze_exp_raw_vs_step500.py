import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import Config


def clean_raw_df(df):
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore").reset_index(drop=True)


def to_superclass_matrix(df, mapping, class_names):
    df = clean_raw_df(df)
    out = np.zeros((len(df), len(class_names)), dtype=np.int8)
    for i, cls in enumerate(class_names):
        children = [k for k, v in mapping.items() if v == cls and k in df.columns]
        if children:
            out[:, i] = df[children].max(axis=1).to_numpy(dtype=np.int8, copy=False)
    return out


def safe_ratio(num, den):
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def summarize_label_matrix(y, class_names):
    y = np.asarray(y, dtype=np.int8)
    total = int(len(y))
    pos_counts = y.sum(axis=0).astype(int)
    active = y.sum(axis=1)
    none_count = int(np.sum(active == 0))
    single_count = int(np.sum(active == 1))
    multi_count = int(np.sum(active > 1))

    return {
        "total_samples": total,
        "label_positive_counts": {class_names[i]: int(pos_counts[i]) for i in range(len(class_names))},
        "label_positive_ratio": {class_names[i]: safe_ratio(int(pos_counts[i]), total) for i in range(len(class_names))},
        "sample_composition_counts": {
            "none_label": none_count,
            "single_label": single_count,
            "multi_label": multi_count,
        },
        "sample_composition_ratio": {
            "none_label": safe_ratio(none_count, total),
            "single_label": safe_ratio(single_count, total),
            "multi_label": safe_ratio(multi_count, total),
        },
    }


def plot_per_exp_bars(per_exp_summary, class_names, key_path, title, ylabel, out_file):
    exp_ids = sorted(per_exp_summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.flatten()

    for idx, exp_id in enumerate(exp_ids):
        ax = axes[idx]
        cur = per_exp_summary[exp_id]
        for k in key_path:
            cur = cur[k]
        vals = [cur[c] for c in class_names]
        ax.bar(class_names, vals, color="#4C72B0")
        ax.set_title(f"Experiment {exp_id}")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)

    for j in range(len(exp_ids), len(axes)):
        axes[j].axis("off")
    fig.suptitle(title, fontsize=13)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def plot_compare_ratio(per_exp_summary, class_names, out_file):
    exp_ids = sorted(per_exp_summary.keys())
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    axes = axes.flatten()
    x = np.arange(len(class_names))
    w = 0.38

    for idx, exp_id in enumerate(exp_ids):
        ax = axes[idx]
        raw_ratio = [per_exp_summary[exp_id]["raw"]["label_positive_ratio"][c] for c in class_names]
        step_ratio = [per_exp_summary[exp_id]["step500"]["label_positive_ratio"][c] for c in class_names]
        ax.bar(x - w / 2, raw_ratio, width=w, label="raw(points)", color="#55A868")
        ax.bar(x + w / 2, step_ratio, width=w, label="step500(windows)", color="#C44E52")
        ax.set_title(f"Experiment {exp_id}")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=35)
        ax.set_ylabel("Positive ratio")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=8)

    for j in range(len(exp_ids), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Raw vs Step500 Label Positive Ratio by Experiment", fontsize=13)
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and summarize label distribution for raw data and step500 windowed samples."
    )
    parser.add_argument("--raw_file", default="cps_data_multi_label.pkl")
    parser.add_argument("--step500_file", default="cps_windows_2s_2000hz_step_500.pkl")
    parser.add_argument("--output_dir", default="output/exp_data_profile")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    class_names = sorted(list(set(config.data.superclass_mapping.values())))

    raw_path = Path("data") / Path(args.raw_file).name
    step_path = Path("data") / Path(args.step500_file).name
    raw_meta = pd.read_pickle(raw_path)
    step_payload = pd.read_pickle(step_path)

    if not isinstance(step_payload, dict) or step_payload.get("kind") != "window_samples":
        raise ValueError(f"{step_path} is not a window_samples payload.")

    y_step = np.asarray(step_payload["y"], dtype=np.int8)
    exp_step = np.asarray(step_payload["experiment"], dtype=np.int16)
    step_class_names = list(step_payload["label_cols"])
    if step_class_names != class_names:
        # align if order differs
        idx_map = [step_class_names.index(c) for c in class_names]
        y_step = y_step[:, idx_map]

    experiments = sorted(raw_meta["experiment"].unique().tolist())
    per_exp = {}

    for exp_id in experiments:
        exp_rows = raw_meta[raw_meta["experiment"] == exp_id]
        mats = []
        total_rows = int(len(exp_rows))
        for _, row in exp_rows.iterrows():
            mats.append(to_superclass_matrix(row["data"], config.data.superclass_mapping, class_names))
        y_raw = np.vstack(mats) if mats else np.zeros((0, len(class_names)), dtype=np.int8)
        raw_summary = summarize_label_matrix(y_raw, class_names)
        raw_summary["source_segments"] = total_rows

        mask = exp_step == exp_id
        y_exp_step = y_step[mask]
        step_summary = summarize_label_matrix(y_exp_step, class_names)

        per_exp[int(exp_id)] = {
            "raw": raw_summary,
            "step500": step_summary,
        }

    raw_count_plot = out_dir / "raw_points_label_count_by_exp.png"
    step_count_plot = out_dir / "step500_windows_label_count_by_exp.png"
    ratio_cmp_plot = out_dir / "raw_vs_step500_ratio_by_exp.png"

    plot_per_exp_bars(
        per_exp,
        class_names,
        key_path=["raw", "label_positive_counts"],
        title="Raw Data: Positive Label Count by Experiment",
        ylabel="Count (points)",
        out_file=raw_count_plot,
    )
    plot_per_exp_bars(
        per_exp,
        class_names,
        key_path=["step500", "label_positive_counts"],
        title="Step=500 Window Samples: Positive Label Count by Experiment",
        ylabel="Count (windows)",
        out_file=step_count_plot,
    )
    plot_compare_ratio(per_exp, class_names, ratio_cmp_plot)

    overall_raw = summarize_label_matrix(
        np.vstack(
            [
                to_superclass_matrix(row["data"], config.data.superclass_mapping, class_names)
                for _, row in raw_meta.iterrows()
            ]
        ),
        class_names,
    )
    overall_step = summarize_label_matrix(y_step, class_names)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "raw_file": str(raw_path),
        "step500_file": str(step_path),
        "class_names": class_names,
        "experiments": per_exp,
        "overall": {
            "raw": overall_raw,
            "step500": overall_step,
        },
        "plots": {
            "raw_count_plot": str(raw_count_plot),
            "step500_count_plot": str(step_count_plot),
            "ratio_compare_plot": str(ratio_cmp_plot),
        },
    }

    report_path = out_dir / "exp_raw_vs_step500_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Saved: {raw_count_plot}")
    print(f"Saved: {step_count_plot}")
    print(f"Saved: {ratio_cmp_plot}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
