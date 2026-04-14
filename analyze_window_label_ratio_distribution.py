from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import Config


def clean(df):
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore").reset_index(drop=True)


def build_superclass_labels(df, config, target_cols):
    df = clean(df)
    out = np.zeros((len(df), len(target_cols)), dtype=np.int8)
    for idx, superclass in enumerate(target_cols):
        children = [
            child
            for child, parent in config.data.superclass_mapping.items()
            if parent == superclass and child in df.columns
        ]
        if children:
            out[:, idx] = df[children].max(axis=1).to_numpy(dtype=np.int8, copy=False)
    return out


def summarize_distribution(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-window label ratio distributions for step-windowed datasets."
    )
    parser.add_argument("--steps", default="100,200,400,500")
    parser.add_argument("--raw", default="cps_data_multi_label.pkl")
    parser.add_argument("--window_size", type=int, default=4000)
    parser.add_argument("--output_dir", default="output/label_ratio_analysis")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config()
    raw_path = Path("data") / Path(args.raw).name
    raw_meta = pd.read_pickle(raw_path)
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    target_cols = sorted(list(set(config.data.superclass_mapping.values())))
    label_to_idx = {name: i for i, name in enumerate(target_cols)}

    print(f"Raw dataset: {raw_path}")
    print(f"Window size: {args.window_size}")
    print(f"Steps: {steps}")
    print(f"Labels: {target_cols}")

    label_cache = {}
    prefix_cache = {}
    for source_index, row in raw_meta.iterrows():
        label_mat = build_superclass_labels(row["data"], config, target_cols)
        label_cache[int(source_index)] = label_mat
        prefix = np.zeros((label_mat.shape[0] + 1, label_mat.shape[1]), dtype=np.int64)
        if len(label_mat) > 0:
            prefix[1:] = np.cumsum(label_mat, axis=0, dtype=np.int64)
        prefix_cache[int(source_index)] = prefix

    summary_rows = []
    hist_rows = []
    threshold_rows = []
    bins = np.linspace(0.0, 1.0, 51)

    driving_idx = [label_to_idx[k] for k in target_cols if k.startswith("Driving(")]
    lifting_idx = [label_to_idx[k] for k in target_cols if k.startswith("Lifting(")]

    for step in steps:
        step_ratio_blocks = []
        step_endpoint_blocks = []
        total_windows = 0

        for source_index, row in raw_meta.iterrows():
            source_index = int(source_index)
            labels = label_cache[source_index]
            n = len(labels)
            if n < args.window_size:
                continue

            starts = np.arange(0, n - args.window_size + 1, step, dtype=np.int64)
            if len(starts) == 0:
                continue

            prefix = prefix_cache[source_index]
            sums = prefix[starts + args.window_size] - prefix[starts]
            ratios = (sums.astype(np.float32) / float(args.window_size)).astype(np.float32, copy=False)
            endpoints = labels[starts + args.window_size - 1]

            step_ratio_blocks.append(ratios)
            step_endpoint_blocks.append(endpoints)
            total_windows += len(starts)

        if not step_ratio_blocks:
            print(f"[step={step}] no windows generated, skip.")
            continue

        ratio_all = np.vstack(step_ratio_blocks)
        endpoint_all = np.vstack(step_endpoint_blocks).astype(np.int8, copy=False)

        print(f"[step={step}] windows={total_windows}, ratio_shape={ratio_all.shape}")

        # Explain current threshold behavior (after single-label filtering) on endpoint labels
        active_counts = np.sum(endpoint_all, axis=1)
        single_mask = active_counts <= 1
        single_y = endpoint_all[single_mask]
        drive_ratio_current = np.mean(single_y[:, driving_idx], axis=1) if driving_idx else np.zeros(len(single_y))
        lift_ratio_current = np.mean(single_y[:, lifting_idx], axis=1) if lifting_idx else np.zeros(len(single_y))

        unique_drive = sorted(set(np.round(drive_ratio_current.astype(np.float32), 4)))
        unique_lift = sorted(set(np.round(lift_ratio_current.astype(np.float32), 4)))
        print(f"  current-filter driving ratio unique values: {unique_drive}")
        print(f"  current-filter lifting ratio unique values: {unique_lift}")

        for thr in [0.1, 0.2, 0.3]:
            keep_drive = (drive_ratio_current == 0) | (drive_ratio_current >= thr)
            keep_lift = (lift_ratio_current == 0) | (lift_ratio_current >= thr)
            keep = keep_drive & keep_lift
            threshold_rows.append(
                {
                    "step": step,
                    "single_label_samples": int(len(single_y)),
                    "threshold": thr,
                    "kept": int(np.sum(keep)),
                    "filtered": int(len(single_y) - np.sum(keep)),
                }
            )

        # Per-label distribution summary + histogram bins
        for label_idx, label_name in enumerate(target_cols):
            vals_all = ratio_all[:, label_idx]
            pos_mask = endpoint_all[:, label_idx] == 1
            vals_pos = ratio_all[pos_mask, label_idx]

            for scope_name, vals in [("all_windows", vals_all), ("endpoint_positive", vals_pos)]:
                stat = summarize_distribution(vals)
                summary_rows.append(
                    {
                        "step": step,
                        "label": label_name,
                        "scope": scope_name,
                        **stat,
                    }
                )

                counts, edges = np.histogram(vals, bins=bins)
                for bi in range(len(counts)):
                    hist_rows.append(
                        {
                            "step": step,
                            "label": label_name,
                            "scope": scope_name,
                            "bin_left": float(edges[bi]),
                            "bin_right": float(edges[bi + 1]),
                            "count": int(counts[bi]),
                        }
                    )

        # Visualization: one figure per step, 6 subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for label_idx, label_name in enumerate(target_cols):
            ax = axes.flat[label_idx]
            vals_all = ratio_all[:, label_idx]
            pos_mask = endpoint_all[:, label_idx] == 1
            vals_pos = ratio_all[pos_mask, label_idx]

            ax.hist(vals_all, bins=50, density=True, alpha=0.55, color="#4C72B0", label="all windows")
            if len(vals_pos) > 0:
                ax.hist(vals_pos, bins=50, density=True, alpha=0.55, color="#DD8452", label="endpoint=1")

            ax.set_title(label_name, fontsize=10)
            ax.set_xlim(0.0, 1.0)
            ax.set_xlabel("Label ratio within 2s window")
            ax.set_ylabel("Density")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"Step={step} | Label-ratio distribution within window", fontsize=12)
        fig_path = out_dir / f"label_ratio_distribution_step_{step}.png"
        fig.savefig(fig_path, dpi=160)
        plt.close(fig)
        print(f"  saved plot: {fig_path}")

    summary_df = pd.DataFrame(summary_rows)
    hist_df = pd.DataFrame(hist_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    summary_path = out_dir / "label_ratio_summary.csv"
    hist_path = out_dir / "label_ratio_hist_bins.csv"
    threshold_path = out_dir / "current_filter_threshold_check.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    hist_df.to_csv(hist_path, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_path, index=False, encoding="utf-8-sig")

    print("\nSaved files:")
    print(f"  - {summary_path}")
    print(f"  - {hist_path}")
    print(f"  - {threshold_path}")


if __name__ == "__main__":
    main()
