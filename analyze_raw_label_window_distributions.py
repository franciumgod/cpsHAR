import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import Config


@dataclass
class StepAnalysisResult:
    step: int
    combo_counts: np.ndarray
    combo_time_counts: np.ndarray
    class_prev_counts: np.ndarray
    class_next_counts: np.ndarray
    class_run_count: np.ndarray
    class_run_duration_sum: np.ndarray
    source_zero_counts_by_bin: Dict[int, np.ndarray]
    source_total_counts_by_bin: Dict[int, np.ndarray]
    source_total_windows: Dict[int, int]
    source_zero_windows: Dict[int, int]


def combo_id_from_labels(labels_2d: np.ndarray) -> np.ndarray:
    n_labels = labels_2d.shape[1]
    weights = (1 << np.arange(n_labels, dtype=np.int64))
    return labels_2d.astype(np.int64, copy=False) @ weights


def combo_id_to_name(combo_id: int, label_cols: List[str]) -> str:
    if combo_id == 0:
        return "NONE(0-label)"
    active = []
    for i, label in enumerate(label_cols):
        if (combo_id >> i) & 1:
            active.append(label)
    return " + ".join(active)


def make_plot_zero_ratio_by_position(
    out_path: Path,
    result: StepAnalysisResult,
    n_time_bins: int,
) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(n_time_bins)
    x_pct = (x + 0.5) / n_time_bins * 100.0

    overall_zero = np.zeros(n_time_bins, dtype=np.int64)
    overall_total = np.zeros(n_time_bins, dtype=np.int64)

    for source_idx in sorted(result.source_total_counts_by_bin.keys()):
        total = result.source_total_counts_by_bin[source_idx]
        zero = result.source_zero_counts_by_bin[source_idx]
        overall_zero += zero
        overall_total += total
        ratio = np.divide(zero, total, out=np.zeros_like(zero, dtype=np.float64), where=total > 0)
        plt.plot(x_pct, ratio, marker="o", linewidth=1.2, alpha=0.8, label=f"source_{source_idx}")

    overall_ratio = np.divide(
        overall_zero,
        overall_total,
        out=np.zeros_like(overall_zero, dtype=np.float64),
        where=overall_total > 0,
    )
    plt.plot(x_pct, overall_ratio, color="black", linewidth=2.2, label="overall")

    plt.title(f"Zero-label Window Ratio by Timeline Position (step={result.step})")
    plt.xlabel("Normalized timeline position (%)")
    plt.ylabel("Zero-label ratio")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_plot_combo_time_heatmap(
    out_path: Path,
    result: StepAnalysisResult,
    label_cols: List[str],
    n_time_bins: int,
    top_k: int = 12,
) -> None:
    combo_counts = result.combo_counts.copy()
    nonzero_ids = np.flatnonzero(combo_counts > 0)
    if len(nonzero_ids) == 0:
        return

    order = nonzero_ids[np.argsort(combo_counts[nonzero_ids])[::-1]]
    top_ids = order[:top_k]

    mat = result.combo_time_counts[top_ids].astype(np.float64)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

    y_labels = [combo_id_to_name(int(cid), label_cols) for cid in top_ids]
    x_labels = [f"{int(i * 100 / n_time_bins)}-{int((i + 1) * 100 / n_time_bins)}%" for i in range(n_time_bins)]

    plt.figure(figsize=(14, max(5, 0.45 * len(top_ids))))
    im = plt.imshow(mat, aspect="auto", cmap="YlGnBu")
    plt.colorbar(im, fraction=0.02, pad=0.01, label="Within-combo timeline share")
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=8)
    plt.xticks(np.arange(n_time_bins), x_labels, rotation=45, ha="right", fontsize=8)
    plt.title(f"Top Label-combo Timeline Distribution Heatmap (step={result.step})")
    plt.xlabel("Normalized timeline bin")
    plt.ylabel("Label combo")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_plot_class_context_heatmap(
    out_path: Path,
    counts_2d: np.ndarray,
    label_cols: List[str],
    context_labels: List[str],
    step: int,
    title_prefix: str,
) -> None:
    mat = counts_2d.astype(np.float64)
    row_sum = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)

    plt.figure(figsize=(max(9, 0.6 * len(context_labels)), 6))
    im = plt.imshow(mat, aspect="auto", cmap="OrRd")
    plt.colorbar(im, fraction=0.02, pad=0.01, label="Probability")
    plt.yticks(np.arange(len(label_cols)), label_cols, fontsize=8)
    plt.xticks(np.arange(len(context_labels)), context_labels, rotation=45, ha="right", fontsize=8)
    plt.title(f"{title_prefix} Context Distribution by Class (step={step})")
    plt.xlabel("Context combo")
    plt.ylabel("Target class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def analyze_step(
    raw_meta: pd.DataFrame,
    label_cols: List[str],
    window_size: int,
    step: int,
    n_time_bins: int,
) -> StepAnalysisResult:
    n_labels = len(label_cols)
    n_combo = 1 << n_labels
    start_token = n_combo
    end_token = n_combo + 1
    n_context_combo = n_combo + 2

    combo_counts = np.zeros(n_combo, dtype=np.int64)
    combo_time_counts = np.zeros((n_combo, n_time_bins), dtype=np.int64)
    class_prev_counts = np.zeros((n_labels, n_context_combo), dtype=np.int64)
    class_next_counts = np.zeros((n_labels, n_context_combo), dtype=np.int64)
    class_run_count = np.zeros(n_labels, dtype=np.int64)
    class_run_duration_sum = np.zeros(n_labels, dtype=np.int64)

    source_zero_counts_by_bin: Dict[int, np.ndarray] = {}
    source_total_counts_by_bin: Dict[int, np.ndarray] = {}
    source_total_windows: Dict[int, int] = {}
    source_zero_windows: Dict[int, int] = {}

    for source_idx, row in raw_meta.iterrows():
        source_df = row["data"]
        labels = source_df[label_cols].to_numpy(dtype=np.int8, copy=False)
        n = len(labels)
        if n < window_size:
            continue

        starts = np.arange(0, n - window_size + 1, step, dtype=np.int64)
        endpoint_indices = starts + (window_size - 1)
        endpoint_labels = labels[endpoint_indices]
        combo_ids = combo_id_from_labels(endpoint_labels)

        combo_counts += np.bincount(combo_ids, minlength=n_combo)

        denom = max(1, n - window_size)
        pos_norm = starts / float(denom)
        bin_idx = np.minimum((pos_norm * n_time_bins).astype(np.int64), n_time_bins - 1)
        np.add.at(combo_time_counts, (combo_ids, bin_idx), 1)

        total_by_bin = np.bincount(bin_idx, minlength=n_time_bins)
        zero_mask = combo_ids == 0
        zero_by_bin = np.bincount(bin_idx[zero_mask], minlength=n_time_bins)

        source_total_counts_by_bin[int(source_idx)] = total_by_bin
        source_zero_counts_by_bin[int(source_idx)] = zero_by_bin
        source_total_windows[int(source_idx)] = int(len(combo_ids))
        source_zero_windows[int(source_idx)] = int(np.sum(zero_mask))

        for class_idx in range(n_labels):
            active = endpoint_labels[:, class_idx] == 1
            if not np.any(active):
                continue

            prev_active = np.concatenate(([False], active[:-1]))
            next_active = np.concatenate((active[1:], [False]))
            run_starts = np.flatnonzero(active & ~prev_active)
            run_ends = np.flatnonzero(active & ~next_active)
            if len(run_starts) == 0:
                continue

            class_run_count[class_idx] += len(run_starts)
            class_run_duration_sum[class_idx] += np.sum(run_ends - run_starts + 1, dtype=np.int64)

            prev_combo = np.where(run_starts > 0, combo_ids[run_starts - 1], start_token)
            next_combo = np.where(run_ends < len(combo_ids) - 1, combo_ids[run_ends + 1], end_token)

            class_prev_counts[class_idx] += np.bincount(prev_combo, minlength=n_context_combo)
            class_next_counts[class_idx] += np.bincount(next_combo, minlength=n_context_combo)

    return StepAnalysisResult(
        step=step,
        combo_counts=combo_counts,
        combo_time_counts=combo_time_counts,
        class_prev_counts=class_prev_counts,
        class_next_counts=class_next_counts,
        class_run_count=class_run_count,
        class_run_duration_sum=class_run_duration_sum,
        source_zero_counts_by_bin=source_zero_counts_by_bin,
        source_total_counts_by_bin=source_total_counts_by_bin,
        source_total_windows=source_total_windows,
        source_zero_windows=source_zero_windows,
    )


def export_step_outputs(
    out_dir: Path,
    result: StepAnalysisResult,
    raw_meta: pd.DataFrame,
    label_cols: List[str],
    n_time_bins: int,
) -> Dict[str, object]:
    n_labels = len(label_cols)
    n_combo = 1 << n_labels
    start_token = n_combo
    end_token = n_combo + 1

    total_windows = int(np.sum(result.combo_counts))
    zero_windows = int(result.combo_counts[0]) if len(result.combo_counts) > 0 else 0
    zero_ratio = (zero_windows / total_windows) if total_windows > 0 else 0.0

    records = []
    nonzero_combo_ids = np.flatnonzero(result.combo_counts > 0)
    for combo_id in nonzero_combo_ids:
        counts_by_bin = result.combo_time_counts[combo_id]
        c = int(result.combo_counts[combo_id])
        first10 = int(np.sum(counts_by_bin[: max(1, n_time_bins // 10)], dtype=np.int64))
        last10 = int(np.sum(counts_by_bin[-max(1, n_time_bins // 10):], dtype=np.int64))
        records.append(
            {
                "combo_id": int(combo_id),
                "combo_name": combo_id_to_name(int(combo_id), label_cols),
                "count": c,
                "ratio": c / total_windows if total_windows > 0 else 0.0,
                "first_10pct_count": first10,
                "first_10pct_ratio_within_combo": first10 / c if c > 0 else 0.0,
                "last_10pct_count": last10,
                "last_10pct_ratio_within_combo": last10 / c if c > 0 else 0.0,
            }
        )
    combo_df = pd.DataFrame(records).sort_values("count", ascending=False)
    combo_csv_path = out_dir / f"step_{result.step}_combo_distribution.csv"
    combo_df.to_csv(combo_csv_path, index=False, encoding="utf-8-sig")

    zero_bin_rows = []
    for source_idx in sorted(result.source_total_windows.keys()):
        row_meta = raw_meta.iloc[int(source_idx)]
        total_by_bin = result.source_total_counts_by_bin[source_idx]
        zero_by_bin = result.source_zero_counts_by_bin[source_idx]
        for b in range(n_time_bins):
            total_b = int(total_by_bin[b])
            zero_b = int(zero_by_bin[b])
            zero_bin_rows.append(
                {
                    "source_index": source_idx,
                    "scenario": int(row_meta["scenario"]),
                    "experiment": int(row_meta["experiment"]),
                    "bin_index": b,
                    "bin_start_pct": b * 100.0 / n_time_bins,
                    "bin_end_pct": (b + 1) * 100.0 / n_time_bins,
                    "total_windows": total_b,
                    "zero_windows": zero_b,
                    "zero_ratio": zero_b / total_b if total_b > 0 else 0.0,
                }
            )
    zero_bin_df = pd.DataFrame(zero_bin_rows)
    zero_bin_csv_path = out_dir / f"step_{result.step}_zero_distribution_by_source_bin.csv"
    zero_bin_df.to_csv(zero_bin_csv_path, index=False, encoding="utf-8-sig")

    context_rows_prev = []
    context_rows_next = []
    context_vocab = list(range(n_combo)) + [start_token, end_token]
    context_name_map = {cid: combo_id_to_name(cid, label_cols) for cid in range(n_combo)}
    context_name_map[start_token] = "__START__"
    context_name_map[end_token] = "__END__"

    for class_idx, class_name in enumerate(label_cols):
        prev_counts = result.class_prev_counts[class_idx]
        next_counts = result.class_next_counts[class_idx]
        prev_total = int(np.sum(prev_counts))
        next_total = int(np.sum(next_counts))
        for cid in context_vocab:
            pc = int(prev_counts[cid])
            nc = int(next_counts[cid])
            if pc > 0:
                context_rows_prev.append(
                    {
                        "class_name": class_name,
                        "context_combo_id": cid,
                        "context_combo_name": context_name_map[cid],
                        "count": pc,
                        "ratio_within_class": pc / prev_total if prev_total > 0 else 0.0,
                    }
                )
            if nc > 0:
                context_rows_next.append(
                    {
                        "class_name": class_name,
                        "context_combo_id": cid,
                        "context_combo_name": context_name_map[cid],
                        "count": nc,
                        "ratio_within_class": nc / next_total if next_total > 0 else 0.0,
                    }
                )

    prev_df = pd.DataFrame(context_rows_prev).sort_values(["class_name", "count"], ascending=[True, False])
    next_df = pd.DataFrame(context_rows_next).sort_values(["class_name", "count"], ascending=[True, False])
    prev_csv_path = out_dir / f"step_{result.step}_class_context_prev.csv"
    next_csv_path = out_dir / f"step_{result.step}_class_context_next.csv"
    prev_df.to_csv(prev_csv_path, index=False, encoding="utf-8-sig")
    next_df.to_csv(next_csv_path, index=False, encoding="utf-8-sig")

    run_rows = []
    for class_idx, class_name in enumerate(label_cols):
        run_count = int(result.class_run_count[class_idx])
        run_dur_sum = int(result.class_run_duration_sum[class_idx])
        avg_dur = run_dur_sum / run_count if run_count > 0 else 0.0
        run_rows.append(
            {
                "class_name": class_name,
                "run_count": run_count,
                "avg_run_length_in_windows": avg_dur,
                "total_run_windows": run_dur_sum,
            }
        )
    run_df = pd.DataFrame(run_rows).sort_values("run_count", ascending=False)
    run_csv_path = out_dir / f"step_{result.step}_class_run_stats.csv"
    run_df.to_csv(run_csv_path, index=False, encoding="utf-8-sig")

    # Plot 1: zero ratio by timeline position
    make_plot_zero_ratio_by_position(
        out_path=out_dir / f"step_{result.step}_zero_ratio_by_position.png",
        result=result,
        n_time_bins=n_time_bins,
    )

    # Plot 2: timeline heatmap for most frequent combos
    make_plot_combo_time_heatmap(
        out_path=out_dir / f"step_{result.step}_top_combo_time_heatmap.png",
        result=result,
        label_cols=label_cols,
        n_time_bins=n_time_bins,
        top_k=12,
    )

    # Plot 3/4: class context heatmaps (previous / next)
    prev_sum = result.class_prev_counts.sum(axis=0)
    next_sum = result.class_next_counts.sum(axis=0)
    top_prev_ids = np.argsort(prev_sum)[::-1][:10]
    top_next_ids = np.argsort(next_sum)[::-1][:10]
    prev_context_labels = []
    next_context_labels = []
    for cid in top_prev_ids:
        if cid == start_token:
            prev_context_labels.append("__START__")
        elif cid == end_token:
            prev_context_labels.append("__END__")
        else:
            prev_context_labels.append(combo_id_to_name(int(cid), label_cols))
    for cid in top_next_ids:
        if cid == start_token:
            next_context_labels.append("__START__")
        elif cid == end_token:
            next_context_labels.append("__END__")
        else:
            next_context_labels.append(combo_id_to_name(int(cid), label_cols))

    make_plot_class_context_heatmap(
        out_path=out_dir / f"step_{result.step}_class_prev_context_heatmap.png",
        counts_2d=result.class_prev_counts[:, top_prev_ids],
        label_cols=label_cols,
        context_labels=prev_context_labels,
        step=result.step,
        title_prefix="Previous",
    )
    make_plot_class_context_heatmap(
        out_path=out_dir / f"step_{result.step}_class_next_context_heatmap.png",
        counts_2d=result.class_next_counts[:, top_next_ids],
        label_cols=label_cols,
        context_labels=next_context_labels,
        step=result.step,
        title_prefix="Next",
    )

    top_zero_context_prev = (
        prev_df[prev_df["context_combo_id"] == 0]
        .sort_values("count", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    top_zero_context_next = (
        next_df[next_df["context_combo_id"] == 0]
        .sort_values("count", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    return {
        "step": result.step,
        "total_windows": total_windows,
        "zero_windows": zero_windows,
        "zero_ratio": zero_ratio,
        "combo_distribution_csv": str(combo_csv_path),
        "zero_bin_csv": str(zero_bin_csv_path),
        "class_context_prev_csv": str(prev_csv_path),
        "class_context_next_csv": str(next_csv_path),
        "class_run_stats_csv": str(run_csv_path),
        "top_zero_context_prev": top_zero_context_prev,
        "top_zero_context_next": top_zero_context_next,
    }


def main() -> None:
    cfg = Config()
    label_cols = list(cfg.data.label_cols)
    raw_path = Path("data") / cfg.data.raw_dataset_file

    output_root = Path("output") / "raw_label_distribution_analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    with raw_path.open("rb") as f:
        raw_meta = pickle.load(f)
    if not isinstance(raw_meta, pd.DataFrame):
        raise TypeError("Expected raw dataset payload to be a pandas DataFrame.")

    window_size = int(2 * cfg.prep.original_freq)  # align with existing 2s@2000Hz windowing
    steps = [1, 250]
    n_time_bins = 20

    run_meta = {
        "raw_path": str(raw_path),
        "window_size": window_size,
        "steps": steps,
        "n_time_bins": n_time_bins,
        "label_cols": label_cols,
        "sources": [],
    }
    for source_idx, row in raw_meta.iterrows():
        run_meta["sources"].append(
            {
                "source_index": int(source_idx),
                "scenario": int(row["scenario"]),
                "experiment": int(row["experiment"]),
                "raw_length": int(len(row["data"])),
            }
        )

    all_summaries = []
    for step in steps:
        print(f"[INFO] analyzing step={step} ...")
        result = analyze_step(
            raw_meta=raw_meta,
            label_cols=label_cols,
            window_size=window_size,
            step=step,
            n_time_bins=n_time_bins,
        )
        summary = export_step_outputs(
            out_dir=output_root,
            result=result,
            raw_meta=raw_meta,
            label_cols=label_cols,
            n_time_bins=n_time_bins,
        )
        all_summaries.append(summary)
        print(
            f"[INFO] step={step}: total_windows={summary['total_windows']} "
            f"zero_windows={summary['zero_windows']} zero_ratio={summary['zero_ratio']:.4f}"
        )

    final_summary = {
        "meta": run_meta,
        "steps": all_summaries,
    }
    summary_json_path = output_root / "summary.json"
    summary_json_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] summary written to: {summary_json_path}")


if __name__ == "__main__":
    main()
