import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from utils.config import Config


def combo_id_from_labels(labels_2d: np.ndarray) -> np.ndarray:
    n_labels = labels_2d.shape[1]
    weights = (1 << np.arange(n_labels, dtype=np.int64))
    return labels_2d.astype(np.int64, copy=False) @ weights


def combo_id_to_name(combo_id: int, label_cols: List[str]) -> str:
    if combo_id == 0:
        return "0-label"
    out: List[str] = []
    for i, label in enumerate(label_cols):
        if (combo_id >> i) & 1:
            out.append(label)
    return " + ".join(out)


def find_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    x = mask.astype(np.int8, copy=False)
    diff = np.diff(x, prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def find_value_runs(values: np.ndarray) -> List[Tuple[int, int, int]]:
    if values.size == 0:
        return []
    diff_idx = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], diff_idx))
    ends = np.concatenate((diff_idx - 1, [len(values) - 1]))
    out: List[Tuple[int, int, int]] = []
    for s, e in zip(starts, ends):
        out.append((int(s), int(e), int(values[s])))
    return out


def build_window_endpoints(
    labels: np.ndarray,
    window_size: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(labels)
    if n < window_size:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0, labels.shape[1]), dtype=np.int8),
        )
    starts = np.arange(0, n - window_size + 1, step, dtype=np.int64)
    endpoint_idx = starts + (window_size - 1)
    endpoint_labels = labels[endpoint_idx]
    return starts, endpoint_idx, endpoint_labels


def analyze_none_from_endpoint(
    endpoint_labels: np.ndarray,
    starts: np.ndarray,
    endpoint_idx: np.ndarray,
) -> Dict[str, object]:
    combo_ids = combo_id_from_labels(endpoint_labels)
    none_mask = combo_ids == 0
    runs = find_true_runs(none_mask)

    total = int(len(combo_ids))
    none_total = int(np.sum(none_mask))
    none_ratio = (none_total / total) if total > 0 else 0.0

    head_run = runs[0] if runs and runs[0][0] == 0 else None
    tail_run = runs[-1] if runs and runs[-1][1] == total - 1 else None

    head_count = (head_run[1] - head_run[0] + 1) if head_run is not None else 0
    tail_count = (tail_run[1] - tail_run[0] + 1) if tail_run is not None else 0

    middle_runs: List[Tuple[int, int]] = []
    for i, (s, e) in enumerate(runs):
        if head_run is not None and i == 0:
            continue
        if tail_run is not None and i == len(runs) - 1:
            continue
        middle_runs.append((s, e))
    middle_count = int(sum(e - s + 1 for s, e in middle_runs))

    keep_start = head_count
    keep_end = total - tail_count - 1
    if keep_start > keep_end:
        keep_start = None
        keep_end = None

    run_records = []
    for s, e in runs:
        run_records.append(
            {
                "run_start_win_idx": int(s),
                "run_end_win_idx": int(e),
                "run_len": int(e - s + 1),
                "run_start_window_start_raw_idx": int(starts[s]),
                "run_end_window_start_raw_idx": int(starts[e]),
                "run_start_window_end_raw_idx": int(endpoint_idx[s]),
                "run_end_window_end_raw_idx": int(endpoint_idx[e]),
            }
        )

    return {
        "total_windows": total,
        "none_total": none_total,
        "none_ratio": none_ratio,
        "head_none_count": int(head_count),
        "tail_none_count": int(tail_count),
        "middle_none_count": int(middle_count),
        "head_start_win_idx": int(head_run[0]) if head_run is not None else None,
        "head_end_win_idx": int(head_run[1]) if head_run is not None else None,
        "tail_start_win_idx": int(tail_run[0]) if tail_run is not None else None,
        "tail_end_win_idx": int(tail_run[1]) if tail_run is not None else None,
        "keep_start_win_idx": int(keep_start) if keep_start is not None else None,
        "keep_end_win_idx": int(keep_end) if keep_end is not None else None,
        "runs": run_records,
        "middle_runs": middle_runs,
    }


def ordered_unique(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def build_superclass_order(label_cols: List[str], superclass_mapping: Dict[str, str]) -> List[str]:
    return ordered_unique([superclass_mapping.get(label, label) for label in label_cols])


def categorize_windows(
    endpoint_labels: np.ndarray,
    label_cols: List[str],
    superclass_mapping: Dict[str, str],
    single_superclass_order: List[str],
) -> Tuple[np.ndarray, List[str]]:
    base_names = ["0-label"] + single_superclass_order
    if len(endpoint_labels) == 0:
        return np.empty((0,), dtype=np.int32), base_names

    name_to_base_id = {name: i for i, name in enumerate(base_names)}
    label_to_base_id = np.array(
        [name_to_base_id[superclass_mapping.get(label, label)] for label in label_cols],
        dtype=np.int32,
    )

    n = len(endpoint_labels)
    cat_ids = np.empty((n,), dtype=np.int32)
    sums = endpoint_labels.sum(axis=1)
    zero_mask = sums == 0
    single_mask = sums == 1
    multi_mask = sums > 1

    cat_ids[zero_mask] = 0
    if np.any(single_mask):
        single_label_idx = np.argmax(endpoint_labels[single_mask], axis=1)
        cat_ids[single_mask] = label_to_base_id[single_label_idx]

    category_names = list(base_names)
    if np.any(multi_mask):
        multi_combo_ids = combo_id_from_labels(endpoint_labels[multi_mask])
        unique_combo_ids = np.unique(multi_combo_ids)
        multi_names = [f"MULTI: {combo_id_to_name(int(cid), label_cols)}" for cid in unique_combo_ids]
        category_names.extend(multi_names)

        base_len = len(base_names)
        pos = np.searchsorted(unique_combo_ids, multi_combo_ids)
        cat_ids[multi_mask] = base_len + pos.astype(np.int32)

    return cat_ids, category_names


def _format_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def plot_none_counts(out_path: Path, summary_df: pd.DataFrame, step: int) -> None:
    sub = summary_df[summary_df["step"] == step].sort_values("experiment")
    exp_labels = [f"exp{int(v)}" for v in sub["experiment"]]
    x = np.arange(len(sub))
    head = sub["head_none_count"].to_numpy(dtype=np.int64)
    middle = sub["middle_none_count"].to_numpy(dtype=np.int64)
    tail = sub["tail_none_count"].to_numpy(dtype=np.int64)

    plt.figure(figsize=(10, 5))
    plt.bar(x, head, label="Head None", color="#4C78A8")
    plt.bar(x, middle, bottom=head, label="Middle None", color="#F58518")
    plt.bar(x, tail, bottom=head + middle, label="Tail None", color="#E45756")
    plt.xticks(x, exp_labels)
    plt.ylabel("Window count")
    plt.title(f"None-window counts by exp (step={step})")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_none_ranges_normalized(out_path: Path, summary_df: pd.DataFrame, middle_df: pd.DataFrame, step: int) -> None:
    sub = summary_df[summary_df["step"] == step].sort_values("experiment").reset_index(drop=True)
    x_max = int(sub["total_windows"].max()) if len(sub) > 0 else 1
    x_max = max(1, x_max)
    plt.figure(figsize=(12, 4.5))

    for i, row in sub.iterrows():
        total = int(row["total_windows"])
        if total <= 0:
            continue
        y = i
        plt.hlines(y, 0, total - 1, color="#CCCCCC", linewidth=6, alpha=0.5)

        head = int(row["head_none_count"])
        tail = int(row["tail_none_count"])
        if head > 0:
            plt.hlines(y, 0, head - 1, color="#4C78A8", linewidth=8, label="Head None" if i == 0 else None)
        if tail > 0:
            plt.hlines(
                y,
                total - tail,
                total - 1,
                color="#E45756",
                linewidth=8,
                label="Tail None" if i == 0 else None,
            )

        mids = middle_df[(middle_df["step"] == step) & (middle_df["experiment"] == row["experiment"])]
        for j, m in mids.iterrows():
            s = int(m["run_start_win_idx"])
            e = int(m["run_end_win_idx"])
            plt.hlines(y, s, e, color="#F58518", linewidth=6, label="Middle None" if (i == 0 and j == mids.index[0]) else None)

    plt.yticks(np.arange(len(sub)), [f"exp{int(v)}" for v in sub["experiment"]])
    plt.xlim(0, x_max - 1)
    plt.xlabel("Window index")
    plt.ylabel("Experiment")
    plt.title(f"None-run ranges by exp (step={step}, index-axis)")
    plt.grid(axis="x", alpha=0.25)
    plt.legend(loc="upper center", ncol=3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def sorted_category_names_for_step(
    step_df: pd.DataFrame,
    single_superclass_order: List[str],
) -> List[str]:
    names = sorted(step_df["category_name"].dropna().unique().tolist())
    multi = sorted([n for n in names if n.startswith("MULTI: ")])
    ordered: List[str] = []
    if "NONE" in names:
        ordered.append("NONE")
    for s in single_superclass_order:
        if s in names:
            ordered.append(s)
    ordered.extend(multi)
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered


def build_category_color_map(category_names: List[str], single_superclass_order: List[str]) -> Dict[str, str]:
    color_map: Dict[str, str] = {"0-label": "#111111"}
    single_palette = [
        "#59A14F",
        "#E15759",
        "#F28E2B",
        "#76B7B2",
        "#EDC948",
        "#B07AA1",
    ]
    for i, s in enumerate(single_superclass_order):
        color_map[s] = single_palette[i % len(single_palette)]

    # Multi-label combinations use a dedicated high-contrast palette
    multi_palette = [
        "#D62728", "#1F77B4", "#2CA02C", "#9467BD", "#FF7F0E",
        "#17BECF", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22",
    ]
    multi_idx = 0
    for name in category_names:
        if name in color_map:
            continue
        color_map[name] = multi_palette[multi_idx % len(multi_palette)]
        multi_idx += 1
    return color_map


def plot_category_ranges_normalized(
    out_path: Path,
    summary_df: pd.DataFrame,
    category_runs_df: pd.DataFrame,
    step: int,
    single_superclass_order: List[str],
) -> None:
    sub = summary_df[summary_df["step"] == step].sort_values("experiment").reset_index(drop=True)
    step_runs = category_runs_df[category_runs_df["step"] == step]
    category_names = sorted_category_names_for_step(step_runs, single_superclass_order)
    color_map = build_category_color_map(category_names, single_superclass_order)
    x_max = int(sub["total_windows"].max()) if len(sub) > 0 else 1
    x_max = max(1, x_max)

    plt.figure(figsize=(16, 5))
    for i, row in sub.iterrows():
        total = int(row["total_windows"])
        exp = int(row["experiment"])
        y = i
        if total <= 0:
            continue
        plt.hlines(y, 0, total - 1, color="#DDDDDD", linewidth=6, alpha=0.55)
        rs = step_runs[step_runs["experiment"] == exp].sort_values("run_start_win_idx")
        for _, rr in rs.iterrows():
            c_name = rr["category_name"]
            s = int(rr["run_start_win_idx"])
            e = int(rr["run_end_win_idx"])
            plt.hlines(y, s, e, color=color_map[c_name], linewidth=8, alpha=0.96)

    handles = [Line2D([0], [0], color=color_map[n], lw=6, label=n) for n in category_names]
    plt.yticks(np.arange(len(sub)), [f"exp{int(v)}" for v in sub["experiment"]])
    plt.xlim(0, x_max - 1)
    plt.xlabel("Window index")
    plt.ylabel("Experiment")
    plt.title(f"Window-category ranges by exp (step={step}, 6 superclasses + exact multi combos)")
    plt.grid(axis="x", alpha=0.25)
    ncol = 4 if len(handles) <= 12 else 3
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=ncol, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def _compute_zero_stage_ratios(zero_bin_csv: Path) -> Tuple[float, float, float]:
    z = pd.read_csv(zero_bin_csv)
    g = z.groupby("bin_index", as_index=False)[["total_windows", "zero_windows"]].sum()
    if len(g) == 0:
        return 0.0, 0.0, 0.0
    n = len(g)
    edge = max(1, n // 10)
    first = g.head(edge)
    middle = g.iloc[edge: n - edge] if n - edge > edge else g.iloc[0:0]
    last = g.tail(edge)
    f = float(first["zero_windows"].sum() / first["total_windows"].sum()) if first["total_windows"].sum() > 0 else 0.0
    m = float(middle["zero_windows"].sum() / middle["total_windows"].sum()) if len(middle) > 0 and middle["total_windows"].sum() > 0 else 0.0
    l = float(last["zero_windows"].sum() / last["total_windows"].sum()) if last["total_windows"].sum() > 0 else 0.0
    return f, m, l


def build_visual_markdown_report(
    out_md_path: Path,
    category_counts_df: pd.DataFrame,
    single_superclass_order: List[str],
) -> None:
    raw_dir = Path("raw_label_distribution_analysis")
    none_dir = Path("none_by_exp_analysis")

    combo1 = pd.read_csv(Path("output") / raw_dir / "step_1_combo_distribution.csv")
    combo250 = pd.read_csv(Path("output") / raw_dir / "step_250_combo_distribution.csv")
    none_summary = pd.read_csv(Path("output") / none_dir / "none_summary_by_exp.csv")

    zf1, zm1, zl1 = _compute_zero_stage_ratios(Path("output") / raw_dir / "step_1_zero_distribution_by_source_bin.csv")
    zf250, zm250, zl250 = _compute_zero_stage_ratios(Path("output") / raw_dir / "step_250_zero_distribution_by_source_bin.csv")

    top3_1 = combo1.head(3)[["combo_name", "ratio"]].values.tolist()
    top3_250 = combo250.head(3)[["combo_name", "ratio"]].values.tolist()

    def exp_none_text(step: int) -> str:
        sub = none_summary[none_summary["step"] == step].sort_values("experiment")
        return "; ".join(
            [
                f"exp{int(r['experiment'])}: head={int(r['head_none_count'])}, middle={int(r['middle_none_count'])}, tail={int(r['tail_none_count'])}"
                for _, r in sub.iterrows()
            ]
        )

    def cat_text(step: int) -> str:
        sub = category_counts_df[category_counts_df["step"] == step]
        if len(sub) == 0:
            return "No category data."
        agg = sub.groupby("category_name", as_index=False)["count"].sum().sort_values("count", ascending=False)
        total = int(agg["count"].sum())
        agg["ratio"] = agg["count"] / total if total > 0 else 0.0
        top = agg.head(6)
        items = [f"{r['category_name']}({_format_pct(float(r['ratio']))})" for _, r in top.iterrows()]
        multi_total = agg[agg["category_name"].str.startswith("MULTI: ")]["ratio"].sum()
        return f"Top categories: {', '.join(items)}; all multi-label total: {_format_pct(float(multi_total))}."

    lines: List[str] = []
    lines.append("# CPSHAR Visual Analysis Summary")
    lines.append("")
    lines.append("This report merges all previous and newly generated figures.")
    lines.append("")
    lines.append("## X-axis Meaning")
    lines.append("")
    lines.append("- `Window index` means direct index of the generated windows in each exp timeline.")
    lines.append("- `0` is the first window; larger values are later windows.")
    lines.append("- Because exp lengths differ, each row naturally ends at a different max index.")
    lines.append("")
    lines.append("## Superclass Rule For New Category Plots")
    lines.append("")
    lines.append(f"- Single-label windows are merged into 6 superclasses: {', '.join(single_superclass_order)}.")
    lines.append("- Multi-label windows are kept as separate categories by exact combination, e.g. `MULTI: Driving(curve) + Lifting(lowering)`.")
    lines.append("")

    entries = [
        ("Figure 1", "step=1 zero-label position ratio", raw_dir / "step_1_zero_ratio_by_position.png",
         f"Zero-label is concentrated near both ends: first 10%={_format_pct(zf1)}, middle 80%={_format_pct(zm1)}, last 10%={_format_pct(zl1)}."),
        ("Figure 2", "step=250 zero-label position ratio", raw_dir / "step_250_zero_ratio_by_position.png",
         f"Same pattern as step=1: first 10%={_format_pct(zf250)}, middle 80%={_format_pct(zm250)}, last 10%={_format_pct(zl250)}."),
        ("Figure 3", "step=1 top combo timeline heatmap", raw_dir / "step_1_top_combo_time_heatmap.png",
         f"Top-3 combos: {top3_1[0][0]}({_format_pct(float(top3_1[0][1]))}), {top3_1[1][0]}({_format_pct(float(top3_1[1][1]))}), {top3_1[2][0]}({_format_pct(float(top3_1[2][1]))})."),
        ("Figure 4", "step=250 top combo timeline heatmap", raw_dir / "step_250_top_combo_time_heatmap.png",
         f"Top-3 combos consistent with step=1: {top3_250[0][0]}, {top3_250[1][0]}, {top3_250[2][0]}."),
        ("Figure 5", "step=1 none count split by exp", none_dir / "step_1_none_counts_stacked.png",
         exp_none_text(1) + "."),
        ("Figure 6", "step=250 none count split by exp", none_dir / "step_250_none_counts_stacked.png",
         exp_none_text(250) + "."),
        ("Figure 7", "step=1 none run ranges (index-axis)", none_dir / "step_1_none_ranges_by_index.png",
         "Head and tail none-runs dominate; exp4 still has a small middle none-run."),
        ("Figure 8", "step=250 none run ranges (index-axis)", none_dir / "step_250_none_ranges_by_index.png",
         "Same trend as step=1, with a very short middle none-run in exp4."),
        ("Figure 9 (NEW)", "step=1 category ranges (6 superclasses + exact multi-label combos)",
         none_dir / "step_1_category_ranges_by_index.png", cat_text(1)),
        ("Figure 10 (NEW)", "step=250 category ranges (6 superclasses + exact multi-label combos)",
         none_dir / "step_250_category_ranges_by_index.png", cat_text(250)),
    ]

    for fig_no, title, rel_path, summary in entries:
        lines.append(f"## {fig_no}: {title}")
        lines.append("")
        lines.append(f"![{fig_no}]({rel_path.as_posix()})")
        lines.append("")
        lines.append(f"- Summary: {summary}")
        lines.append("")

    out_md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = Config()
    label_cols = list(cfg.data.label_cols)
    superclass_mapping = dict(cfg.data.superclass_mapping)
    single_superclass_order = build_superclass_order(label_cols, superclass_mapping)

    raw_path = Path("data") / cfg.data.raw_dataset_file
    out_dir = Path("output") / "none_by_exp_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    with raw_path.open("rb") as f:
        raw_meta = pickle.load(f)
    if not isinstance(raw_meta, pd.DataFrame):
        raise TypeError("Expected raw dataset payload to be pandas DataFrame.")

    window_size = int(2 * cfg.prep.original_freq)
    steps = [1, 250]

    summary_rows: List[Dict[str, object]] = []
    middle_rows: List[Dict[str, object]] = []
    run_rows: List[Dict[str, object]] = []
    category_run_rows: List[Dict[str, object]] = []
    category_count_rows: List[Dict[str, object]] = []

    for step in steps:
        for source_idx, row in raw_meta.iterrows():
            exp = int(row["experiment"])
            scenario = int(row["scenario"])
            labels = row["data"][label_cols].to_numpy(dtype=np.int8, copy=False)
            starts, endpoint_idx, endpoint_labels = build_window_endpoints(labels, window_size, step)

            none_result = analyze_none_from_endpoint(endpoint_labels, starts, endpoint_idx)
            summary_rows.append(
                {
                    "step": step,
                    "source_index": int(source_idx),
                    "scenario": scenario,
                    "experiment": exp,
                    "total_windows": none_result["total_windows"],
                    "none_total": none_result["none_total"],
                    "none_ratio": none_result["none_ratio"],
                    "head_none_count": none_result["head_none_count"],
                    "tail_none_count": none_result["tail_none_count"],
                    "middle_none_count": none_result["middle_none_count"],
                    "head_start_win_idx": none_result["head_start_win_idx"],
                    "head_end_win_idx": none_result["head_end_win_idx"],
                    "tail_start_win_idx": none_result["tail_start_win_idx"],
                    "tail_end_win_idx": none_result["tail_end_win_idx"],
                    "keep_start_win_idx": none_result["keep_start_win_idx"],
                    "keep_end_win_idx": none_result["keep_end_win_idx"],
                }
            )

            for s, e in none_result["middle_runs"]:
                middle_rows.append(
                    {
                        "step": step,
                        "source_index": int(source_idx),
                        "scenario": scenario,
                        "experiment": exp,
                        "run_start_win_idx": int(s),
                        "run_end_win_idx": int(e),
                        "run_len": int(e - s + 1),
                    }
                )

            for rr in none_result["runs"]:
                run_rows.append(
                    {
                        "step": step,
                        "source_index": int(source_idx),
                        "scenario": scenario,
                        "experiment": exp,
                        **rr,
                    }
                )

            cat_ids, cat_names = categorize_windows(
                endpoint_labels=endpoint_labels,
                label_cols=label_cols,
                superclass_mapping=superclass_mapping,
                single_superclass_order=single_superclass_order,
            )
            total_windows = len(cat_ids)
            if total_windows == 0:
                continue

            cat_counts = np.bincount(cat_ids, minlength=len(cat_names))
            for cid, cnt in enumerate(cat_counts):
                category_count_rows.append(
                    {
                        "step": step,
                        "source_index": int(source_idx),
                        "scenario": scenario,
                        "experiment": exp,
                        "category_id_local": int(cid),
                        "category_name": cat_names[cid],
                        "count": int(cnt),
                        "ratio": float(cnt / total_windows),
                    }
                )

            cat_runs = find_value_runs(cat_ids)
            for s, e, cid in cat_runs:
                category_run_rows.append(
                    {
                        "step": step,
                        "source_index": int(source_idx),
                        "scenario": scenario,
                        "experiment": exp,
                        "category_name": cat_names[cid],
                        "run_start_win_idx": int(s),
                        "run_end_win_idx": int(e),
                        "run_len": int(e - s + 1),
                        "run_start_window_start_raw_idx": int(starts[s]),
                        "run_end_window_start_raw_idx": int(starts[e]),
                        "run_start_window_end_raw_idx": int(endpoint_idx[s]),
                        "run_end_window_end_raw_idx": int(endpoint_idx[e]),
                    }
                )

    summary_df = pd.DataFrame(summary_rows).sort_values(["step", "experiment"])
    middle_df = pd.DataFrame(middle_rows).sort_values(["step", "experiment", "run_start_win_idx"])
    run_df = pd.DataFrame(run_rows).sort_values(["step", "experiment", "run_start_win_idx"])
    category_runs_df = pd.DataFrame(category_run_rows).sort_values(["step", "experiment", "run_start_win_idx"])
    category_counts_df = pd.DataFrame(category_count_rows).sort_values(["step", "experiment", "category_name"])

    summary_path = out_dir / "none_summary_by_exp.csv"
    middle_path = out_dir / "none_middle_runs_by_exp.csv"
    runs_path = out_dir / "none_all_runs_by_exp.csv"
    category_runs_path = out_dir / "category_runs_by_exp.csv"
    category_counts_path = out_dir / "category_counts_by_exp.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    middle_df.to_csv(middle_path, index=False, encoding="utf-8-sig")
    run_df.to_csv(runs_path, index=False, encoding="utf-8-sig")
    category_runs_df.to_csv(category_runs_path, index=False, encoding="utf-8-sig")
    category_counts_df.to_csv(category_counts_path, index=False, encoding="utf-8-sig")

    for step in steps:
        plot_none_counts(out_dir / f"step_{step}_none_counts_stacked.png", summary_df, step)
        plot_none_ranges_normalized(out_dir / f"step_{step}_none_ranges_by_index.png", summary_df, middle_df, step)
        plot_category_ranges_normalized(
            out_dir / f"step_{step}_category_ranges_by_index.png",
            summary_df,
            category_runs_df,
            step,
            single_superclass_order,
        )

    consistency_rows = []
    for step in steps:
        sub = summary_df[summary_df["step"] == step]
        consistency_rows.append(
            {
                "step": step,
                "head_none_count_all_equal": bool(sub["head_none_count"].nunique() == 1),
                "tail_none_count_all_equal": bool(sub["tail_none_count"].nunique() == 1),
                "head_values": sub["head_none_count"].tolist(),
                "tail_values": sub["tail_none_count"].tolist(),
            }
        )

    report = {
        "raw_path": str(raw_path),
        "window_size": window_size,
        "steps": steps,
        "single_superclass_order": single_superclass_order,
        "summary_csv": str(summary_path),
        "middle_runs_csv": str(middle_path),
        "all_runs_csv": str(runs_path),
        "category_runs_csv": str(category_runs_path),
        "category_counts_csv": str(category_counts_path),
        "consistency": consistency_rows,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    visual_md_path = Path("output") / "visual_analysis_summary.md"
    build_visual_markdown_report(
        out_md_path=visual_md_path,
        category_counts_df=category_counts_df,
        single_superclass_order=single_superclass_order,
    )

    print(f"[INFO] summary: {summary_path}")
    print(f"[INFO] middle runs: {middle_path}")
    print(f"[INFO] all runs: {runs_path}")
    print(f"[INFO] category runs: {category_runs_path}")
    print(f"[INFO] category counts: {category_counts_path}")
    print(f"[INFO] report: {report_path}")
    print(f"[INFO] visual markdown: {visual_md_path}")


if __name__ == "__main__":
    main()
