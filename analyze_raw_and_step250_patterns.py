import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.config import Config


RAW_PATH = Path("data") / "cps_data_multi_label.pkl"
STEP250_PATH = Path("data") / "cps_windows_2s_2000hz_step_250.pkl"
OUT_DIR = Path("output") / "raw_step250_pattern_analysis"


def apply_superclass_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    target_superclasses = sorted(set(mapping.values()))
    for super_name in target_superclasses:
        children = [k for k, v in mapping.items() if v == super_name and k in out.columns]
        if children:
            out[super_name] = out[children].max(axis=1)
        else:
            out[super_name] = 0
    return out


def state_name_from_vector(vec: np.ndarray, state_cols: List[str]) -> str:
    active = [state_cols[i] for i, v in enumerate(vec) if int(v) == 1]
    return " + ".join(active) if active else "0-label"


def compress_state_sequence(states: List[str]) -> List[str]:
    if not states:
        return []
    out = [states[0]]
    for s in states[1:]:
        if s != out[-1]:
            out.append(s)
    return out


def trim_zero_edges(state_vec: np.ndarray) -> Tuple[int, int]:
    active = state_vec.sum(axis=1) > 0
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return -1, -1
    return int(idx[0]), int(idx[-1])


def build_source_cache(raw_meta: pd.DataFrame, state_cols: List[str], mapping: Dict[str, str]):
    cache = {}
    trim_rows = []
    for source_index, row in raw_meta.iterrows():
        df = apply_superclass_mapping(row["data"], mapping)
        vec = df[state_cols].to_numpy(dtype=np.int8, copy=False)
        states = [state_name_from_vector(v, state_cols) for v in vec]
        valid_start, valid_end = trim_zero_edges(vec)
        cache[int(source_index)] = {
            "scenario": int(row["scenario"]),
            "experiment": int(row["experiment"]),
            "df": df,
            "state_vec": vec,
            "states": states,
            "valid_start": valid_start,
            "valid_end": valid_end,
        }
        trim_rows.append(
            {
                "source_index": int(source_index),
                "scenario": int(row["scenario"]),
                "experiment": int(row["experiment"]),
                "raw_len": int(len(df)),
                "valid_start": valid_start,
                "valid_end": valid_end,
                "valid_len": int(valid_end - valid_start + 1) if valid_start >= 0 else 0,
            }
        )
    trim_df = pd.DataFrame(trim_rows)
    return cache, trim_df


def analyze_raw_transitions(cache: Dict[int, Dict], state_cols: List[str]):
    transition_counter = Counter()
    state_run_counter = Counter()
    run_rows = []
    all_states = set()
    for source_index, info in cache.items():
        vs, ve = info["valid_start"], info["valid_end"]
        if vs < 0:
            continue
        trimmed = info["states"][vs : ve + 1]
        compressed = compress_state_sequence(trimmed)
        all_states.update(compressed)

        starts = [0]
        for i in range(1, len(trimmed)):
            if trimmed[i] != trimmed[i - 1]:
                starts.append(i)
        starts.append(len(trimmed))
        for i in range(len(starts) - 1):
            s = starts[i]
            e = starts[i + 1] - 1
            state = trimmed[s]
            state_run_counter[state] += 1
            run_rows.append(
                {
                    "source_index": source_index,
                    "experiment": info["experiment"],
                    "state": state,
                    "run_start_trim_idx": s,
                    "run_end_trim_idx": e,
                    "run_len": e - s + 1,
                }
            )
        for a, b in zip(compressed[:-1], compressed[1:]):
            transition_counter[(a, b)] += 1

    ordered_states = sorted(all_states)
    mat = pd.DataFrame(0, index=ordered_states, columns=ordered_states, dtype=np.int64)
    for (a, b), c in transition_counter.items():
        mat.loc[a, b] = c
    runs_df = pd.DataFrame(run_rows)
    trans_df = pd.DataFrame(
        [{"from_state": a, "to_state": b, "count": c} for (a, b), c in transition_counter.items()]
    ).sort_values("count", ascending=False)
    return mat, trans_df, runs_df


def plot_transition_heatmap(mat: pd.DataFrame, out_path: Path) -> None:
    if mat.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 9))
    arr = mat.to_numpy(dtype=np.float64)
    vmax = max(1.0, float(arr.max()))
    im = ax.imshow(arr, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0:
                ax.text(j, i, f"{int(arr[i,j])}", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title("Raw transition counts after trimming head/tail 0-label segments")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def label_state_from_payload(y_row: np.ndarray, state_cols: List[str]) -> str:
    return state_name_from_vector(y_row.astype(np.int8), state_cols)


def analyze_step250_windows(payload: Dict, cache: Dict[int, Dict], state_cols: List[str]):
    X = payload["X"]
    y = payload["y"]
    source_index = payload["source_index"]
    start_idx = payload["start_idx"]
    window_size = int(payload["window_size"])

    rows = []
    ratio_rows = []
    valid_count = 0
    invalid_count = 0
    template_counter_by_target = defaultdict(Counter)

    all_state_names = set()
    for info in cache.values():
        all_state_names.update(set(info["states"]))
    for i in range(len(X)):
        src = int(source_index[i])
        s0 = int(start_idx[i])
        s1 = s0 + window_size - 1
        info = cache[src]
        vs, ve = info["valid_start"], info["valid_end"]
        if vs < 0 or s0 < vs or s1 > ve:
            invalid_count += 1
            continue
        valid_count += 1

        state_seq = info["states"][s0 : s1 + 1]
        compressed = compress_state_sequence(state_seq)
        unique_states = list(dict.fromkeys(state_seq))
        target_state = label_state_from_payload(y[i], state_cols)
        template = " -> ".join(compressed)
        template_counter_by_target[target_state][template] += 1
        all_state_names.update(unique_states)

        state_counts = Counter(state_seq)
        for st in all_state_names:
            ratio_rows.append(
                {
                    "sample_index": i,
                    "target_state": target_state,
                    "inside_state": st,
                    "ratio": state_counts.get(st, 0) / window_size,
                }
            )
        rows.append(
            {
                "sample_index": i,
                "experiment": int(payload["experiment"][i]),
                "source_index": src,
                "start_idx": s0,
                "target_state": target_state,
                "segment_count": len(compressed),
                "unique_state_count": len(unique_states),
                "template": template,
                "first_state": compressed[0] if compressed else "",
                "last_state": compressed[-1] if compressed else "",
            }
        )

    summary_df = pd.DataFrame(rows)
    ratios_df = pd.DataFrame(ratio_rows)

    template_rows = []
    for target, counter in template_counter_by_target.items():
        total = sum(counter.values())
        for template, c in counter.most_common():
            template_rows.append(
                {
                    "target_state": target,
                    "template": template,
                    "count": c,
                    "ratio_within_target": c / total if total > 0 else 0.0,
                    "segment_count": len(template.split(" -> ")) if template else 0,
                }
            )
    templates_df = pd.DataFrame(template_rows)

    ratio_summary_df = (
        ratios_df.groupby(["target_state", "inside_state"], as_index=False)
        .agg(
            mean_ratio=("ratio", "mean"),
            median_ratio=("ratio", "median"),
            max_ratio=("ratio", "max"),
        )
    )

    meta = {
        "total_step250_windows": int(len(X)),
        "valid_windows_after_trim_filter": int(valid_count),
        "invalid_windows_removed_by_zero_edge_filter": int(invalid_count),
    }
    return summary_df, templates_df, ratio_summary_df, meta


def plot_segment_count_distribution(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        return
    counts = summary_df["segment_count"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    for x, y in zip(counts.index.astype(str), counts.values):
        ax.text(x, y, str(int(y)), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Compressed segment count inside sample")
    ax.set_ylabel("Window count")
    ax.set_title("How many state-composition segments exist inside each valid step250 window")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_multi_template_topbars(templates_df: pd.DataFrame, out_path: Path) -> None:
    multis = sorted([x for x in templates_df["target_state"].dropna().unique().tolist() if " + " in x])
    if not multis:
        return
    fig, axes = plt.subplots(len(multis), 1, figsize=(12, 3.4 * len(multis)))
    if len(multis) == 1:
        axes = [axes]
    for ax, target in zip(axes, multis):
        ss = templates_df[templates_df["target_state"] == target].sort_values("count", ascending=False).head(6)
        ax.barh(np.arange(len(ss)), ss["ratio_within_target"], color="#f58518")
        ax.set_yticks(np.arange(len(ss)))
        ax.set_yticklabels(ss["template"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Ratio within target state")
        ax.set_title(target)
        for i, (_, r) in enumerate(ss.iterrows()):
            ax.text(float(r["ratio_within_target"]) + 0.01, i, f"{r['count']} ({r['ratio_within_target']*100:.1f}%)", va="center", fontsize=8)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Top internal state templates for multi-label target windows", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_multi_ratio_heatmap(ratio_summary_df: pd.DataFrame, out_path: Path) -> None:
    multis = sorted([x for x in ratio_summary_df["target_state"].dropna().unique().tolist() if " + " in x])
    states = sorted(ratio_summary_df["inside_state"].dropna().unique().tolist())
    if not multis or not states:
        return
    mat = np.zeros((len(multis), len(states)), dtype=np.float64)
    for i, t in enumerate(multis):
        for j, s in enumerate(states):
            r = ratio_summary_df[(ratio_summary_df["target_state"] == t) & (ratio_summary_df["inside_state"] == s)]
            if len(r) > 0:
                mat[i, j] = float(r["mean_ratio"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=max(0.4, float(mat.max()) if mat.size else 0.4))
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(np.arange(len(states)))
    ax.set_xticklabels(states, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(multis)))
    ax.set_yticklabels(multis, fontsize=8)
    ax.set_xlabel("Inside-window state composition")
    ax.set_ylabel("Target window state")
    ax.set_title("Mean inside-window composition ratio for multi-label target windows")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_markdown(
    trim_df: pd.DataFrame,
    trans_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    ratio_summary_df: pd.DataFrame,
    meta: Dict,
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Raw + Step250 Pattern Analysis")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Raw data is first mapped to 6 superclass labels.")
    lines.append("- Exact multi-label states are treated as a separate composition state.")
    lines.append("- Leading/trailing `0-label` segments in each source are removed before all downstream analysis.")
    lines.append("- Step250 sample analysis keeps only windows fully inside trimmed valid ranges.")
    lines.append("")

    lines.append("## Zero-edge Trimming")
    lines.append("")
    lines.append("| source_index | exp | raw_len | valid_start | valid_end | valid_len |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in trim_df.iterrows():
        lines.append(
            f"| {int(r['source_index'])} | {int(r['experiment'])} | {int(r['raw_len'])} | {int(r['valid_start'])} | {int(r['valid_end'])} | {int(r['valid_len'])} |"
        )
    lines.append("")

    lines.append("## Raw Transition Points")
    lines.append("")
    lines.append("Top transitions after trimming:")
    lines.append("")
    lines.append("| from_state | to_state | count |")
    lines.append("|---|---|---:|")
    for _, r in trans_df.head(15).iterrows():
        lines.append(f"| {r['from_state']} | {r['to_state']} | {int(r['count'])} |")
    lines.append("")
    lines.append("![raw transitions](raw_transition_heatmap.png)")
    lines.append("")

    max_seg = int(summary_df["segment_count"].max()) if not summary_df.empty else 0
    max_unique = int(summary_df["unique_state_count"].max()) if not summary_df.empty else 0
    seg_dist = summary_df["segment_count"].value_counts().sort_index()
    uniq_dist = summary_df["unique_state_count"].value_counts().sort_index()

    lines.append("## Step250 Sample Complexity")
    lines.append("")
    lines.append(f"- Total step250 windows: {meta['total_step250_windows']}")
    lines.append(f"- Valid windows kept after trimming filter: {meta['valid_windows_after_trim_filter']}")
    lines.append(f"- Invalid windows removed because they touched head/tail zero-label region: {meta['invalid_windows_removed_by_zero_edge_filter']}")
    lines.append(f"- Max compressed segment count inside one sample: {max_seg}")
    lines.append(f"- Max unique composition count inside one sample: {max_unique}")
    lines.append("")
    lines.append("Segment-count distribution:")
    lines.append("")
    lines.append("| segment_count | windows |")
    lines.append("|---:|---:|")
    for k, v in seg_dist.items():
        lines.append(f"| {int(k)} | {int(v)} |")
    lines.append("")
    lines.append("Unique-composition-count distribution:")
    lines.append("")
    lines.append("| unique_state_count | windows |")
    lines.append("|---:|---:|")
    for k, v in uniq_dist.items():
        lines.append(f"| {int(k)} | {int(v)} |")
    lines.append("")
    lines.append("![segment count](sample_segment_count_distribution.png)")
    lines.append("")

    multis = sorted([x for x in templates_df["target_state"].dropna().unique().tolist() if " + " in x])
    lines.append("## Multi-label Window Forms")
    lines.append("")
    for target in multis:
        ss = templates_df[templates_df["target_state"] == target].sort_values("count", ascending=False).head(8)
        lines.append(f"### {target}")
        lines.append("")
        lines.append("| template | count | ratio_within_target |")
        lines.append("|---|---:|---:|")
        for _, r in ss.iterrows():
            lines.append(f"| {r['template']} | {int(r['count'])} | {float(r['ratio_within_target']):.4f} |")
        rs = ratio_summary_df[ratio_summary_df["target_state"] == target].sort_values("mean_ratio", ascending=False)
        lines.append("")
        lines.append("Mean inside-window composition ratios:")
        lines.append("")
        lines.append("| inside_state | mean_ratio | median_ratio | max_ratio |")
        lines.append("|---|---:|---:|---:|")
        for _, r in rs.head(8).iterrows():
            lines.append(f"| {r['inside_state']} | {float(r['mean_ratio']):.4f} | {float(r['median_ratio']):.4f} | {float(r['max_ratio']):.4f} |")
        lines.append("")
    lines.append("![multi templates](multi_template_topbars.png)")
    lines.append("")
    lines.append("![multi mean ratios](multi_ratio_heatmap.png)")
    lines.append("")

    lines.append("## Direct Answers")
    lines.append("")
    lines.append("1. Multi-label windows are not uniformly mixed. They usually appear as ordered composition templates after compressing consecutive states.")
    lines.append("2. Exact forms differ by target multi-label. For example, `Driving(curve) + Lifting(lowering)` windows often look like `Driving(curve) -> Driving(curve) + Lifting(lowering)`, rather than random interleaving.")
    lines.append("3. Transition points can be counted cleanly on raw data after trimming invalid 0-label edges, and the dominant transitions are listed above.")
    lines.append("4. The assumption that a sample usually contains at most 3 composition segments should be checked against `max compressed segment count` above. This analysis gives the exact maximum instead of relying on intuition.")
    lines.append("5. Sample labels are last-point labels, so many windows include substantial proportion of previous-state data. That effect is visible in the mean inside-window composition ratios.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    mapping = dict(cfg.data.superclass_mapping)
    state_cols = sorted(set(mapping.values()))

    with RAW_PATH.open("rb") as f:
        raw_meta = pickle.load(f)
    with STEP250_PATH.open("rb") as f:
        step250 = pickle.load(f)

    cache, trim_df = build_source_cache(raw_meta, state_cols, mapping)
    trim_df.to_csv(OUT_DIR / "trim_ranges.csv", index=False, encoding="utf-8-sig")

    mat, trans_df, raw_runs_df = analyze_raw_transitions(cache, state_cols)
    trans_df.to_csv(OUT_DIR / "raw_transition_counts.csv", index=False, encoding="utf-8-sig")
    raw_runs_df.to_csv(OUT_DIR / "raw_runs.csv", index=False, encoding="utf-8-sig")
    plot_transition_heatmap(mat, OUT_DIR / "raw_transition_heatmap.png")

    summary_df, templates_df, ratio_summary_df, meta = analyze_step250_windows(step250, cache, state_cols)
    summary_df.to_csv(OUT_DIR / "step250_window_summary.csv", index=False, encoding="utf-8-sig")
    templates_df.to_csv(OUT_DIR / "step250_window_templates.csv", index=False, encoding="utf-8-sig")
    ratio_summary_df.to_csv(OUT_DIR / "step250_window_ratio_summary.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "analysis_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_segment_count_distribution(summary_df, OUT_DIR / "sample_segment_count_distribution.png")
    plot_multi_template_topbars(templates_df, OUT_DIR / "multi_template_topbars.png")
    plot_multi_ratio_heatmap(ratio_summary_df, OUT_DIR / "multi_ratio_heatmap.png")

    build_markdown(
        trim_df=trim_df,
        trans_df=trans_df,
        summary_df=summary_df,
        templates_df=templates_df,
        ratio_summary_df=ratio_summary_df,
        meta=meta,
        out_path=OUT_DIR / "REPORT.md",
    )
    print(f"[INFO] output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
