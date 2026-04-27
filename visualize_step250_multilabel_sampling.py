import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


RAW_FREQ = 2000.0
DATA_PATH = Path("data") / "cps_windows_2s_2000hz_step_250.pkl"
OUT_DIR = Path("output") / "step250_multilabel_sampling_viz"


def interval_downsample(x: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    y = x[::factor]
    idx = np.arange(0, len(x), factor, dtype=np.int64)
    return y, idx.astype(np.float64)


def sliding_window_downsample(x: np.ndarray, window: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n < window:
        return np.empty((0, x.shape[1]), dtype=np.float64), np.empty((0,), dtype=np.float64)
    starts = np.arange(0, n - window + 1, step, dtype=np.int64)
    out = np.empty((len(starts), x.shape[1]), dtype=np.float64)
    for i, s in enumerate(starts):
        out[i] = x[s : s + window].mean(axis=0, dtype=np.float64)
    center_idx = starts + (window - 1) / 2.0
    return out, center_idx.astype(np.float64)


def format_label_text(y_row: np.ndarray, label_cols: List[str]) -> str:
    active = [label_cols[i] for i, v in enumerate(y_row) if int(v) == 1]
    return " + ".join(active) if active else "0-label"


def plot_nine_axis(
    series: np.ndarray,
    sample_idx: np.ndarray,
    sensor_cols: List[str],
    out_path: Path,
    title: str,
    color: str = "#1f77b4",
) -> None:
    t = sample_idx / RAW_FREQ
    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    axes = axes.reshape(-1)
    for i, col in enumerate(sensor_cols):
        ax = axes[i]
        ax.plot(t, series[:, i], linewidth=1.0, color=color)
        ax.set_title(col, fontsize=10)
        ax.grid(alpha=0.25)
    for j in range(len(sensor_cols), 9):
        axes[j].set_axis_off()
    for ax in axes[-3:]:
        if ax.has_data():
            ax.set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overlay_comparison(
    cfg_order: List[str],
    cfg_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cfg_colors: Dict[str, str],
    baseline_key: str,
    sensor_cols: List[str],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    axes = axes.reshape(-1)

    for i, sensor in enumerate(sensor_cols):
        ax = axes[i]
        for key in cfg_order:
            series, idx = cfg_data[key]
            t = idx / RAW_FREQ
            is_baseline = key == baseline_key
            ax.plot(
                t,
                series[:, i],
                color=cfg_colors[key],
                linewidth=2.2 if is_baseline else 1.0,
                alpha=1.0 if is_baseline else 0.72,
                zorder=3 if is_baseline else 2,
            )
        ax.set_title(sensor, fontsize=10)
        ax.grid(alpha=0.25)

    for j in range(len(sensor_cols), 9):
        axes[j].set_axis_off()
    for ax in axes[-3:]:
        if ax.has_data():
            ax.set_xlabel("Time (s)")

    legend_handles = [
        Line2D([0], [0], color=cfg_colors[k], lw=2.4 if k == baseline_key else 1.6, label=k)
        for k in cfg_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(5, len(cfg_order)),
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(title, fontsize=13, y=0.965)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def pick_sample(payload: Dict) -> int:
    label_cols = payload["label_cols"]
    y = payload["y"]
    curve_idx = label_cols.index("Driving(curve)")
    lower_idx = label_cols.index("Lifting(lowering)")
    mask = (y[:, curve_idx] == 1) & (y[:, lower_idx] == 1) & (y.sum(axis=1) == 2)
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        mask = (y[:, curve_idx] == 1) & (y[:, lower_idx] == 1)
        idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        raise ValueError("No sample found containing both Driving(curve) and Lifting(lowering).")
    return int(idxs[len(idxs) // 2])


def rmse_against_baseline(
    baseline_series: np.ndarray,
    baseline_idx: np.ndarray,
    target_series: np.ndarray,
    target_idx: np.ndarray,
    sensor_cols: List[str],
) -> Tuple[float, Dict[str, float]]:
    t_base = baseline_idx / RAW_FREQ
    t_tar = target_idx / RAW_FREQ
    per_sensor: Dict[str, float] = {}
    vals = []
    for j, s in enumerate(sensor_cols):
        if len(t_tar) < 2:
            per_sensor[s] = float("nan")
            continue
        mask = (t_base >= t_tar.min()) & (t_base <= t_tar.max())
        if not np.any(mask):
            per_sensor[s] = float("nan")
            continue
        y_interp = np.interp(t_base[mask], t_tar, target_series[:, j])
        err = y_interp - baseline_series[mask, j]
        rmse = float(np.sqrt(np.mean(err * err)))
        per_sensor[s] = rmse
        vals.append(rmse)
    global_rmse = float(np.mean(vals)) if vals else float("nan")
    return global_rmse, per_sensor


def build_outputs(payload: Dict, sample_idx_global: int):
    sensor_cols = list(payload["sensor_cols"])
    label_cols = list(payload["label_cols"])
    X = payload["X"][sample_idx_global].astype(np.float64)
    y = payload["y"][sample_idx_global]
    exp = int(payload["experiment"][sample_idx_global])
    src = int(payload["source_index"][sample_idx_global])
    start_idx = int(payload["start_idx"][sample_idx_global])
    label_text = format_label_text(y, label_cols)

    base_title = (
        f"sample_idx={sample_idx_global} | exp={exp} | source={src} | "
        f"start_idx={start_idx} | labels={label_text}"
    )

    # Keep all variants in memory for later comparison.
    cfg_data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    out_paths: Dict[str, str] = {}

    # 1) Original
    cfg_data["original_2000hz"] = (X, np.arange(len(X), dtype=np.float64))
    p = OUT_DIR / "01_original_2000hz.png"
    plot_nine_axis(
        series=cfg_data["original_2000hz"][0],
        sample_idx=cfg_data["original_2000hz"][1],
        sensor_cols=sensor_cols,
        out_path=p,
        title=f"Original 2000Hz | {base_title}",
    )
    out_paths["original_2000hz"] = str(p)

    # 2) Interval 400 / 100
    for hz, factor, file_no in [(400, 5, 2), (100, 20, 3)]:
        key = f"interval_{hz}hz"
        cfg_data[key] = interval_downsample(X, factor=factor)
        p = OUT_DIR / f"{file_no:02d}_interval_{hz}hz.png"
        plot_nine_axis(
            series=cfg_data[key][0],
            sample_idx=cfg_data[key][1],
            sensor_cols=sensor_cols,
            out_path=p,
            title=f"Interval downsample to {hz}Hz (factor={factor}) | {base_title}",
        )
        out_paths[key] = str(p)

    # 3) Fixed window=40, step=5/10/20/40
    for file_no, st in enumerate([5, 10, 20, 40], start=4):
        key = f"w40_s{st}"
        cfg_data[key] = sliding_window_downsample(X, window=40, step=st)
        p = OUT_DIR / f"{file_no:02d}_slide_w40_s{st}.png"
        plot_nine_axis(
            series=cfg_data[key][0],
            sample_idx=cfg_data[key][1],
            sensor_cols=sensor_cols,
            out_path=p,
            title=f"Sliding-window mean | window=40 step={st} | {base_title}",
        )
        out_paths[key] = str(p)

    # 4) Fixed step=10, window=10/20/30/40
    for file_no, w in enumerate([10, 20, 30, 40], start=8):
        key = f"w{w}_s10"
        cfg_data[key] = sliding_window_downsample(X, window=w, step=10)
        p = OUT_DIR / f"{file_no:02d}_slide_w{w}_s10.png"
        plot_nine_axis(
            series=cfg_data[key][0],
            sample_idx=cfg_data[key][1],
            sensor_cols=sensor_cols,
            out_path=p,
            title=f"Sliding-window mean | window={w} step=10 | {base_title}",
        )
        if w == 40:
            out_paths["w40_s10_repeat"] = str(p)
        else:
            out_paths[key] = str(p)

    # Comparison centered at W40S10
    baseline_key = "w40_s10"
    group1 = ["w40_s5", "w40_s10", "w40_s20", "w40_s40"]
    group2 = ["w10_s10", "w20_s10", "w30_s10", "w40_s10"]
    color_map = {
        "w40_s5": "#1f77b4",
        "w40_s10": "#111111",  # baseline
        "w40_s20": "#d62728",
        "w40_s40": "#2ca02c",
        "w10_s10": "#9467bd",
        "w20_s10": "#ff7f0e",
        "w30_s10": "#17becf",
    }

    p = OUT_DIR / "12_compare_fixed_w40_center_w40s10.png"
    plot_overlay_comparison(
        cfg_order=group1,
        cfg_data=cfg_data,
        cfg_colors=color_map,
        baseline_key=baseline_key,
        sensor_cols=sensor_cols,
        out_path=p,
        title=f"Comparison around W40S10 | fixed window=40, varying step (5/10/20/40) | {base_title}",
    )
    out_paths["compare_fixed_w40"] = str(p)

    p = OUT_DIR / "13_compare_fixed_s10_center_w40s10.png"
    plot_overlay_comparison(
        cfg_order=group2,
        cfg_data=cfg_data,
        cfg_colors=color_map,
        baseline_key=baseline_key,
        sensor_cols=sensor_cols,
        out_path=p,
        title=f"Comparison around W40S10 | fixed step=10, varying window (10/20/30/40) | {base_title}",
    )
    out_paths["compare_fixed_s10"] = str(p)

    # Quantitative comparison (RMSE against baseline).
    rows = []
    base_series, base_idx = cfg_data[baseline_key]
    for grp_name, cfgs in [
        ("fixed_window40_vary_step", group1),
        ("fixed_step10_vary_window", group2),
    ]:
        for key in cfgs:
            if key == baseline_key:
                continue
            g_rmse, per_sensor = rmse_against_baseline(
                baseline_series=base_series,
                baseline_idx=base_idx,
                target_series=cfg_data[key][0],
                target_idx=cfg_data[key][1],
                sensor_cols=sensor_cols,
            )
            row = {"group": grp_name, "config": key, "global_rmse_mean9axis": g_rmse}
            for s, v in per_sensor.items():
                row[f"rmse_{s}"] = v
            rows.append(row)
    rmse_df = __import__("pandas").DataFrame(rows)
    rmse_csv = OUT_DIR / "comparison_rmse_vs_w40s10.csv"
    rmse_df.to_csv(rmse_csv, index=False, encoding="utf-8-sig")
    out_paths["rmse_csv"] = str(rmse_csv)

    meta = {
        "sample_idx": sample_idx_global,
        "experiment": exp,
        "source_index": src,
        "start_idx": start_idx,
        "label_text": label_text,
        "sensor_cols": sensor_cols,
        "label_cols": label_cols,
        "comparison_baseline": baseline_key,
    }
    (OUT_DIR / "selected_sample_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_paths, rmse_df


def _md_table_from_rmse(rmse_df) -> List[str]:
    lines = []
    if rmse_df.empty:
        return lines
    show = rmse_df[["group", "config", "global_rmse_mean9axis"]].copy()
    show = show.sort_values(["group", "global_rmse_mean9axis"])
    lines.append("| Group | Config | Global RMSE (mean over 9 axes) |")
    lines.append("|---|---:|---:|")
    for _, r in show.iterrows():
        lines.append(
            f"| {r['group']} | `{r['config']}` | {float(r['global_rmse_mean9axis']):.6f} |"
        )
    return lines


def write_observation_md(out_paths: Dict[str, str], rmse_df) -> None:
    md = []
    md.append("# Step250 Multi-label Sample Visualization Notes")
    md.append("")
    md.append("选取样本：`Driving(curve) + Lifting(lowering)`，对同一窗口做不同采样策略对比。")
    md.append("")
    md.append("## 对比方式说明")
    md.append("")
    md.append("- 颜色按**配置**统一，不按信号变化。")
    md.append("- 对比中心为 `W40S10`（黑色粗线）。")
    md.append("- 新增两张总览图，避免来回翻单图。")
    md.append("")
    md.append("## 关键观察")
    md.append("")
    md.append("1. 固定 `window=40` 时，`step` 越小（5/10）曲线更密、更接近原始；`step=40` 最稀疏。")
    md.append("2. 固定 `step=10` 时，`window` 越大（10->40）平滑越强，峰值削弱更明显。")
    md.append("3. `W40S10` 作为中间配置，在保留主要形态与抑噪之间较平衡。")
    md.append("")
    md.append("## 中心对比图（推荐先看）")
    md.append("")
    md.append("### A. 固定 window=40，step 变化（以 W40S10 为中心）")
    md.append("")
    md.append("![compare fixed w40](12_compare_fixed_w40_center_w40s10.png)")
    md.append("")
    md.append("### B. 固定 step=10，window 变化（以 W40S10 为中心）")
    md.append("")
    md.append("![compare fixed s10](13_compare_fixed_s10_center_w40s10.png)")
    md.append("")
    md.append("## 定量对比（相对 W40S10 的 RMSE）")
    md.append("")
    md.extend(_md_table_from_rmse(rmse_df))
    md.append("")
    md.append("## 单图索引")
    md.append("")
    ordered = [
        ("01 原始 2000Hz", "original_2000hz"),
        ("02 等间隔 400Hz", "interval_400hz"),
        ("03 等间隔 100Hz", "interval_100hz"),
        ("04 滑窗 w40 s5", "w40_s5"),
        ("05 滑窗 w40 s10", "w40_s10"),
        ("06 滑窗 w40 s20", "w40_s20"),
        ("07 滑窗 w40 s40", "w40_s40"),
        ("08 滑窗 w10 s10", "w10_s10"),
        ("09 滑窗 w20 s10", "w20_s10"),
        ("10 滑窗 w30 s10", "w30_s10"),
        ("11 滑窗 w40 s10", "w40_s10_repeat"),
    ]
    for title, key in ordered:
        rel = Path(out_paths[key]).name
        md.append(f"### {title}")
        md.append("")
        md.append(f"![{title}]({rel})")
        md.append("")

    (OUT_DIR / "OBSERVATIONS.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("rb") as f:
        payload = pickle.load(f)
    idx = pick_sample(payload)
    out_paths, rmse_df = build_outputs(payload, idx)
    write_observation_md(out_paths, rmse_df)
    print(f"[INFO] output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
