import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path("output") / "none_by_exp_analysis"
    counts = pd.read_csv(base / "category_counts_by_exp.csv")
    runs = pd.read_csv(base / "category_runs_by_exp.csv")
    return counts, runs


def get_multis(counts: pd.DataFrame) -> List[str]:
    return sorted(
        [
            c
            for c in counts["category_name"].dropna().unique().tolist()
            if str(c).startswith("MULTI: ")
        ]
    )


def build_total_windows_table(counts: pd.DataFrame) -> pd.DataFrame:
    return (
        counts.groupby(["step", "experiment"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total_windows"})
    )


def build_multi_window_table(counts: pd.DataFrame, multis: List[str]) -> pd.DataFrame:
    total = build_total_windows_table(counts)
    sub = (
        counts[counts["category_name"].isin(multis)]
        .groupby(["step", "experiment", "category_name"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "multi_windows"})
    )
    sub = sub.merge(total, on=["step", "experiment"], how="left")
    sub["ratio_in_exp"] = np.where(sub["total_windows"] > 0, sub["multi_windows"] / sub["total_windows"], 0.0)
    return sub


def build_transition_table(runs: pd.DataFrame, multis: List[str]) -> pd.DataFrame:
    rows = []
    for (step, source, exp), g in runs.groupby(["step", "source_index", "experiment"], sort=False):
        g = g.sort_values("run_start_win_idx").reset_index(drop=True)
        cat_seq = g["category_name"].tolist()
        for i, cat in enumerate(cat_seq):
            if cat not in multis:
                continue
            prev_cat = cat_seq[i - 1] if i > 0 else "__START__"
            next_cat = cat_seq[i + 1] if i + 1 < len(cat_seq) else "__END__"
            rows.append(
                {
                    "step": int(step),
                    "source_index": int(source),
                    "experiment": int(exp),
                    "multi_category": cat,
                    "run_start_win_idx": int(g.loc[i, "run_start_win_idx"]),
                    "run_end_win_idx": int(g.loc[i, "run_end_win_idx"]),
                    "run_len": int(g.loc[i, "run_len"]),
                    "prev_category": prev_cat,
                    "next_category": next_cat,
                }
            )
    return pd.DataFrame(rows)


def attach_position_ratio(trans: pd.DataFrame, total_windows: pd.DataFrame) -> pd.DataFrame:
    if len(trans) == 0:
        return trans.copy()
    t = trans.merge(total_windows, on=["step", "experiment"], how="left")
    midpoint = (t["run_start_win_idx"].astype(np.float64) + t["run_end_win_idx"].astype(np.float64)) / 2.0
    denom = np.maximum(1.0, t["total_windows"].astype(np.float64) - 1.0)
    t["midpoint_ratio"] = midpoint / denom
    return t


def short_multi_name(m: str) -> str:
    return m.replace("MULTI: ", "")


def plot_share_heatmap(out_path: Path, multi_win: pd.DataFrame, multis: List[str], steps: List[int], exps: List[int]) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(6.2 * len(steps), 4.8), sharey=True)
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        mat = np.zeros((len(multis), len(exps)), dtype=np.float64)
        ann = np.zeros((len(multis), len(exps)), dtype=np.int64)
        ss = multi_win[multi_win["step"] == step]
        for i, m in enumerate(multis):
            for j, e in enumerate(exps):
                r = ss[(ss["category_name"] == m) & (ss["experiment"] == e)]
                if len(r) > 0:
                    mat[i, j] = float(r["ratio_in_exp"].iloc[0]) * 100.0
                    ann[i, j] = int(r["multi_windows"].iloc[0])
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}%\n(n={ann[i,j]})", ha="center", va="center", fontsize=8)
        ax.set_xticks(np.arange(len(exps)))
        ax.set_xticklabels([f"exp{e}" for e in exps])
        ax.set_yticks(np.arange(len(multis)))
        ax.set_yticklabels([short_multi_name(m) for m in multis], fontsize=8)
        ax.set_title(f"step={step}")
        ax.set_xlabel("Experiment")
    axes[0].set_ylabel("Multi-label composition")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Share in experiment (%)")
    fig.suptitle("Per-exp share of each multi-label composition", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.12, wspace=0.16)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_runlen_box(out_path: Path, trans: pd.DataFrame, multis: List[str], steps: List[int]) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(6.4 * len(steps), 4.8), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        ss = trans[trans["step"] == step]
        data = []
        labels = []
        for m in multis:
            vals = ss[ss["multi_category"] == m]["run_len"].to_numpy(dtype=np.float64)
            if len(vals) == 0:
                vals = np.array([0.0], dtype=np.float64)
            data.append(vals)
            labels.append(short_multi_name(m))
        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, showfliers=True)
        colors = ["#1f77b4", "#d62728", "#9467bd", "#2ca02c"]
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor(colors[i % len(colors)])
            b.set_alpha(0.55)
        ax.set_yscale("log")
        ax.set_title(f"step={step}")
        ax.set_xlabel("Multi-label composition")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Run length (log scale, in windows)")
    fig.suptitle("Run-length distribution per multi-label composition", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_side_heat_matrix(trans: pd.DataFrame, multis: List[str], side_col: str, top_n_ctx: int = 8):
    ctx_rank = (
        trans.groupby(side_col, as_index=False)["run_len"]
        .count()
        .rename(columns={"run_len": "count"})
        .sort_values("count", ascending=False)
        .head(top_n_ctx)
    )
    contexts = ctx_rank[side_col].tolist()
    mat = np.zeros((len(multis), len(contexts)), dtype=np.float64)
    for i, m in enumerate(multis):
        ss = trans[trans["multi_category"] == m]
        total = max(1, len(ss))
        for j, c in enumerate(contexts):
            mat[i, j] = float((ss[side_col] == c).sum()) / float(total)
    return contexts, mat


def plot_transition_heatmaps(out_path: Path, trans: pd.DataFrame, multis: List[str], steps: List[int]) -> None:
    fig, axes = plt.subplots(len(steps), 2, figsize=(13, 4.2 * len(steps)))
    if len(steps) == 1:
        axes = np.array([axes])
    for r, step in enumerate(steps):
        ss = trans[trans["step"] == step]
        for c, side in enumerate(["prev_category", "next_category"]):
            ax = axes[r, c]
            contexts, mat = _build_side_heat_matrix(ss, multis, side_col=side, top_n_ctx=8)
            im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.4, float(mat.max()) if mat.size else 0.4))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
            ax.set_yticks(np.arange(len(multis)))
            ax.set_yticklabels([short_multi_name(m) for m in multis], fontsize=8)
            ax.set_xticks(np.arange(len(contexts)))
            ax.set_xticklabels(contexts, rotation=30, ha="right", fontsize=8)
            ax.set_title(f"step={step} | {'prev' if side=='prev_category' else 'next'} context")
            if c == 0:
                ax.set_ylabel("Multi-label composition")
            ax.set_xlabel("Context category")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Ratio within each multi-label (run-level)")
    fig.suptitle("Top context transitions around each multi-label composition", fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.16, wspace=0.22, hspace=0.35)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_position_violin(out_path: Path, trans_pos: pd.DataFrame, multis: List[str], steps: List[int]) -> None:
    fig, axes = plt.subplots(1, len(steps), figsize=(6.4 * len(steps), 4.8), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        ss = trans_pos[trans_pos["step"] == step]
        data = []
        labels = []
        for m in multis:
            vals = ss[ss["multi_category"] == m]["midpoint_ratio"].to_numpy(dtype=np.float64)
            if len(vals) == 0:
                vals = np.array([0.0], dtype=np.float64)
            data.append(vals)
            labels.append(short_multi_name(m))
        vp = ax.violinplot(data, showmeans=True, showextrema=True)
        for b in vp["bodies"]:
            b.set_alpha(0.55)
            b.set_facecolor("#4c78a8")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"step={step}")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Multi-label composition")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Run midpoint position ratio in experiment")
    fig.suptitle("Where each multi-label tends to appear along timeline", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_interactive_dashboard(
    out_path: Path,
    multi_win: pd.DataFrame,
    trans_pos: pd.DataFrame,
    multis: List[str],
    steps: List[int],
    exps: List[int],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Per-exp share (%)",
            "Run length histogram",
            "Prev context top-8",
            "Next context top-8",
        ),
    )
    trace_meta: List[Tuple[int, str]] = []

    for step in steps:
        for m in multis:
            ss_win = multi_win[(multi_win["step"] == step) & (multi_win["category_name"] == m)]
            y_share = []
            for e in exps:
                r = ss_win[ss_win["experiment"] == e]
                y_share.append(float(r["ratio_in_exp"].iloc[0]) * 100.0 if len(r) > 0 else 0.0)
            fig.add_trace(
                go.Bar(
                    x=[f"exp{e}" for e in exps],
                    y=y_share,
                    marker_color="#1f77b4",
                    name="share",
                    showlegend=False,
                    visible=(step == steps[0] and m == multis[0]),
                    hovertemplate="exp=%{x}<br>share=%{y:.3f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

            ss_run = trans_pos[(trans_pos["step"] == step) & (trans_pos["multi_category"] == m)]
            fig.add_trace(
                go.Histogram(
                    x=ss_run["run_len"].tolist(),
                    nbinsx=20,
                    marker_color="#d62728",
                    name="run_len",
                    showlegend=False,
                    visible=(step == steps[0] and m == multis[0]),
                    hovertemplate="run_len=%{x}<br>count=%{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            for side_col, row, col, color in [
                ("prev_category", 2, 1, "#9467bd"),
                ("next_category", 2, 2, "#2ca02c"),
            ]:
                cc = (
                    ss_run.groupby(side_col, as_index=False)["run_len"]
                    .count()
                    .rename(columns={"run_len": "count", side_col: "context"})
                    .sort_values("count", ascending=False)
                    .head(8)
                )
                fig.add_trace(
                    go.Bar(
                        x=cc["context"].tolist(),
                        y=cc["count"].tolist(),
                        marker_color=color,
                        name=side_col,
                        showlegend=False,
                        visible=(step == steps[0] and m == multis[0]),
                        hovertemplate="context=%{x}<br>count=%{y}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            trace_meta.extend([(int(step), m)] * 4)

    buttons = []
    for step in steps:
        for m in multis:
            vis = [t_step == int(step) and t_multi == m for t_step, t_multi in trace_meta]
            buttons.append(
                dict(
                    label=f"step={step} | {short_multi_name(m)}",
                    method="update",
                    args=[
                        {"visible": vis},
                        {"title": f"Detailed multi-label dashboard | step={step} | {short_multi_name(m)}"},
                    ],
                )
            )

    fig.update_layout(
        title=f"Detailed multi-label dashboard | step={steps[0]} | {short_multi_name(multis[0])}",
        template="plotly_white",
        width=1300,
        height=760,
        margin=dict(l=60, r=40, t=110, b=90),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.15,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
    )
    fig.update_xaxes(title_text="Experiment", row=1, col=1)
    fig.update_yaxes(title_text="Share (%)", row=1, col=1)
    fig.update_xaxes(title_text="Run length (windows)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Prev context", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Next context", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def build_summary_tables(
    multi_win: pd.DataFrame,
    trans_pos: pd.DataFrame,
    multis: List[str],
    steps: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ctx_rows = []
    for step in steps:
        total_windows_step = int(multi_win[multi_win["step"] == step]["total_windows"].drop_duplicates().sum())
        for m in multis:
            w = multi_win[(multi_win["step"] == step) & (multi_win["category_name"] == m)]
            run = trans_pos[(trans_pos["step"] == step) & (trans_pos["multi_category"] == m)]
            multi_windows = int(w["multi_windows"].sum())
            ratio_all = (multi_windows / total_windows_step) if total_windows_step > 0 else 0.0
            run_lens = run["run_len"].to_numpy(dtype=np.float64)
            midpoint = run["midpoint_ratio"].to_numpy(dtype=np.float64)

            rows.append(
                {
                    "step": int(step),
                    "multi_category": m,
                    "multi_windows": multi_windows,
                    "ratio_in_all_windows": ratio_all,
                    "n_runs": int(len(run_lens)),
                    "run_len_mean": float(np.mean(run_lens)) if len(run_lens) > 0 else 0.0,
                    "run_len_median": float(np.median(run_lens)) if len(run_lens) > 0 else 0.0,
                    "run_len_p90": float(np.percentile(run_lens, 90)) if len(run_lens) > 0 else 0.0,
                    "midpoint_mean": float(np.mean(midpoint)) if len(midpoint) > 0 else 0.0,
                    "midpoint_median": float(np.median(midpoint)) if len(midpoint) > 0 else 0.0,
                }
            )

            for side_col, side_name in [("prev_category", "prev"), ("next_category", "next")]:
                g = (
                    run.groupby(side_col, as_index=False)["run_len"]
                    .count()
                    .rename(columns={"run_len": "count", side_col: "context_category"})
                    .sort_values("count", ascending=False)
                )
                total = max(1, int(g["count"].sum()))
                for _, r in g.head(6).iterrows():
                    ctx_rows.append(
                        {
                            "step": int(step),
                            "multi_category": m,
                            "context_side": side_name,
                            "context_category": r["context_category"],
                            "count": int(r["count"]),
                            "ratio": float(r["count"] / total),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(ctx_rows)


def write_markdown(
    out_path: Path,
    summary_df: pd.DataFrame,
    ctx_df: pd.DataFrame,
    multis: List[str],
    steps: List[int],
) -> None:
    lines = []
    lines.append("# Multi-label Composition Detailed Analysis")
    lines.append("")
    lines.append("每个多标签组合均按 step=1 / step=250 做了细粒度分析。")
    lines.append("")
    lines.append("## 可视化清单")
    lines.append("")
    lines.append("- `multilabel_share_heatmap.png`: 每个多标签在各 exp 的占比热力图")
    lines.append("- `multilabel_runlen_boxplot.png`: 每个多标签段长分布（log）")
    lines.append("- `multilabel_transition_heatmaps.png`: 每个多标签前后上下文转移比例")
    lines.append("- `multilabel_position_violin.png`: 每个多标签沿时间轴出现位置分布")
    lines.append("- `multilabel_detailed_dashboard.html`: 交互仪表盘（step+多标签切换）")
    lines.append("")

    for m in multis:
        lines.append(f"## {m}")
        lines.append("")
        for step in steps:
            s = summary_df[(summary_df["step"] == step) & (summary_df["multi_category"] == m)]
            if len(s) == 0:
                continue
            r = s.iloc[0]
            lines.append(f"### step={step}")
            lines.append(
                f"- 窗口数: {int(r['multi_windows'])} | 全部窗口占比: {r['ratio_in_all_windows']*100:.3f}% | 运行段数量: {int(r['n_runs'])}"
            )
            lines.append(
                f"- 段长: mean={r['run_len_mean']:.2f}, median={r['run_len_median']:.2f}, p90={r['run_len_p90']:.2f}"
            )
            lines.append(
                f"- 时序位置(0~1): mean={r['midpoint_mean']:.3f}, median={r['midpoint_median']:.3f}"
            )

            for side in ["prev", "next"]:
                c = ctx_df[
                    (ctx_df["step"] == step)
                    & (ctx_df["multi_category"] == m)
                    & (ctx_df["context_side"] == side)
                ].sort_values("count", ascending=False)
                top_txt = "; ".join(
                    [f"{x['context_category']}({x['ratio']*100:.1f}%,n={int(x['count'])})" for _, x in c.head(4).iterrows()]
                )
                lines.append(f"- {side} Top4: {top_txt}")
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path("output") / "multilabel_detailed_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, runs = load_data()
    multis = get_multis(counts)
    steps = sorted(counts["step"].unique().tolist())
    exps = sorted(counts["experiment"].unique().tolist())

    multi_win = build_multi_window_table(counts, multis)
    trans = build_transition_table(runs, multis)
    total_windows = build_total_windows_table(counts)
    trans_pos = attach_position_ratio(trans, total_windows)

    multi_win.to_csv(out_dir / "multi_windows_by_exp.csv", index=False, encoding="utf-8-sig")
    trans_pos.to_csv(out_dir / "multi_transitions_runs.csv", index=False, encoding="utf-8-sig")

    plot_share_heatmap(out_dir / "multilabel_share_heatmap.png", multi_win, multis, steps, exps)
    plot_runlen_box(out_dir / "multilabel_runlen_boxplot.png", trans_pos, multis, steps)
    plot_transition_heatmaps(out_dir / "multilabel_transition_heatmaps.png", trans_pos, multis, steps)
    plot_position_violin(out_dir / "multilabel_position_violin.png", trans_pos, multis, steps)

    build_interactive_dashboard(
        out_path=out_dir / "multilabel_detailed_dashboard.html",
        multi_win=multi_win,
        trans_pos=trans_pos,
        multis=multis,
        steps=steps,
        exps=exps,
    )

    summary_df, ctx_df = build_summary_tables(multi_win, trans_pos, multis, steps)
    summary_df.to_csv(out_dir / "multilabel_summary_stats.csv", index=False, encoding="utf-8-sig")
    ctx_df.to_csv(out_dir / "multilabel_top_contexts.csv", index=False, encoding="utf-8-sig")

    write_markdown(
        out_path=out_dir / "README_multilabel_detailed_analysis.md",
        summary_df=summary_df,
        ctx_df=ctx_df,
        multis=multis,
        steps=steps,
    )

    print(f"[INFO] output dir: {out_dir}")


if __name__ == "__main__":
    main()
