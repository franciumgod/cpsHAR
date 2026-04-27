import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = Path("output") / "none_by_exp_analysis"
    counts = pd.read_csv(base / "category_counts_by_exp.csv")
    runs = pd.read_csv(base / "category_runs_by_exp.csv")
    return counts, runs


def collect_curve_categories(counts: pd.DataFrame) -> Tuple[str, List[str], List[str]]:
    curve_single = "Driving(curve)"
    multi_curve = sorted(
        [
            c
            for c in counts["category_name"].dropna().unique().tolist()
            if c.startswith("MULTI: ") and "Driving(curve)" in c
        ]
    )
    curve_family = [curve_single] + multi_curve
    return curve_single, multi_curve, curve_family


def make_curve_family_share_plot(
    out_path: Path,
    counts: pd.DataFrame,
    curve_single: str,
    multi_curve: List[str],
) -> pd.DataFrame:
    curve_family = [curve_single] + multi_curve
    agg = (
        counts.groupby(["step", "experiment", "category_name"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "windows"})
    )
    total = (
        counts.groupby(["step", "experiment"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "total_windows"})
    )

    sub = agg[agg["category_name"].isin(curve_family)].copy()
    sub = sub.merge(total, on=["step", "experiment"], how="left")
    sum_curve = (
        sub.groupby(["step", "experiment"], as_index=False)["windows"]
        .sum()
        .rename(columns={"windows": "curve_family_windows"})
    )
    sub = sub.merge(sum_curve, on=["step", "experiment"], how="left")
    sub["ratio_in_curve_family"] = np.where(
        sub["curve_family_windows"] > 0,
        sub["windows"] / sub["curve_family_windows"],
        0.0,
    )
    sub["curve_family_ratio_in_all"] = np.where(
        sub["total_windows"] > 0,
        sub["curve_family_windows"] / sub["total_windows"],
        0.0,
    )

    steps = sorted(sub["step"].unique().tolist())
    exps = sorted(sub["experiment"].unique().tolist())
    cats = [curve_single] + multi_curve

    color_map = {
        curve_single: "#1f77b4",
    }
    palette = ["#d62728", "#9467bd", "#2ca02c", "#ff7f0e"]
    for i, c in enumerate(multi_curve):
        color_map[c] = palette[i % len(palette)]

    fig, axes = plt.subplots(1, len(steps), figsize=(6.2 * len(steps), 5), sharey=True)
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        ss = sub[sub["step"] == step]
        x = np.arange(len(exps))
        bottom = np.zeros(len(exps), dtype=np.float64)

        for cat in cats:
            vals = []
            for exp in exps:
                r = ss[(ss["experiment"] == exp) & (ss["category_name"] == cat)]
                vals.append(float(r["ratio_in_curve_family"].iloc[0]) if len(r) > 0 else 0.0)
            vals = np.array(vals, dtype=np.float64)
            ax.bar(x, vals, bottom=bottom, color=color_map[cat], label=cat, alpha=0.92)
            bottom += vals

        for i, exp in enumerate(exps):
            rr = ss[ss["experiment"] == exp]
            ratio_all = float(rr["curve_family_ratio_in_all"].iloc[0]) if len(rr) > 0 else 0.0
            ax.text(
                i,
                1.01,
                f"{ratio_all*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"exp{int(e)}" for e in exps])
        ax.set_ylim(0.0, 1.08)
        ax.set_title(f"step={step}")
        ax.grid(axis="y", alpha=0.25)
        ax.set_xlabel("Experiment")
    axes[0].set_ylabel("Composition inside curve-family windows")
    fig.suptitle(
        "Driving(curve) vs curve-related multi-label composition\n"
        "(text above bars: curve-family share in all windows)",
        fontsize=13,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(cats)), bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return sub.sort_values(["step", "experiment", "category_name"])


def build_transition_table(
    runs: pd.DataFrame,
    multi_curve: List[str],
) -> pd.DataFrame:
    rows = []
    for (step, source, exp), g in runs.groupby(["step", "source_index", "experiment"], sort=False):
        g = g.sort_values("run_start_win_idx").reset_index(drop=True)
        cats = g["category_name"].tolist()
        for i, cat in enumerate(cats):
            if cat not in multi_curve:
                continue
            prev_cat = cats[i - 1] if i > 0 else "__START__"
            next_cat = cats[i + 1] if i + 1 < len(cats) else "__END__"
            rows.append(
                {
                    "step": int(step),
                    "source_index": int(source),
                    "experiment": int(exp),
                    "multi_category": cat,
                    "prev_category": prev_cat,
                    "next_category": next_cat,
                    "run_len": int(g.loc[i, "run_len"]),
                }
            )
    return pd.DataFrame(rows)


def make_adjacent_curve_ratio_plot(
    out_path: Path,
    trans: pd.DataFrame,
    curve_single: str,
) -> pd.DataFrame:
    if len(trans) == 0:
        return pd.DataFrame()
    data = []
    for (step, multi), g in trans.groupby(["step", "multi_category"]):
        prev_curve = float((g["prev_category"] == curve_single).mean())
        next_curve = float((g["next_category"] == curve_single).mean())
        data.append(
            {
                "step": int(step),
                "multi_category": multi,
                "prev_is_curve_ratio": prev_curve,
                "next_is_curve_ratio": next_curve,
                "n_runs": int(len(g)),
            }
        )
    df = pd.DataFrame(data).sort_values(["step", "multi_category"])

    steps = sorted(df["step"].unique().tolist())
    multis = sorted(df["multi_category"].unique().tolist())
    x = np.arange(len(multis))
    width = 0.34

    fig, axes = plt.subplots(1, len(steps), figsize=(6.4 * len(steps), 5), sharey=True)
    if len(steps) == 1:
        axes = [axes]

    for ax, step in zip(axes, steps):
        ss = df[df["step"] == step]
        prev_vals = [float(ss[ss["multi_category"] == m]["prev_is_curve_ratio"].iloc[0]) for m in multis]
        next_vals = [float(ss[ss["multi_category"] == m]["next_is_curve_ratio"].iloc[0]) for m in multis]
        n_vals = [int(ss[ss["multi_category"] == m]["n_runs"].iloc[0]) for m in multis]

        ax.bar(x - width / 2, prev_vals, width=width, color="#1f77b4", label="Prev is Driving(curve)")
        ax.bar(x + width / 2, next_vals, width=width, color="#ff7f0e", label="Next is Driving(curve)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("MULTI: ", "") for m in multis], rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.set_title(f"step={step}")
        for i, n in enumerate(n_vals):
            ax.text(i, 1.01, f"n={n}", ha="center", va="bottom", fontsize=9)

    axes[0].set_ylabel("Ratio")
    fig.suptitle("How often curve-related multi-label runs are adjacent to Driving(curve)", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return df


def make_context_topn_plot(
    out_path: Path,
    trans: pd.DataFrame,
    top_n: int = 6,
) -> pd.DataFrame:
    if len(trans) == 0:
        return pd.DataFrame()
    rec = []
    for side in ["prev_category", "next_category"]:
        tmp = (
            trans.groupby(["step", "multi_category", side], as_index=False)["run_len"]
            .count()
            .rename(columns={"run_len": "count", side: "context_category"})
        )
        tmp["context_side"] = "prev" if side == "prev_category" else "next"
        rec.append(tmp)
    ctx = pd.concat(rec, ignore_index=True)

    out_rows = []
    for (step, multi, side), g in ctx.groupby(["step", "multi_category", "context_side"]):
        g = g.sort_values("count", ascending=False)
        total = int(g["count"].sum())
        for _, r in g.head(top_n).iterrows():
            out_rows.append(
                {
                    "step": int(step),
                    "multi_category": multi,
                    "context_side": side,
                    "context_category": r["context_category"],
                    "count": int(r["count"]),
                    "ratio": float(r["count"] / total) if total > 0 else 0.0,
                }
            )
    top_df = pd.DataFrame(out_rows)

    steps = sorted(top_df["step"].unique().tolist())
    fig, axes = plt.subplots(len(steps), 2, figsize=(14, 4.8 * len(steps)))
    if len(steps) == 1:
        axes = np.array([axes])

    for row_i, step in enumerate(steps):
        for col_i, side in enumerate(["prev", "next"]):
            ax = axes[row_i, col_i]
            ss = top_df[(top_df["step"] == step) & (top_df["context_side"] == side)]
            if len(ss) == 0:
                ax.set_axis_off()
                continue

            labels = []
            values = []
            colors = []
            for multi in sorted(ss["multi_category"].unique().tolist()):
                g = ss[ss["multi_category"] == multi].sort_values("ratio", ascending=True)
                for _, r in g.iterrows():
                    labels.append(f"{multi.replace('MULTI: ', '')} | {r['context_category']}")
                    values.append(float(r["ratio"]))
                    colors.append("#1f77b4" if "Driving(curve)" in str(r["context_category"]) else "#9e9e9e")

            y = np.arange(len(labels))
            ax.barh(y, values, color=colors, alpha=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlim(0, 1.0)
            ax.grid(axis="x", alpha=0.25)
            ax.set_title(f"step={step} | {side} context top-{top_n}")
            ax.set_xlabel("Ratio inside each (multi, side)")

    fig.suptitle(
        "Top context categories around curve-related multi-label runs\n"
        "(blue bars indicate contexts containing Driving(curve))",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return top_df


def make_interactive_context_plot(
    out_path: Path,
    trans: pd.DataFrame,
    top_n: int = 8,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    if len(trans) == 0:
        return

    rows = []
    for side_col, side_name in [("prev_category", "prev"), ("next_category", "next")]:
        g = (
            trans.groupby(["step", "multi_category", side_col], as_index=False)["run_len"]
            .count()
            .rename(columns={"run_len": "count", side_col: "context_category"})
        )
        g["side"] = side_name
        rows.append(g)
    ctx = pd.concat(rows, ignore_index=True)

    steps = sorted(ctx["step"].unique().tolist())
    multis = sorted(ctx["multi_category"].unique().tolist())

    fig = go.Figure()
    trace_keys: List[Tuple[int, str, str]] = []
    color_map = {
        multis[0]: "#1f77b4" if len(multis) > 0 else "#1f77b4",
        multis[1]: "#d62728" if len(multis) > 1 else "#d62728",
    }
    for step in steps:
        for side in ["prev", "next"]:
            ss = ctx[(ctx["step"] == step) & (ctx["side"] == side)]
            top_ctx = (
                ss.groupby("context_category", as_index=False)["count"]
                .sum()
                .sort_values("count", ascending=False)
                .head(top_n)["context_category"]
                .tolist()
            )
            ss = ss[ss["context_category"].isin(top_ctx)]
            for m in multis:
                mm = ss[ss["multi_category"] == m]
                x = mm["context_category"].tolist()
                y = mm["count"].tolist()
                fig.add_trace(
                    go.Bar(
                        x=x,
                        y=y,
                        name=m.replace("MULTI: ", ""),
                        marker_color=color_map.get(m, "#7f7f7f"),
                        visible=(step == steps[0] and side == "prev"),
                        hovertemplate=(
                            "step=" + str(step) + "<br>"
                            "side=" + side + "<br>"
                            "multi=" + m + "<br>"
                            "context=%{x}<br>"
                            "count=%{y}<extra></extra>"
                        ),
                    )
                )
                trace_keys.append((int(step), side, m))

    buttons = []
    for step in steps:
        for side in ["prev", "next"]:
            vis = [k[0] == int(step) and k[1] == side for k in trace_keys]
            buttons.append(
                dict(
                    label=f"step={step} | {side}",
                    method="update",
                    args=[
                        {"visible": vis},
                        {
                            "title": f"Curve-multi context distribution ({side}, step={step})",
                            "xaxis": {"title": f"{side} context category"},
                            "yaxis": {"title": "Run count"},
                        },
                    ],
                )
            )

    fig.update_layout(
        barmode="group",
        title=f"Curve-multi context distribution (prev, step={steps[0]})",
        xaxis_title="Context category",
        yaxis_title="Run count",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.16,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
        legend=dict(orientation="h", x=0.0, y=1.04),
        margin=dict(l=60, r=30, t=110, b=140),
        template="plotly_white",
        width=1200,
        height=620,
    )
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def write_markdown_report(
    out_path: Path,
    share_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    top_ctx_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Curve-related Multi-label Analysis")
    lines.append("")
    lines.append("目标：分析左下角多标签（含 `Driving(curve)`）与 `Driving(curve)` 单标签的关系。")
    lines.append("")
    lines.append("## 图 1")
    lines.append("")
    lines.append("![图1](curve_multilabel_analysis/curve_family_share_by_exp.png)")
    lines.append("")
    lines.append("- 含义：在 `Driving(curve)` 家族窗口内部，比较 `Driving(curve)` 与两类 `curve+lifting` 多标签的构成。")
    lines.append("- 读图：柱顶百分比是该家族在全部窗口中的占比。")
    lines.append("")

    if len(adj_df) > 0:
        lines.append("## 图 2")
        lines.append("")
        lines.append("![图2](curve_multilabel_analysis/curve_multi_adjacent_curve_ratio.png)")
        lines.append("")
        lines.append("- 含义：每个多标签段前后相邻类别是否为 `Driving(curve)` 的比例。")
        lines.append("- 结论（摘要）：")
        for _, r in adj_df.sort_values(["step", "multi_category"]).iterrows():
            lines.append(
                f"  - step={int(r['step'])} | {r['multi_category']} | "
                f"prev={r['prev_is_curve_ratio']*100:.1f}% | next={r['next_is_curve_ratio']*100:.1f}% | n={int(r['n_runs'])}"
            )
        lines.append("")

    if len(top_ctx_df) > 0:
        lines.append("## 图 3")
        lines.append("")
        lines.append("![图3](curve_multilabel_analysis/curve_multi_context_topn.png)")
        lines.append("")
        lines.append("- 含义：多标签段前后上下文 Top-N 分布（蓝色表示上下文本身含 `Driving(curve)`）。")
        lines.append("")
        lines.append("## 交互图")
        lines.append("")
        lines.append("- 文件：`curve_multilabel_analysis/curve_multi_context_interactive.html`")
        lines.append("- 可切换 `step` 和 `prev/next` 观察上下文计数分布。")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path("output") / "curve_multilabel_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, runs = load_inputs()
    curve_single, multi_curve, _ = collect_curve_categories(counts)
    if not multi_curve:
        raise ValueError("No multi-label categories containing Driving(curve) found.")

    share_df = make_curve_family_share_plot(
        out_path=out_dir / "curve_family_share_by_exp.png",
        counts=counts,
        curve_single=curve_single,
        multi_curve=multi_curve,
    )
    share_df.to_csv(out_dir / "curve_family_share_by_exp.csv", index=False, encoding="utf-8-sig")

    trans_df = build_transition_table(runs, multi_curve=multi_curve)
    trans_df.to_csv(out_dir / "curve_multi_transitions.csv", index=False, encoding="utf-8-sig")

    adj_df = make_adjacent_curve_ratio_plot(
        out_path=out_dir / "curve_multi_adjacent_curve_ratio.png",
        trans=trans_df,
        curve_single=curve_single,
    )
    adj_df.to_csv(out_dir / "curve_multi_adjacent_curve_ratio.csv", index=False, encoding="utf-8-sig")

    top_ctx_df = make_context_topn_plot(
        out_path=out_dir / "curve_multi_context_topn.png",
        trans=trans_df,
        top_n=6,
    )
    top_ctx_df.to_csv(out_dir / "curve_multi_context_topn.csv", index=False, encoding="utf-8-sig")
    make_interactive_context_plot(
        out_path=out_dir / "curve_multi_context_interactive.html",
        trans=trans_df,
        top_n=8,
    )

    write_markdown_report(
        out_path=out_dir / "README_curve_multilabel_analysis.md",
        share_df=share_df,
        adj_df=adj_df,
        top_ctx_df=top_ctx_df,
    )

    print(f"[INFO] output dir: {out_dir}")


if __name__ == "__main__":
    main()
