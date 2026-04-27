import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative


def load_single_superclass_order(report_path: Path, runs_df: pd.DataFrame) -> List[str]:
    if report_path.exists():
        obj = json.loads(report_path.read_text(encoding="utf-8"))
        order = obj.get("single_superclass_order")
        if isinstance(order, list) and order:
            return [str(x) for x in order]

    names = sorted(runs_df["category_name"].dropna().unique().tolist())
    return [n for n in names if n != "0-label" and not n.startswith("MULTI: ")]


def ordered_categories(all_names: List[str], single_superclass_order: List[str]) -> List[str]:
    names = sorted(set(all_names))
    multi = sorted([n for n in names if n.startswith("MULTI: ")])
    out: List[str] = []
    if "0-label" in names:
        out.append("0-label")
    for s in single_superclass_order:
        if s in names:
            out.append(s)
    out.extend(multi)
    for n in names:
        if n not in out:
            out.append(n)
    return out


def build_color_map(category_names: List[str], single_superclass_order: List[str]) -> Dict[str, str]:
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

    pool = qualitative.Alphabet + qualitative.Dark24 + qualitative.Light24
    used = set(color_map.values())
    p_idx = 0
    for name in category_names:
        if name in color_map:
            continue
        while p_idx < len(pool) and pool[p_idx] in used:
            p_idx += 1
        color = pool[p_idx % len(pool)]
        color_map[name] = color
        used.add(color)
        p_idx += 1
    return color_map


def build_figure(runs_df: pd.DataFrame, single_superclass_order: List[str]) -> go.Figure:
    steps = sorted(runs_df["step"].unique().tolist())
    exps = sorted(runs_df["experiment"].unique().tolist())
    exp_labels = [f"exp{int(x)}" for x in exps]

    cats = ordered_categories(runs_df["category_name"].dropna().tolist(), single_superclass_order)
    color_map = build_color_map(cats, single_superclass_order)

    fig = go.Figure()
    trace_step: List[int] = []
    x_max_by_step: Dict[int, int] = {}

    for step in steps:
        sub_step = runs_df[runs_df["step"] == step]
        x_max_by_step[int(step)] = int(sub_step["run_end_win_idx"].max()) if len(sub_step) > 0 else 1
        for cat in cats:
            sub = sub_step[sub_step["category_name"] == cat]
            x: List[float] = []
            y: List[str] = []
            custom: List[List[int]] = []

            for _, r in sub.iterrows():
                exp_label = f"exp{int(r['experiment'])}"
                s = int(r["run_start_win_idx"])
                e = int(r["run_end_win_idx"])
                ln = int(r["run_len"])
                x.extend([s, e, None])
                y.extend([exp_label, exp_label, None])
                custom.extend([[int(step), s, e, ln], [int(step), s, e, ln], [0, 0, 0, 0]])

            trace = go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": color_map[cat], "width": 10},
                name=cat,
                customdata=custom,
                hovertemplate=(
                    "category=%{fullData.name}<br>"
                    "exp=%{y}<br>"
                    "step=%{customdata[0]}<br>"
                    "start_idx=%{customdata[1]}<br>"
                    "end_idx=%{customdata[2]}<br>"
                    "run_len=%{customdata[3]}<extra></extra>"
                ),
                visible=(step == steps[0]),
            )
            fig.add_trace(trace)
            trace_step.append(int(step))

    buttons = []
    for step in steps:
        visible = [s == int(step) for s in trace_step]
        buttons.append(
            {
                "label": f"step={int(step)}",
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Interactive Category Ranges by Exp (step={int(step)})",
                        "xaxis": {"range": [0, max(1, x_max_by_step[int(step)])]},
                    },
                ],
            }
        )

    init_step = int(steps[0]) if steps else 1
    fig.update_layout(
        title=f"Interactive Category Ranges by Exp (step={init_step})",
        xaxis_title="Window index",
        yaxis_title="Experiment",
        yaxis={"categoryorder": "array", "categoryarray": exp_labels},
        xaxis={"range": [0, max(1, x_max_by_step.get(init_step, 1))], "rangeslider": {"visible": True}},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.01,
                "y": 1.15,
                "xanchor": "left",
                "yanchor": "top",
            }
        ],
        legend={"orientation": "v", "x": 1.02, "y": 1.0},
        margin={"l": 80, "r": 300, "t": 110, "b": 80},
        template="plotly_white",
        width=1500,
        height=580,
    )
    return fig


def main() -> None:
    out_dir = Path("output") / "none_by_exp_analysis"
    runs_path = out_dir / "category_runs_by_exp.csv"
    report_path = out_dir / "report.json"
    html_path = out_dir / "interactive_category_ranges.html"

    runs_df = pd.read_csv(runs_path)
    single_superclass_order = load_single_superclass_order(report_path, runs_df)
    fig = build_figure(runs_df, single_superclass_order)
    fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    print(f"[INFO] interactive html: {html_path}")


if __name__ == "__main__":
    main()
