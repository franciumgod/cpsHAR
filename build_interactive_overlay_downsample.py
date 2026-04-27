import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


RAW_FREQ = 2000.0
DATA_PATH = Path("data") / "cps_windows_2s_2000hz_step_250.pkl"
OUT_DIR = Path("output") / "step250_multilabel_sampling_viz"
OUT_HTML = OUT_DIR / "interactive_overlay_original_vs_downsample.html"


def interval_downsample(x: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray]:
    y = x[::factor]
    idx = np.arange(0, len(x), factor, dtype=np.int64).astype(np.float64)
    return y, idx


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


def load_or_pick_sample(payload: Dict) -> int:
    meta_path = OUT_DIR / "selected_sample_meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            idx = int(meta.get("sample_idx", -1))
            if 0 <= idx < len(payload["X"]):
                return idx
        except Exception:
            pass
    return pick_sample(payload)


def build_downsample_choices(x: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    choices: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    choices["interval_400Hz"] = interval_downsample(x, factor=5)
    choices["interval_100Hz"] = interval_downsample(x, factor=20)
    for s in [5, 10, 20, 40]:
        choices[f"slide_w40_s{s}"] = sliding_window_downsample(x, window=40, step=s)
    for w in [10, 20, 30, 40]:
        choices[f"slide_w{w}_s10"] = sliding_window_downsample(x, window=w, step=10)
    return choices


def build_interactive_html(
    x_raw: np.ndarray,
    sensor_cols: List[str],
    choices: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title_meta: str,
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        raise RuntimeError("plotly is required for interactive visualization.") from e

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=sensor_cols,
        horizontal_spacing=0.04,
        vertical_spacing=0.08,
    )

    raw_t = np.arange(len(x_raw), dtype=np.float64) / RAW_FREQ
    sensor_colors = [
        "#1f77b4", "#d62728", "#2ca02c",
        "#9467bd", "#ff7f0e", "#17becf",
        "#8c564b", "#e377c2", "#bcbd22",
    ]

    # 1) Background original traces (always visible)
    for i, col in enumerate(sensor_cols):
        row = i // 3 + 1
        col_idx = i % 3 + 1
        fig.add_trace(
            go.Scatter(
                x=raw_t,
                y=x_raw[:, i],
                mode="lines",
                line=dict(color=sensor_colors[i], width=1.2),
                name=f"Original-{col}",
                legendgroup="original",
                showlegend=(i == 0),
                opacity=0.85,
                hovertemplate=f"{col}<br>t=%{{x:.4f}}s<br>raw=%{{y:.6f}}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )

    # 2) Overlay traces by each downsample choice (black)
    choice_names = list(choices.keys())
    overlay_trace_choice_idx: List[int] = []
    base_trace_count = len(sensor_cols)
    for c_idx, name in enumerate(choice_names):
        x_ds, idx_ds = choices[name]
        t_ds = idx_ds / RAW_FREQ
        for i, col in enumerate(sensor_cols):
            row = i // 3 + 1
            col_idx = i % 3 + 1
            trace_idx = len(fig.data)
            fig.add_trace(
                go.Scatter(
                    x=t_ds,
                    y=x_ds[:, i] if len(x_ds) > 0 else [],
                    mode="lines",
                    line=dict(color="#000000", width=1.8),
                    name=f"Overlay-{name}",
                    legendgroup="overlay",
                    showlegend=(i == 0),
                    visible=(c_idx == 0),
                    hovertemplate=f"{col}<br>t=%{{x:.4f}}s<br>{name}=%{{y:.6f}}<extra></extra>",
                ),
                row=row,
                col=col_idx,
            )
            overlay_trace_choice_idx.append(c_idx)

    # Build dropdown
    buttons = []
    total_traces = len(fig.data)
    for c_idx, name in enumerate(choice_names):
        visible = [True] * base_trace_count + [idx == c_idx for idx in overlay_trace_choice_idx]
        if len(visible) != total_traces:
            raise RuntimeError("Visibility vector length mismatch.")
        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"visible": visible},
                    {
                        "title": f"Original(2000Hz) + Overlay({name}) | {title_meta}",
                    },
                ],
            )
        )

    fig.update_layout(
        title=f"Original(2000Hz) + Overlay({choice_names[0]}) | {title_meta}",
        template="plotly_white",
        width=1500,
        height=980,
        margin=dict(l=50, r=30, t=110, b=60),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                x=0.01,
                y=1.12,
                xanchor="left",
                yanchor="top",
                showactive=True,
            )
        ],
        legend=dict(orientation="h", x=0.0, y=1.02),
    )

    for i in range(1, 10):
        fig.update_xaxes(title_text="Time (s)", row=(i - 1) // 3 + 1, col=(i - 1) % 3 + 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT_HTML), include_plotlyjs=True, full_html=True)


def main() -> None:
    with DATA_PATH.open("rb") as f:
        payload = pickle.load(f)

    idx = load_or_pick_sample(payload)
    x_raw = payload["X"][idx].astype(np.float64)
    y = payload["y"][idx]
    sensor_cols = list(payload["sensor_cols"])
    label_cols = list(payload["label_cols"])
    exp = int(payload["experiment"][idx])
    src = int(payload["source_index"][idx])
    start_idx = int(payload["start_idx"][idx])
    label_text = " + ".join([label_cols[i] for i, v in enumerate(y) if int(v) == 1]) or "0-label"
    title_meta = f"sample_idx={idx}, exp={exp}, source={src}, start_idx={start_idx}, labels={label_text}"

    choices = build_downsample_choices(x_raw)
    build_interactive_html(
        x_raw=x_raw,
        sensor_cols=sensor_cols,
        choices=choices,
        title_meta=title_meta,
    )
    print(f"[INFO] interactive html: {OUT_HTML}")


if __name__ == "__main__":
    main()
