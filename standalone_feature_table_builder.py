import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute


RAW_FREQ = 2000
DROP_COLS = ["Error", "Synchronization", "None", "transportation", "container"]
BASE_SENSOR_COLS = ["Acc.x", "Acc.y", "Acc.z", "Gyro.x", "Gyro.y", "Gyro.z", "Baro.x"]
SYNTHETIC_SENSOR_COLS = ["Acc.norm", "Gyro.norm"]
ORIGINAL_LABEL_COLS = [
    "Driving(straight)",
    "Driving(curve)",
    "Lifting(raising)",
    "Lifting(lowering)",
    "Standing",
    "Docking",
    "Forks(entering or leaving front)",
    "Forks(entering or leaving side)",
    "Wrapping",
    "Wrapping(preparation)",
]
SUPERCLASS_MAPPING = {
    "Driving(curve)": "Driving(curve)",
    "Driving(straight)": "Driving(straight)",
    "Lifting(lowering)": "Lifting(lowering)",
    "Lifting(raising)": "Lifting(raising)",
    "Wrapping": "Turntable wrapping",
    "Wrapping(preparation)": "Stationary processes",
    "Docking": "Stationary processes",
    "Forks(entering or leaving front)": "Stationary processes",
    "Forks(entering or leaving side)": "Stationary processes",
    "Standing": "Stationary processes",
}
SUPERCLASS_LABEL_COLS = sorted(set(SUPERCLASS_MAPPING.values()))
DEFAULT_SPECIAL_RULES = {
    "Driving(curve)": {"window_points": 4000, "step_points": 800},
    "Stationary processes": {"window_points": 4000, "step_points": 800},
    "Turntable wrapping": {"window_points": 4000, "step_points": 800},
    "Lifting(raising)": {"window_points": 2000, "step_points": 200},
    "Lifting(lowering)": {"window_points": 2000, "step_points": 200},
}
SPECIAL_PRIORITY = [
    "Lifting(raising)",
    "Lifting(lowering)",
    "Driving(curve)",
    "Stationary processes",
    "Turntable wrapping",
    "Driving(straight)",
]
RESOLUTION_SPECS = (
    ("high", RAW_FREQ),
    ("middle", 500),
    ("low", 100),
)


@dataclass
class WindowSpec:
    sample_id: int
    scenario: str
    experiment: int
    state_name: str
    start_idx: int
    end_idx: int
    real_points: int
    padded_points: int
    window_points: int
    step_points: int


def _clean_segment(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").reset_index(drop=True)


def _add_synthetic_axes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Acc.norm"] = np.sqrt(
        np.asarray(out["Acc.x"], dtype=np.float64) ** 2
        + np.asarray(out["Acc.y"], dtype=np.float64) ** 2
        + np.asarray(out["Acc.z"], dtype=np.float64) ** 2
    )
    out["Gyro.norm"] = np.sqrt(
        np.asarray(out["Gyro.x"], dtype=np.float64) ** 2
        + np.asarray(out["Gyro.y"], dtype=np.float64) ** 2
        + np.asarray(out["Gyro.z"], dtype=np.float64) ** 2
    )
    return out


def _apply_superclass_mapping(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for target in SUPERCLASS_LABEL_COLS:
        children = [child for child, parent in SUPERCLASS_MAPPING.items() if parent == target]
        existing = [c for c in children if c in out.columns]
        out[target] = out[existing].max(axis=1) if existing else 0
    out = out.drop(columns=[c for c in ORIGINAL_LABEL_COLS if c in out.columns], errors="ignore")
    return out


def _ensure_columns(df: pd.DataFrame, cols: Sequence[str], fill_value: float = 0.0) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = fill_value
    return out


def _resolve_label_space(df: pd.DataFrame, label_space: str) -> Tuple[pd.DataFrame, List[str]]:
    label_space = str(label_space).strip().lower()
    out = df.copy()
    out = _ensure_columns(out, ORIGINAL_LABEL_COLS, 0)
    if label_space == "superclass":
        out = _apply_superclass_mapping(out)
        out = _ensure_columns(out, SUPERCLASS_LABEL_COLS, 0)
        return out, list(SUPERCLASS_LABEL_COLS)
    return out, list(ORIGINAL_LABEL_COLS)


def _bool_string(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _state_names_from_label_matrix(label_matrix: np.ndarray, label_cols: Sequence[str]) -> np.ndarray:
    names = []
    for row in label_matrix:
        active = [label_cols[i] for i, v in enumerate(row) if int(v) == 1]
        names.append("+".join(active) if active else "NONE")
    return np.asarray(names, dtype=object)


def _find_state_segments(state_names: np.ndarray) -> List[Tuple[str, int, int]]:
    if len(state_names) == 0:
        return []
    change_idx = np.flatnonzero(state_names[1:] != state_names[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(state_names)]))
    return [(str(state_names[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _resolve_sampling_rule_for_state(
    state_name: str,
    default_window_points: int,
    default_step_points: int,
    special_mode: bool,
    sampling_rules: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[int, int]:
    if not special_mode:
        return int(default_window_points), int(default_step_points)

    merged_rules = dict(DEFAULT_SPECIAL_RULES)
    if sampling_rules:
        merged_rules.update(sampling_rules)

    active = set(state_name.split("+")) if state_name and state_name != "NONE" else set()
    for label_name in SPECIAL_PRIORITY:
        if label_name in active and label_name in merged_rules:
            rule = merged_rules[label_name]
            return int(rule["window_points"]), int(rule["step_points"])

    return int(default_window_points), int(default_step_points)


def _build_window_specs_for_segment(
    *,
    state_name: str,
    segment_start: int,
    segment_end: int,
    scenario: str,
    experiment: int,
    sample_id_start: int,
    default_window_points: int,
    default_step_points: int,
    special_mode: bool,
    allow_short_segment_padding: bool,
    include_tail_window: bool,
    sampling_rules: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[WindowSpec]:
    specs: List[WindowSpec] = []
    seg_len = int(segment_end - segment_start)
    if seg_len <= 0:
        return specs

    window_points, step_points = _resolve_sampling_rule_for_state(
        state_name=state_name,
        default_window_points=default_window_points,
        default_step_points=default_step_points,
        special_mode=special_mode,
        sampling_rules=sampling_rules,
    )

    sample_id = int(sample_id_start)
    if seg_len < window_points:
        if not allow_short_segment_padding:
            return specs
        specs.append(
            WindowSpec(
                sample_id=sample_id,
                scenario=str(scenario),
                experiment=int(experiment),
                state_name=str(state_name),
                start_idx=int(segment_start),
                end_idx=int(segment_end),
                real_points=int(seg_len),
                padded_points=int(window_points),
                window_points=int(window_points),
                step_points=int(step_points),
            )
        )
        return specs

    starts = list(range(segment_start, segment_end - window_points + 1, step_points))
    if include_tail_window:
        tail_start = int(segment_end - window_points)
        if not starts or starts[-1] != tail_start:
            starts.append(tail_start)

    starts = sorted(set(int(v) for v in starts))
    for local_idx, start_idx in enumerate(starts):
        end_idx = int(start_idx + window_points)
        specs.append(
            WindowSpec(
                sample_id=sample_id + local_idx,
                scenario=str(scenario),
                experiment=int(experiment),
                state_name=str(state_name),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                real_points=int(window_points),
                padded_points=int(window_points),
                window_points=int(window_points),
                step_points=int(step_points),
            )
        )
    return specs


def _iter_window_payloads(
    raw_df: pd.DataFrame,
    *,
    label_space: str,
    default_window_points: int,
    default_step_points: int,
    special_mode: bool,
    include_synthetic_axes: bool,
    allow_short_segment_padding: bool,
    include_tail_window: bool,
    sampling_rules: Optional[Dict[str, Dict[str, int]]] = None,
    max_samples: Optional[int] = None,
) -> Iterable[Dict]:
    sensor_cols = list(BASE_SENSOR_COLS) + (list(SYNTHETIC_SENSOR_COLS) if include_synthetic_axes else [])
    next_sample_id = 0

    for row in raw_df.itertuples(index=False):
        segment_df = _clean_segment(row.data)
        if include_synthetic_axes:
            segment_df = _add_synthetic_axes(segment_df)

        segment_df, label_cols = _resolve_label_space(segment_df, label_space)
        keep_cols = [c for c in sensor_cols + label_cols if c in segment_df.columns]
        segment_df = _ensure_columns(segment_df[keep_cols].copy(), sensor_cols + label_cols, 0.0)

        if special_mode:
            state_names = _state_names_from_label_matrix(
                segment_df[label_cols].to_numpy(dtype=np.int8, copy=False),
                label_cols,
            )
            state_segments = _find_state_segments(state_names)
        else:
            state_segments = [("GLOBAL", 0, len(segment_df))]

        for state_name, seg_start, seg_end in state_segments:
            specs = _build_window_specs_for_segment(
                state_name=state_name,
                segment_start=seg_start,
                segment_end=seg_end,
                scenario=getattr(row, "scenario", "unknown"),
                experiment=getattr(row, "experiment", -1),
                sample_id_start=next_sample_id,
                default_window_points=default_window_points,
                default_step_points=default_step_points,
                special_mode=special_mode,
                allow_short_segment_padding=allow_short_segment_padding,
                include_tail_window=include_tail_window,
                sampling_rules=sampling_rules,
            )
            next_sample_id += len(specs)
            for spec in specs:
                real_slice = segment_df.iloc[spec.start_idx:spec.end_idx].copy()
                sensor_array = real_slice[sensor_cols].to_numpy(dtype=np.float32, copy=False)
                label_array = real_slice[label_cols].to_numpy(dtype=np.int8, copy=False)

                if len(sensor_array) < spec.window_points:
                    pad_rows = int(spec.window_points - len(sensor_array))
                    if len(sensor_array) == 0:
                        sensor_array = np.zeros((spec.window_points, len(sensor_cols)), dtype=np.float32)
                    else:
                        sensor_array = np.pad(sensor_array, ((0, pad_rows), (0, 0)), mode="edge")

                label_end = label_array[-1] if len(label_array) > 0 else np.zeros((len(label_cols),), dtype=np.int8)
                label_ratio = (
                    label_array.mean(axis=0, dtype=np.float32)
                    if len(label_array) > 0
                    else np.zeros((len(label_cols),), dtype=np.float32)
                )
                active_count = int(np.sum(label_end))

                yield {
                    "sample_id": spec.sample_id,
                    "scenario": spec.scenario,
                    "experiment": spec.experiment,
                    "state_name": spec.state_name,
                    "start_idx": spec.start_idx,
                    "end_idx": spec.end_idx,
                    "real_points": spec.real_points,
                    "padded_points": spec.padded_points,
                    "window_points": spec.window_points,
                    "window_sec": float(spec.window_points / RAW_FREQ),
                    "step_points": spec.step_points,
                    "active_label_count": active_count,
                    "sensor_cols": sensor_cols,
                    "label_cols": label_cols,
                    "sensor_array": sensor_array,
                    "label_end": label_end,
                    "label_ratio": label_ratio,
                }

                if max_samples is not None and next_sample_id >= int(max_samples):
                    return


def _downsample_direct_from_raw(sensor_array: np.ndarray, target_freq: int) -> np.ndarray:
    if target_freq >= RAW_FREQ:
        return np.asarray(sensor_array, dtype=np.float32)

    factor = max(1, int(round(RAW_FREQ / float(target_freq))))
    if factor <= 1:
        return np.asarray(sensor_array, dtype=np.float32)

    n_rows, n_cols = sensor_array.shape
    usable = (n_rows // factor) * factor
    if usable == 0:
        return np.asarray(sensor_array, dtype=np.float32)

    reshaped = sensor_array[:usable].reshape(-1, factor, n_cols)
    pooled = reshaped.mean(axis=1, dtype=np.float32)
    if usable < n_rows:
        tail = sensor_array[usable:].mean(axis=0, keepdims=True, dtype=np.float32)
        pooled = np.vstack([pooled, tail])
    return np.asarray(pooled, dtype=np.float32)


def _safe_skew(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    value = skew(x, bias=False, nan_policy="omit")
    return float(0.0 if not np.isfinite(value) else value)


def _safe_kurtosis(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    value = kurtosis(x, fisher=True, bias=False, nan_policy="omit")
    return float(0.0 if not np.isfinite(value) else value)


def _zero_crossing_rate(x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    signs = np.signbit(x)
    return float(np.mean(signs[1:] != signs[:-1]))


def _linear_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    idx = np.arange(n, dtype=np.float64)
    idx_centered = idx - idx.mean()
    denom = np.sum(idx_centered * idx_centered)
    if denom <= 0:
        return 0.0
    y = x.astype(np.float64, copy=False)
    return float(np.sum(idx_centered * (y - y.mean())) / denom)


def _manual_time_features(signal_1d: np.ndarray, axis_name: str) -> Dict[str, float]:
    x = np.asarray(signal_1d, dtype=np.float64)
    if len(x) == 0:
        return {}

    abs_x = np.abs(x)
    q25, q50, q75 = np.percentile(x, [25, 50, 75])
    peaks, _ = find_peaks(x)
    troughs, _ = find_peaks(-x)

    feats = {
        f"{axis_name}__mean": float(np.mean(x)),
        f"{axis_name}__std": float(np.std(x)),
        f"{axis_name}__min": float(np.min(x)),
        f"{axis_name}__max": float(np.max(x)),
        f"{axis_name}__median": float(q50),
        f"{axis_name}__q25": float(q25),
        f"{axis_name}__q75": float(q75),
        f"{axis_name}__iqr": float(q75 - q25),
        f"{axis_name}__abs_mean": float(np.mean(abs_x)),
        f"{axis_name}__rms": float(np.sqrt(np.mean(x * x))),
        f"{axis_name}__energy": float(np.sum(x * x)),
        f"{axis_name}__mad": float(np.mean(np.abs(x - np.mean(x)))),
        f"{axis_name}__skew": _safe_skew(x),
        f"{axis_name}__kurtosis": _safe_kurtosis(x),
        f"{axis_name}__zero_cross_rate": _zero_crossing_rate(x),
        f"{axis_name}__peak_count": float(len(peaks)),
        f"{axis_name}__trough_count": float(len(troughs)),
        f"{axis_name}__monotonic_up_ratio": float(np.mean(np.diff(x) >= 0)) if len(x) > 1 else 0.0,
        f"{axis_name}__slope": _linear_slope(x),
    }

    for diff_order in (1, 2, 3):
        dx = np.diff(x, n=diff_order)
        prefix = f"{axis_name}__diff{diff_order}"
        if len(dx) == 0:
            feats[f"{prefix}__mean"] = 0.0
            feats[f"{prefix}__std"] = 0.0
            feats[f"{prefix}__abs_mean"] = 0.0
            feats[f"{prefix}__rms"] = 0.0
            feats[f"{prefix}__max"] = 0.0
            feats[f"{prefix}__min"] = 0.0
            feats[f"{prefix}__skew"] = 0.0
            feats[f"{prefix}__kurtosis"] = 0.0
        else:
            feats[f"{prefix}__mean"] = float(np.mean(dx))
            feats[f"{prefix}__std"] = float(np.std(dx))
            feats[f"{prefix}__abs_mean"] = float(np.mean(np.abs(dx)))
            feats[f"{prefix}__rms"] = float(np.sqrt(np.mean(dx * dx)))
            feats[f"{prefix}__max"] = float(np.max(dx))
            feats[f"{prefix}__min"] = float(np.min(dx))
            feats[f"{prefix}__skew"] = _safe_skew(dx)
            feats[f"{prefix}__kurtosis"] = _safe_kurtosis(dx)

    for lag in range(1, 31):
        if len(x) <= lag:
            feats[f"{axis_name}__lag{lag}__delta_mean"] = 0.0
            feats[f"{axis_name}__lag{lag}__delta_abs_mean"] = 0.0
            feats[f"{axis_name}__lag{lag}__autocorr"] = 0.0
            continue
        delta = x[lag:] - x[:-lag]
        feats[f"{axis_name}__lag{lag}__delta_mean"] = float(np.mean(delta))
        feats[f"{axis_name}__lag{lag}__delta_abs_mean"] = float(np.mean(np.abs(delta)))
        corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        feats[f"{axis_name}__lag{lag}__autocorr"] = float(0.0 if not np.isfinite(corr) else corr)

    return feats


def _manual_rfft_features(signal_1d: np.ndarray, axis_name: str, sample_rate: int) -> Dict[str, float]:
    x = np.asarray(signal_1d, dtype=np.float64)
    if len(x) == 0:
        return {}

    mag = np.abs(np.fft.rfft(x))
    power = mag * mag
    freqs = np.fft.rfftfreq(len(x), d=1.0 / float(sample_rate))
    total_power = float(np.sum(power) + 1e-12)
    prob = power / total_power
    spectral_entropy = float(-np.sum(prob * np.log(prob + 1e-12)))
    centroid = float(np.sum(freqs * power) / total_power)
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / total_power))
    rolloff_idx = int(np.searchsorted(np.cumsum(power), 0.85 * total_power, side="left"))
    rolloff_hz = float(freqs[min(rolloff_idx, len(freqs) - 1)])
    dominant_idx = int(np.argmax(mag))
    dominant_hz = float(freqs[dominant_idx])
    flatness = float(np.exp(np.mean(np.log(power + 1e-12))) / (np.mean(power + 1e-12)))

    nyquist = sample_rate / 2.0
    band_specs = {
        "band_low": (0.0, 0.1 * nyquist),
        "band_mid": (0.1 * nyquist, 0.3 * nyquist),
        "band_high": (0.3 * nyquist, nyquist + 1e-9),
    }

    feats = {
        f"{axis_name}__rfft_mag_mean": float(np.mean(mag)),
        f"{axis_name}__rfft_mag_std": float(np.std(mag)),
        f"{axis_name}__rfft_mag_max": float(np.max(mag)),
        f"{axis_name}__rfft_power_total": total_power,
        f"{axis_name}__rfft_spectral_entropy": spectral_entropy,
        f"{axis_name}__rfft_spectral_centroid_hz": centroid,
        f"{axis_name}__rfft_bandwidth_hz": bandwidth,
        f"{axis_name}__rfft_rolloff85_hz": rolloff_hz,
        f"{axis_name}__rfft_dominant_hz": dominant_hz,
        f"{axis_name}__rfft_flatness": flatness,
    }

    for band_name, (low_hz, high_hz) in band_specs.items():
        mask = (freqs >= low_hz) & (freqs < high_hz)
        band_power = float(np.sum(power[mask]))
        feats[f"{axis_name}__{band_name}_power"] = band_power
        feats[f"{axis_name}__{band_name}_ratio"] = float(band_power / total_power)
    return feats


def _extract_manual_feature_row(sensor_array: np.ndarray, sensor_cols: Sequence[str], sample_rate: int, prefix: str) -> Dict[str, float]:
    row = {}
    for axis_idx, axis_name in enumerate(sensor_cols):
        sig = sensor_array[:, axis_idx]
        time_feats = _manual_time_features(sig, axis_name)
        freq_feats = _manual_rfft_features(sig, axis_name, sample_rate)
        for key, value in {**time_feats, **freq_feats}.items():
            row[f"{prefix}__manual__{key}"] = value
    return row


def _tsfresh_fc_parameters(mode: str):
    compact = str(mode).strip().lower()
    if compact == "minimal":
        return MinimalFCParameters()
    if compact == "efficient":
        return EfficientFCParameters()
    return ComprehensiveFCParameters()


def _extract_tsfresh_batch(
    arrays: Sequence[np.ndarray],
    sensor_cols: Sequence[str],
    prefix: str,
    fc_mode: str,
    n_jobs: int,
) -> pd.DataFrame:
    if not arrays:
        return pd.DataFrame()

    long_frames = []
    ids = []
    for sample_pos, arr in enumerate(arrays):
        n_rows = int(arr.shape[0])
        if n_rows == 0:
            continue
        sample_id = sample_pos
        ids.append(sample_id)
        base = pd.DataFrame({"id": np.repeat(sample_id, n_rows), "time": np.arange(n_rows, dtype=np.int32)})
        axis_frames = []
        for axis_idx, axis_name in enumerate(sensor_cols):
            axis_df = base.copy()
            axis_df["kind"] = axis_name
            axis_df["value"] = arr[:, axis_idx].astype(np.float32, copy=False)
            axis_frames.append(axis_df)
        long_frames.append(pd.concat(axis_frames, ignore_index=True))

    if not long_frames:
        return pd.DataFrame(index=np.arange(len(arrays)))

    long_df = pd.concat(long_frames, ignore_index=True)
    feats = extract_features(
        long_df,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=_tsfresh_fc_parameters(fc_mode),
        disable_progressbar=True,
        n_jobs=int(n_jobs),
        impute_function=None,
    )
    impute(feats)
    feats = feats.sort_index()
    feats.index.name = "batch_pos"
    feats.columns = [f"{prefix}__tsfresh__{col}" for col in feats.columns]
    return feats


def _build_feature_rows_for_batch(
    payloads: Sequence[Dict],
    *,
    tsfresh_mode: str,
    tsfresh_n_jobs: int,
) -> pd.DataFrame:
    if not payloads:
        return pd.DataFrame()

    rows = []
    sensor_cols = payloads[0]["sensor_cols"]
    label_cols = payloads[0]["label_cols"]

    resolution_arrays: Dict[str, List[np.ndarray]] = {prefix: [] for prefix, _ in RESOLUTION_SPECS}
    for payload in payloads:
        base_row = {
            "sample_id": payload["sample_id"],
            "scenario": payload["scenario"],
            "experiment": payload["experiment"],
            "state_name": payload["state_name"],
            "start_idx": payload["start_idx"],
            "end_idx": payload["end_idx"],
            "real_points": payload["real_points"],
            "padded_points": payload["padded_points"],
            "window_points": payload["window_points"],
            "window_sec": payload["window_sec"],
            "step_points": payload["step_points"],
            "active_label_count": payload["active_label_count"],
        }
        for idx, label_name in enumerate(label_cols):
            base_row[f"label__{label_name}"] = int(payload["label_end"][idx])
            base_row[f"label_ratio__{label_name}"] = float(payload["label_ratio"][idx])
        rows.append(base_row)

        raw_sensor_array = payload["sensor_array"]
        for prefix, target_freq in RESOLUTION_SPECS:
            array_for_resolution = _downsample_direct_from_raw(raw_sensor_array, target_freq)
            resolution_arrays[prefix].append(array_for_resolution)

    base_df = pd.DataFrame(rows).set_index("sample_id")

    for prefix, target_freq in RESOLUTION_SPECS:
        manual_rows = []
        for array_for_resolution in resolution_arrays[prefix]:
            manual_rows.append(
                _extract_manual_feature_row(
                    sensor_array=array_for_resolution,
                    sensor_cols=sensor_cols,
                    sample_rate=target_freq,
                    prefix=prefix,
                )
            )
        manual_df = pd.DataFrame(manual_rows, index=base_df.index)

        tsfresh_df = _extract_tsfresh_batch(
            arrays=resolution_arrays[prefix],
            sensor_cols=sensor_cols,
            prefix=prefix,
            fc_mode=tsfresh_mode,
            n_jobs=tsfresh_n_jobs,
        )
        tsfresh_df.index = base_df.index
        base_df = base_df.join(manual_df, how="left")
        base_df = base_df.join(tsfresh_df, how="left")

    return base_df.reset_index()


def _save_table(df: pd.DataFrame, output_path: Path, save_format: str) -> None:
    save_format = str(save_format).strip().lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_format == "csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        return
    if save_format == "pickle":
        df.to_pickle(output_path)
        return
    if save_format == "parquet":
        df.to_parquet(output_path, index=False)
        return
    raise ValueError(f"Unsupported save_format: {save_format}")


def build_feature_table(
    *,
    data_path: str,
    output_path: Optional[str] = None,
    label_space: str = "superclass",
    default_window_points: int = 4000,
    default_step_points: int = 500,
    special_mode: bool = False,
    include_synthetic_axes: bool = True,
    allow_short_segment_padding: bool = True,
    include_tail_window: bool = True,
    save_format: str = "parquet",
    batch_size: int = 16,
    tsfresh_mode: str = "comprehensive",
    tsfresh_n_jobs: int = 0,
    max_samples: Optional[int] = None,
    sampling_rules: Optional[Dict[str, Dict[str, int]]] = None,
    return_dataframe: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Standalone builder that:
    1. loads the raw cpsHAR pickle,
    2. cuts timeline windows into samples,
    3. extracts multi-resolution handcrafted + RFFT + tsfresh features,
    4. returns or saves a wide table where each row is one sample.

    Notes:
    - default label space is the 6-class superclass setting.
    - special_mode uses the preset rules requested by the user.
    - 500 Hz and 100 Hz features are both downsampled directly from the raw 2000 Hz window.
    """
    try:
        raw_df = pd.read_pickle(Path(data_path))
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to read the pickle dataset. This usually means the current Python "
            "environment uses a different numpy/pandas stack than the one that created "
            "the pickle. Please switch to a compatible environment or re-save the raw "
            "dataset in a neutral format first."
        ) from exc

    feature_chunks: List[pd.DataFrame] = []
    payload_batch: List[Dict] = []

    for payload in _iter_window_payloads(
        raw_df=raw_df,
        label_space=label_space,
        default_window_points=default_window_points,
        default_step_points=default_step_points,
        special_mode=special_mode,
        include_synthetic_axes=include_synthetic_axes,
        allow_short_segment_padding=allow_short_segment_padding,
        include_tail_window=include_tail_window,
        sampling_rules=sampling_rules,
        max_samples=max_samples,
    ):
        payload_batch.append(payload)
        if len(payload_batch) >= int(batch_size):
            feature_chunks.append(
                _build_feature_rows_for_batch(
                    payload_batch,
                    tsfresh_mode=tsfresh_mode,
                    tsfresh_n_jobs=tsfresh_n_jobs,
                )
            )
            payload_batch = []

    if payload_batch:
        feature_chunks.append(
            _build_feature_rows_for_batch(
                payload_batch,
                tsfresh_mode=tsfresh_mode,
                tsfresh_n_jobs=tsfresh_n_jobs,
            )
        )

    feature_df = pd.concat(feature_chunks, ignore_index=True) if feature_chunks else pd.DataFrame()

    if output_path is not None:
        _save_table(feature_df, Path(output_path), save_format)
        meta = {
            "data_path": str(data_path),
            "label_space": str(label_space),
            "default_window_points": int(default_window_points),
            "default_step_points": int(default_step_points),
            "special_mode": bool(special_mode),
            "include_synthetic_axes": bool(include_synthetic_axes),
            "allow_short_segment_padding": bool(allow_short_segment_padding),
            "include_tail_window": bool(include_tail_window),
            "save_format": str(save_format),
            "batch_size": int(batch_size),
            "tsfresh_mode": str(tsfresh_mode),
            "tsfresh_n_jobs": int(tsfresh_n_jobs),
            "max_samples": None if max_samples is None else int(max_samples),
            "sampling_rules": sampling_rules or DEFAULT_SPECIAL_RULES,
            "num_samples": int(len(feature_df)),
            "num_columns": int(feature_df.shape[1]) if not feature_df.empty else 0,
        }
        meta_path = Path(output_path).with_suffix(Path(output_path).suffix + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    if return_dataframe:
        return feature_df
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Standalone sample cutter + multi-resolution feature-table builder for cpsHAR raw data."
    )
    parser.add_argument("--data_path", default="data/cps_data_multi_label.pkl")
    parser.add_argument("--output_path", default="output/standalone_feature_table.parquet")
    parser.add_argument("--label_space", choices=["superclass", "original"], default="superclass")
    parser.add_argument("--window_points", type=int, default=4000)
    parser.add_argument("--step_points", type=int, default=500)
    parser.add_argument("--special_mode", default="False")
    parser.add_argument("--include_synthetic_axes", default="True")
    parser.add_argument("--allow_short_segment_padding", default="True")
    parser.add_argument("--include_tail_window", default="True")
    parser.add_argument("--save_format", choices=["csv", "parquet", "pickle"], default="parquet")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tsfresh_mode", choices=["minimal", "efficient", "comprehensive"], default="comprehensive")
    parser.add_argument("--tsfresh_n_jobs", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    feature_df = build_feature_table(
        data_path=args.data_path,
        output_path=args.output_path,
        label_space=args.label_space,
        default_window_points=args.window_points,
        default_step_points=args.step_points,
        special_mode=_bool_string(args.special_mode),
        include_synthetic_axes=_bool_string(args.include_synthetic_axes),
        allow_short_segment_padding=_bool_string(args.allow_short_segment_padding),
        include_tail_window=_bool_string(args.include_tail_window),
        save_format=args.save_format,
        batch_size=args.batch_size,
        tsfresh_mode=args.tsfresh_mode,
        tsfresh_n_jobs=args.tsfresh_n_jobs,
        max_samples=args.max_samples,
        return_dataframe=True,
    )

    print(f"Saved feature table: {args.output_path}")
    print(f"Rows: {len(feature_df)}")
    print(f"Columns: {feature_df.shape[1] if not feature_df.empty else 0}")
    if not feature_df.empty:
        preview_cols = list(feature_df.columns[:20])
        print(f"First columns: {preview_cols}")


if __name__ == "__main__":
    main()
