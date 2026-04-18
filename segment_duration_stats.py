import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import Config


@dataclass
class SegmentRecord:
    experiment: int
    state_name: str
    start_idx: int
    end_idx: int
    length_points: int
    duration_sec: float
    start_time: float
    end_time: float


def _clean(df):
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore").reset_index(drop=True)


def _apply_superclass_mapping(df, mapping):
    df = df.copy()
    target_superclasses = list(set(mapping.values()))
    for super_name in target_superclasses:
        children = [child for child, parent in mapping.items() if parent == super_name]
        existing_children = [c for c in children if c in df.columns]
        if existing_children:
            df[super_name] = df[existing_children].max(axis=1)
        else:
            df[super_name] = 0

    cols_to_drop = [k for k in mapping.keys() if k in df.columns and k not in target_superclasses]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df


def _label_matrix_to_state_names(label_matrix, label_names):
    state_names = []
    for row in label_matrix:
        active = [label_names[i] for i, v in enumerate(row) if int(v) == 1]
        state_names.append("+".join(active) if active else "NONE")
    return np.asarray(state_names, dtype=object)


def _find_segments(state_names, time_values, experiment_id):
    if len(state_names) == 0:
        return []

    change_idx = np.flatnonzero(state_names[1:] != state_names[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(state_names)]))

    segments = []
    for start, end in zip(starts, ends):
        length_points = int(end - start)
        start_time = float(time_values[start])
        end_time = float(time_values[end - 1])
        duration_sec = float(length_points / 2000.0)
        segments.append(
            SegmentRecord(
                experiment=int(experiment_id),
                state_name=str(state_names[start]),
                start_idx=int(start),
                end_idx=int(end - 1),
                length_points=length_points,
                duration_sec=duration_sec,
                start_time=start_time,
                end_time=end_time,
            )
        )
    return segments


def _build_exact_state_segments(df, label_names, experiment_id):
    label_matrix = df[label_names].to_numpy(dtype=np.int8, copy=False)
    state_names = _label_matrix_to_state_names(label_matrix, label_names)
    return _find_segments(state_names, df["time"].to_numpy(dtype=np.float64, copy=False), experiment_id)


def _build_per_label_segments(df, label_names, experiment_id):
    time_values = df["time"].to_numpy(dtype=np.float64, copy=False)
    out = {}
    for label_name in label_names:
        active = df[label_name].to_numpy(dtype=np.int8, copy=False).astype(bool)
        state_names = np.where(active, label_name, "INACTIVE")
        segments = _find_segments(state_names, time_values, experiment_id)
        out[label_name] = [seg for seg in segments if seg.state_name == label_name]
    return out


def _aggregate_segments(segments):
    if not segments:
        return {}

    records = {}
    by_state = {}
    for seg in segments:
        by_state.setdefault(seg.state_name, []).append(seg)

    for state_name, state_segments in by_state.items():
        points = np.asarray([seg.length_points for seg in state_segments], dtype=np.int64)
        seconds = np.asarray([seg.duration_sec for seg in state_segments], dtype=np.float64)
        records[state_name] = {
            "segments": int(len(state_segments)),
            "min_points": int(np.min(points)),
            "max_points": int(np.max(points)),
            "mean_points": float(np.mean(points)),
            "p05_points": float(np.percentile(points, 5)),
            "p25_points": float(np.percentile(points, 25)),
            "p50_points": float(np.percentile(points, 50)),
            "p75_points": float(np.percentile(points, 75)),
            "p95_points": float(np.percentile(points, 95)),
            "min_sec": float(np.min(seconds)),
            "max_sec": float(np.max(seconds)),
            "mean_sec": float(np.mean(seconds)),
            "p05_sec": float(np.percentile(seconds, 5)),
            "p25_sec": float(np.percentile(seconds, 25)),
            "p50_sec": float(np.percentile(seconds, 50)),
            "p75_sec": float(np.percentile(seconds, 75)),
            "p95_sec": float(np.percentile(seconds, 95)),
        }
    return records


def _records_to_frame(records):
    rows = []
    for state_name, stats in records.items():
        row = {"state_name": state_name}
        row.update(stats)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["segments", "p50_points"], ascending=[False, False]).reset_index(drop=True)


def _segment_list_to_frame(segments):
    if not segments:
        return pd.DataFrame()
    return pd.DataFrame([asdict(seg) for seg in segments])


def _propose_window_step(records, default_window_points=4000):
    proposals = []
    for state_name, stats in records.items():
        if state_name == "NONE":
            continue

        p05 = max(1, int(round(stats["p05_points"])))
        p25 = max(1, int(round(stats["p25_points"])))
        p50 = max(1, int(round(stats["p50_points"])))

        window_points = int(min(default_window_points, max(200, p25)))
        if p25 >= default_window_points:
            window_points = int(default_window_points)
        elif p50 >= default_window_points:
            window_points = int(min(default_window_points, max(window_points, p25)))

        step_points = int(max(25, min(500, round(window_points / 8))))
        if stats["segments"] < 200 or p25 < 2000:
            step_points = int(max(25, min(step_points, round(max(100, window_points / 10)))))

        proposals.append(
            {
                "state_name": state_name,
                "suggest_window_points": int(window_points),
                "suggest_window_sec": float(window_points / 2000.0),
                "suggest_step_points": int(step_points),
                "suggest_step_sec": float(step_points / 2000.0),
                "basis_p05_points": int(p05),
                "basis_p25_points": int(p25),
                "basis_p50_points": int(p50),
                "segments": int(stats["segments"]),
            }
        )
    if not proposals:
        return pd.DataFrame()
    return pd.DataFrame(proposals).sort_values(["segments", "state_name"], ascending=[False, True]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compute contiguous duration statistics for cpsHAR label states on the raw timeline."
    )
    parser.add_argument("--data", default="data/cps_data_multi_label.pkl")
    parser.add_argument("--label_space", choices=["superclass", "original"], default="superclass")
    parser.add_argument("--output_dir", default="output/segment_stats")
    args = parser.parse_args()

    cfg = Config()
    raw_df = pd.read_pickle(Path(args.data))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.label_space == "superclass":
        label_names = sorted(set(cfg.data.superclass_mapping.values()))
    else:
        label_names = list(cfg.data.label_cols)

    all_exact_segments = []
    per_label_segments = {label_name: [] for label_name in label_names}

    for row in raw_df.itertuples(index=False):
        segment_df = _clean(row.data)
        if args.label_space == "superclass":
            segment_df = _apply_superclass_mapping(segment_df, cfg.data.superclass_mapping)

        keep_cols = ["time"] + [c for c in label_names if c in segment_df.columns]
        segment_df = segment_df[keep_cols].copy()
        for label_name in label_names:
            if label_name not in segment_df.columns:
                segment_df[label_name] = 0
        segment_df = segment_df[["time"] + label_names]

        exact_segments = _build_exact_state_segments(segment_df, label_names, row.experiment)
        all_exact_segments.extend(exact_segments)

        label_seg_map = _build_per_label_segments(segment_df, label_names, row.experiment)
        for label_name, segs in label_seg_map.items():
            per_label_segments[label_name].extend(segs)

    exact_state_stats = _aggregate_segments(all_exact_segments)
    per_label_stats = {label_name: _aggregate_segments(segs).get(label_name) for label_name, segs in per_label_segments.items()}
    per_label_stats = {k: v for k, v in per_label_stats.items() if v is not None}

    exact_state_df = _records_to_frame(exact_state_stats)
    per_label_df = _records_to_frame(per_label_stats)
    proposal_df = _propose_window_step(per_label_stats)
    exact_segment_df = _segment_list_to_frame(all_exact_segments)

    exact_state_df.to_csv(output_dir / f"{args.label_space}_exact_state_summary.csv", index=False, encoding="utf-8-sig")
    per_label_df.to_csv(output_dir / f"{args.label_space}_per_label_summary.csv", index=False, encoding="utf-8-sig")
    proposal_df.to_csv(output_dir / f"{args.label_space}_window_step_proposal.csv", index=False, encoding="utf-8-sig")
    exact_segment_df.to_csv(output_dir / f"{args.label_space}_exact_segments.csv", index=False, encoding="utf-8-sig")

    summary = {
        "label_space": args.label_space,
        "data": str(args.data),
        "per_label_stats": per_label_stats,
        "exact_state_stats": exact_state_stats,
        "proposal": proposal_df.to_dict(orient="records"),
    }
    with open(output_dir / f"{args.label_space}_segment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_dir}")
    print(f"Per-label summary rows: {len(per_label_df)}")
    print(f"Exact-state summary rows: {len(exact_state_df)}")
    if not per_label_df.empty:
        print("\nPer-label duration summary:")
        print(per_label_df.to_string(index=False))
    if not exact_state_df.empty:
        print("\nTop exact-state duration summary:")
        print(exact_state_df.head(20).to_string(index=False))
    if not proposal_df.empty:
        print("\nWindow/step proposal:")
        print(proposal_df.to_string(index=False))


if __name__ == "__main__":
    main()
