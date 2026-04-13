from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from utils.config import Config


def clean(df):
    cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
    return df.reset_index(drop=True)


def add_new_signal(df):
    df = df.copy()
    df["Acc.norm"] = np.sqrt(
        np.asarray(df["Acc.x"]) ** 2
        + np.asarray(df["Acc.y"]) ** 2
        + np.asarray(df["Acc.z"]) ** 2
    )
    df["Gyro.norm"] = np.sqrt(
        np.asarray(df["Gyro.x"]) ** 2
        + np.asarray(df["Gyro.y"]) ** 2
        + np.asarray(df["Gyro.z"]) ** 2
    )
    return df


def apply_superclass_mapping(df, mapping):
    df = df.copy()
    target_superclasses = list(set(mapping.values()))
    for super_name in target_superclasses:
        children = [child for child, parent in mapping.items() if parent == super_name]
        existing_children = [c for c in children if c in df.columns]
        if existing_children:
            df[super_name] = df[existing_children].max(axis=1)
        else:
            df[super_name] = 0

    cols_to_drop = [
        key for key in mapping.keys() if key in df.columns and key not in target_superclasses
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    return df


def build_manifest_for_step(raw_meta, config, window_size, step):
    final_target_cols = sorted(list(set(config.data.superclass_mapping.values())))
    manifest_parts = []

    for source_index, row in raw_meta.iterrows():
        sample_df = clean(row["data"])
        sample_df = add_new_signal(sample_df)
        sample_df = apply_superclass_mapping(sample_df, config.data.superclass_mapping)
        if len(sample_df) < window_size:
            continue

        starts = np.arange(0, len(sample_df) - window_size + 1, step, dtype=np.int32)
        ends = starts + np.int32(window_size)
        labels = sample_df.iloc[ends - 1][final_target_cols].to_numpy(dtype=np.int8, copy=False)

        part = pd.DataFrame(
            {
                "scenario": np.full(len(starts), row["scenario"], dtype=np.int16),
                "experiment": np.full(len(starts), row["experiment"], dtype=np.int16),
                "source_index": np.full(len(starts), source_index, dtype=np.int32),
                "start_idx": starts,
                "end_idx": ends,
            }
        )
        for label_idx, label_name in enumerate(final_target_cols):
            part[label_name] = labels[:, label_idx].astype(np.int8, copy=False)
        manifest_parts.append(part)

        print(
            f"step={step:<4d} | experiment={row['experiment']} | "
            f"source_index={source_index} | windows={len(starts)}"
        )

    if not manifest_parts:
        raise ValueError(f"No windows were generated for step={step}.")

    manifest = pd.concat(manifest_parts, ignore_index=True)
    payload = {
        "kind": "sample_manifest",
        "source_dataset_file": config.data.raw_dataset_file,
        "window_seconds": 2,
        "window_size": window_size,
        "sample_frequency": config.prep.original_freq,
        "step_size": step,
        "sensor_cols": list(config.data.sensor_cols),
        "label_cols": final_target_cols,
        "samples": manifest,
    }
    return payload


def _count_windows(length, window_size, step):
    if length < window_size:
        return 0
    return 1 + (length - window_size) // step


def build_materialized_windows_for_step(raw_meta, config, window_size, step):
    sensor_cols = list(config.data.sensor_cols)
    final_target_cols = sorted(list(set(config.data.superclass_mapping.values())))
    n_channels = len(sensor_cols)
    n_labels = len(final_target_cols)

    total_windows = 0
    source_cache = []
    for source_index, row in raw_meta.iterrows():
        sample_df = clean(row["data"])
        sample_df = add_new_signal(sample_df)
        sample_df = apply_superclass_mapping(sample_df, config.data.superclass_mapping)
        sample_len = len(sample_df)
        n_windows = _count_windows(sample_len, window_size, step)
        if n_windows <= 0:
            continue

        source_cache.append((source_index, row["scenario"], row["experiment"], sample_df, n_windows))
        total_windows += n_windows

    if total_windows == 0:
        raise ValueError(f"No windows were generated for step={step}.")

    X = np.empty((total_windows, window_size, n_channels), dtype=np.float32)
    y = np.empty((total_windows, n_labels), dtype=np.int8)
    experiment = np.empty(total_windows, dtype=np.int16)
    scenario = np.empty(total_windows, dtype=np.int16)
    source_indices = np.empty(total_windows, dtype=np.int32)
    start_indices = np.empty(total_windows, dtype=np.int32)

    write_pos = 0
    for source_index, scenario_id, experiment_id, sample_df, n_windows in source_cache:
        sensor_values = sample_df[sensor_cols].to_numpy(dtype=np.float32, copy=False)
        label_values = sample_df[final_target_cols].to_numpy(dtype=np.int8, copy=False)
        starts = np.arange(0, len(sample_df) - window_size + 1, step, dtype=np.int32)

        batch_size = 256
        for batch_start in range(0, len(starts), batch_size):
            batch_starts = starts[batch_start: batch_start + batch_size]
            batch_len = len(batch_starts)

            idx_matrix = batch_starts[:, None] + np.arange(window_size, dtype=np.int32)[None, :]
            X[write_pos: write_pos + batch_len] = sensor_values[idx_matrix]
            y[write_pos: write_pos + batch_len] = label_values[batch_starts + window_size - 1]
            experiment[write_pos: write_pos + batch_len] = experiment_id
            scenario[write_pos: write_pos + batch_len] = scenario_id
            source_indices[write_pos: write_pos + batch_len] = source_index
            start_indices[write_pos: write_pos + batch_len] = batch_starts
            write_pos += batch_len

        print(
            f"step={step:<4d} | experiment={experiment_id} | "
            f"source_index={source_index} | windows={n_windows}"
        )

    payload = {
        "kind": "window_samples",
        "source_dataset_file": config.data.raw_dataset_file,
        "window_seconds": 2,
        "window_size": window_size,
        "sample_frequency": config.prep.original_freq,
        "step_size": step,
        "sensor_cols": sensor_cols,
        "label_cols": final_target_cols,
        "X": X,
        "y": y,
        "experiment": experiment,
        "scenario": scenario,
        "source_index": source_indices,
        "start_idx": start_indices,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(
        description="Build 2s / 2000Hz window datasets from the raw CPS dataset."
    )
    parser.add_argument(
        "--input",
        default="cps_data_multi_label.pkl",
        help="Raw dataset filename under data/.",
    )
    parser.add_argument(
        "--steps",
        default="1,100,200,400,500",
        help="Comma-separated sample strides in raw timesteps.",
    )
    parser.add_argument(
        "--output_prefix",
        default="cps_windows_2s_2000hz",
        help="Prefix for generated manifest files under data/.",
    )
    args = parser.parse_args()

    config = Config()
    config.data.raw_dataset_file = args.input

    data_root = Path(r"D:\Code\research_intern\Pal2sim\cpsHAR\data")
    raw_path = data_root / Path(args.input).name
    raw_meta = pd.read_pickle(raw_path)

    window_size = int(2 * config.prep.original_freq)
    step_sizes = [int(part.strip()) for part in args.steps.split(",") if part.strip()]

    print(f"Loading raw dataset from {raw_path}")
    print(f"Window size: {window_size} samples (2s @ {config.prep.original_freq}Hz)")

    for step in step_sizes:
        if step == 1:
            payload = build_manifest_for_step(raw_meta, config, window_size=window_size, step=step)
            storage = "manifest"
        else:
            payload = build_materialized_windows_for_step(
                raw_meta,
                config,
                window_size=window_size,
                step=step,
            )
            storage = "materialized"

        output_name = f"{args.output_prefix}_step_{step}.pkl"
        output_path = data_root / output_name
        pd.to_pickle(payload, output_path)
        sample_count = len(payload["samples"]) if "samples" in payload else len(payload["X"])
        print(
            f"Saved dataset: {output_path} | "
            f"type={storage} | samples={sample_count}"
        )


if __name__ == "__main__":
    main()
