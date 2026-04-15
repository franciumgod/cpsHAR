import numpy as np


BIN_LABELS_10PCT = [f"{i * 10:02d}-{(i + 1) * 10:02d}%" for i in range(10)]


def _as_2d(arr):
    out = np.asarray(arr)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _ratio_bin_counts_10pct(values):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    counts = np.zeros((10,), dtype=np.int64)
    if values.size == 0:
        return counts

    clipped = np.clip(values, 0.0, 1.0)
    bin_idx = np.minimum((clipped * 10.0).astype(np.int64), 9)
    binc = np.bincount(bin_idx, minlength=10)
    counts[: len(binc)] = binc[:10]
    return counts


def _bins_to_dict(counts):
    return {BIN_LABELS_10PCT[i]: int(counts[i]) for i in range(10)}


def _safe_mean(num, den):
    if den <= 0:
        return None
    return float(num) / float(den)


def _resolve_ratio_from_manifest(datahandler, sample_df, class_names):
    ratio_cols = [f"{name}__ratio_pre_ds" for name in class_names]
    if all(col in sample_df.columns for col in ratio_cols):
        return sample_df[ratio_cols].to_numpy(dtype=np.float32, copy=False)

    # Fallback for old manifests without pre-computed ratio columns.
    sample_df = sample_df.reset_index(drop=True)
    ratio_pre_ds = np.empty((len(sample_df), len(class_names)), dtype=np.float32)
    if len(sample_df) == 0:
        return ratio_pre_ds

    for row_idx, row in enumerate(sample_df.itertuples(index=False)):
        processed_df = datahandler._get_processed_source_df(int(row.source_index), class_names)
        label_slice = processed_df.iloc[int(row.start_idx):int(row.end_idx)][class_names].to_numpy(
            dtype=np.float32, copy=False
        )
        ratio_pre_ds[row_idx] = np.mean(label_slice, axis=0, dtype=np.float32)
    return ratio_pre_ds


def _collect_dataset_y_ratio(datahandler, class_names):
    if datahandler.dataset_kind == "window_samples":
        payload = datahandler.data
        y = _as_2d(payload["y"]).astype(np.int8, copy=False)
        ratio_pre_ds = payload.get("label_ratio_pre_ds")
        if ratio_pre_ds is None:
            ratio_pre_ds = y.astype(np.float32, copy=False)
        ratio_pre_ds = _as_2d(ratio_pre_ds).astype(np.float32, copy=False)
        return y, ratio_pre_ds

    if datahandler.dataset_kind == "sample_manifest":
        sample_df = datahandler.data
        y = sample_df[class_names].to_numpy(dtype=np.int8, copy=False)
        ratio_pre_ds = _resolve_ratio_from_manifest(datahandler, sample_df, class_names)
        return y, ratio_pre_ds

    raise ValueError("Raw dataset uses dedicated step=1 statistics path.")


def _build_label_ratio_stats_from_arrays(y, ratio_pre_ds, class_names):
    y = _as_2d(y).astype(np.int8, copy=False)
    ratio_pre_ds = _as_2d(ratio_pre_ds).astype(np.float32, copy=False)
    if y.shape != ratio_pre_ds.shape:
        ratio_pre_ds = y.astype(np.float32, copy=False)

    labels_stats = {}
    for label_idx, label_name in enumerate(class_names):
        pos_mask = y[:, label_idx] == 1
        pos_vals = ratio_pre_ds[pos_mask, label_idx]
        pos_count = int(pos_vals.shape[0])
        labels_stats[label_name] = {
            "positive_samples": pos_count,
            "mean_ratio": _safe_mean(float(np.sum(pos_vals, dtype=np.float64)), pos_count),
            "ratio_bins_10pct": _bins_to_dict(_ratio_bin_counts_10pct(pos_vals)),
        }

    return {
        "total_samples": int(y.shape[0]),
        "labels": labels_stats,
    }


def _build_raw_step1_dataset_stats(datahandler, class_names):
    seq_len = int(datahandler.config.prep.original_freq * datahandler.config.prep.seq_len_multiplier)
    n_labels = len(class_names)

    pos_counts = np.zeros((n_labels,), dtype=np.int64)
    pos_ratio_sums = np.zeros((n_labels,), dtype=np.float64)
    pos_bin_counts = np.zeros((n_labels, 10), dtype=np.int64)
    total_windows = 0

    for _, row in datahandler.data.iterrows():
        segment_df = datahandler._prepare_raw_df(row["data"])
        if len(segment_df) < seq_len:
            continue

        label_values = segment_df[class_names].to_numpy(dtype=np.int8, copy=False)
        n_rows = len(label_values)
        starts = np.arange(0, n_rows - seq_len + 1, dtype=np.int64)
        ends = starts + seq_len
        endpoints = label_values[ends - 1]

        prefix = np.zeros((n_rows + 1, n_labels), dtype=np.int64)
        prefix[1:] = np.cumsum(label_values, axis=0, dtype=np.int64)
        ratios = (prefix[ends] - prefix[starts]).astype(np.float32) / float(seq_len)
        total_windows += len(starts)

        for label_idx in range(n_labels):
            pos_mask = endpoints[:, label_idx] == 1
            if not np.any(pos_mask):
                continue
            vals = ratios[pos_mask, label_idx]
            c = int(vals.shape[0])
            pos_counts[label_idx] += c
            pos_ratio_sums[label_idx] += float(np.sum(vals, dtype=np.float64))
            pos_bin_counts[label_idx] += _ratio_bin_counts_10pct(vals)

    labels_stats = {}
    for label_idx, label_name in enumerate(class_names):
        pos_count = int(pos_counts[label_idx])
        labels_stats[label_name] = {
            "positive_samples": pos_count,
            "mean_ratio": _safe_mean(pos_ratio_sums[label_idx], pos_count),
            "ratio_bins_10pct": _bins_to_dict(pos_bin_counts[label_idx]),
        }

    return {
        "total_samples": int(total_windows),
        "labels": labels_stats,
    }


def compute_dataset_label_ratio_stats(datahandler, class_names):
    if datahandler.dataset_kind == "raw":
        base = _build_raw_step1_dataset_stats(datahandler, class_names)
        return {
            "dataset_kind": "raw",
            "ratio_basis": "step=1 windows from raw labels before downsampling",
            "window_size": int(datahandler.config.prep.original_freq * datahandler.config.prep.seq_len_multiplier),
            **base,
        }

    y, ratio_pre_ds = _collect_dataset_y_ratio(datahandler, class_names)
    base = _build_label_ratio_stats_from_arrays(y, ratio_pre_ds, class_names)
    window_size = None
    step_size = None
    if datahandler.dataset_kind == "sample_manifest":
        window_size = datahandler.sample_manifest_meta.get("window_size")
        step_size = datahandler.sample_manifest_meta.get("step_size")
    elif datahandler.dataset_kind == "window_samples":
        window_size = datahandler.data.get("window_size")
        step_size = datahandler.data.get("step_size")
    return {
        "dataset_kind": str(datahandler.dataset_kind),
        "ratio_basis": "stored pre-downsample window label ratio",
        "window_size": int(window_size) if window_size is not None else None,
        "step_size": int(step_size) if step_size is not None else None,
        **base,
    }


def build_fold_classifier_input_stats(y_train, ratio_pre_ds_train, class_names):
    y_train = _as_2d(y_train).astype(np.int8, copy=False)
    ratio_pre_ds_train = _as_2d(ratio_pre_ds_train).astype(np.float32, copy=False)
    if y_train.shape != ratio_pre_ds_train.shape:
        ratio_pre_ds_train = y_train.astype(np.float32, copy=False)

    total_samples = int(y_train.shape[0])
    per_classifier = {}

    for label_idx, label_name in enumerate(class_names):
        pos_mask = y_train[:, label_idx] == 1
        neg_mask = ~pos_mask
        pos_vals = ratio_pre_ds_train[pos_mask, label_idx]

        neg_other_counts = {}
        for other_idx, other_name in enumerate(class_names):
            if other_idx == label_idx:
                continue
            neg_other_counts[other_name] = int(np.sum(y_train[neg_mask, other_idx]))

        per_classifier[label_name] = {
            "total_samples": total_samples,
            "positive_samples": int(np.sum(pos_mask)),
            "negative_samples": int(np.sum(neg_mask)),
            "positive_target_ratio_bins_10pct": _bins_to_dict(_ratio_bin_counts_10pct(pos_vals)),
            "negative_other_label_counts": neg_other_counts,
        }

    return {
        "training_pool_samples": total_samples,
        "classifiers": per_classifier,
    }


def print_dataset_label_ratio_stats(stats):
    print("\n[SampleStats] Dataset-level positive label-ratio distribution")
    print(
        f"  dataset_kind={stats.get('dataset_kind')} | "
        f"total_samples={stats.get('total_samples')} | "
        f"basis={stats.get('ratio_basis')}"
    )
    labels = stats.get("labels", {})
    for label_name, info in labels.items():
        bins_text = ", ".join(
            f"{bin_name}:{info['ratio_bins_10pct'].get(bin_name, 0)}" for bin_name in BIN_LABELS_10PCT
        )
        print(
            f"  - {label_name:<24} positives={info.get('positive_samples', 0):<8d} "
            f"mean_ratio={info.get('mean_ratio')}"
        )
        print(f"    bins(10%): {bins_text}")


def print_fold_classifier_input_stats(fold_id, stats, class_names):
    print(f"\n[SampleStats] Fold {fold_id} classifier input composition")
    print(f"  training_pool_samples={stats.get('training_pool_samples', 0)}")

    classifiers = stats.get("classifiers", {})
    for label_name in class_names:
        info = classifiers.get(label_name, {})
        bins = info.get("positive_target_ratio_bins_10pct", {})
        bins_text = ", ".join(f"{bin_name}:{bins.get(bin_name, 0)}" for bin_name in BIN_LABELS_10PCT)
        neg_counts = info.get("negative_other_label_counts", {})
        neg_text = ", ".join(f"{k}:{v}" for k, v in neg_counts.items())

        print(
            f"  - {label_name:<24} total={info.get('total_samples', 0):<8d} "
            f"positive={info.get('positive_samples', 0):<8d} "
            f"negative={info.get('negative_samples', 0):<8d}"
        )
        print(f"    positive ratio bins(10%): {bins_text}")
        print(f"    negative other-label counts: {neg_text}")
