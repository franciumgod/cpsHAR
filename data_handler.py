from pathlib import Path

import numpy as np
import pandas as pd
# from numpy.lib._stride_tricks_impl import sliding_window_view
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


class DataHandler:
    def __init__(
        self,
        config,
        downsample_method="sliding_window",
        subsample_method="interval",
        n_train_data_samples=100_000,
        preprocess_order="downsample_first",
        single_label_only=False,
        downsample_window_size=40,
        downsample_window_step=20,
    ):
        self.config = config
        self.include_synthetic_axes = bool(getattr(self.config.data, "use_synthetic_axes", True))
        self.config.data.set_sensor_cols(include_synthetic_axes=self.include_synthetic_axes)
        self.scaler = None
        self.downsample_method = (
            str(downsample_method).lower() if downsample_method is not None else "false"
        )
        self.downsample_window_size = downsample_window_size
        self.downsample_window_step = downsample_window_step
        self.subsample_method = (
            str(subsample_method).lower() if subsample_method is not None else "false"
        )
        self.n_train_data_samples = int(n_train_data_samples)
        self.preprocess_order = str(preprocess_order).lower()
        self.single_label_only = str(single_label_only).strip().lower() in {"1", "true", "yes", "y", "on"}

        self.data_root = Path(r"D:\Code\research_intern\Pal2sim\cpsHAR\data")
        self.local_path = self.data_root / Path(self.config.data.dataset_file).name
        self.data = None
        self.dataset_kind = None
        self.sample_manifest_meta = {}
        self.raw_source_data = None
        self._source_segment_cache = {}
        self.last_split_ratio_pre_ds = {}

        self._load_data_set()

    def _get_merged_data(self, meta_subset):
        merged = [row["data"] for _, row in meta_subset.iterrows()]
        return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

    def get_last_split_ratio_pre_ds(self):
        return dict(self.last_split_ratio_pre_ds)

    def _clean(self, df):
        cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        return df.reset_index(drop=True)

    def _add_new_signal(self, df):
        df = df.copy()
        if not self.include_synthetic_axes:
            drop_cols = [
                col for col in getattr(self.config.data, "synthetic_sensor_cols", [])
                if col in df.columns
            ]
            if drop_cols:
                df = df.drop(columns=drop_cols, errors="ignore")
            return df

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

    def _get_challenge_data_numpy(self, df, seq_len, sensor_cols, label_cols):
        data_values = df[sensor_cols].to_numpy(dtype=np.float32)
        label_values = df[label_cols].to_numpy(dtype=np.int8)

        X_view = sliding_window_view(data_values, window_shape=seq_len, axis=0)
        y_start_index = seq_len - 1
        y_view = label_values[y_start_index: y_start_index + len(X_view)]

        min_len = min(len(X_view), len(y_view))
        X_view = X_view[:min_len]
        y_view = y_view[:min_len]
        if min_len == 0:
            ratio_view = np.empty((0, len(label_cols)), dtype=np.float32)
            return X_view, y_view, ratio_view

        starts = np.arange(0, min_len, dtype=np.int64)
        prefix = np.zeros((len(label_values) + 1, len(label_cols)), dtype=np.int64)
        if len(label_values) > 0:
            prefix[1:] = np.cumsum(label_values, axis=0, dtype=np.int64)
        sums = prefix[starts + seq_len] - prefix[starts]
        ratio_view = (sums.astype(np.float32) / float(seq_len)).astype(np.float32, copy=False)
        return X_view, y_view, ratio_view

    def _detect_dataset_kind(self, payload):
        if isinstance(payload, dict) and payload.get("kind") == "sample_manifest":
            return "sample_manifest"
        if isinstance(payload, dict) and payload.get("kind") == "window_samples":
            return "window_samples"

        if isinstance(payload, pd.DataFrame):
            cols = set(payload.columns)
            if {"scenario", "experiment", "data"}.issubset(cols):
                return "raw"
            if {"scenario", "experiment", "source_index", "start_idx", "end_idx"}.issubset(cols):
                return "sample_manifest"

        return "raw"

    def _load_data_set(self):
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.local_path.exists():
            print(f"Dataset not found at {self.local_path}")
        else:
            print(f"Local dataset found: {self.local_path}")

        print("Loading data into memory...")
        payload = pd.read_pickle(self.local_path)
        self.dataset_kind = self._detect_dataset_kind(payload)

        if self.dataset_kind == "sample_manifest":
            if isinstance(payload, dict):
                self.sample_manifest_meta = {k: v for k, v in payload.items() if k != "samples"}
                self.data = payload["samples"].copy()
                if "sensor_cols" in payload:
                    self.config.data.set_sensor_cols(
                        include_synthetic_axes=self.include_synthetic_axes,
                        available_sensor_cols=list(payload["sensor_cols"]),
                    )
                source_name = self.sample_manifest_meta.get(
                    "source_dataset_file",
                    self.config.data.raw_dataset_file,
                )
            else:
                self.data = payload.copy()
                source_name = self.config.data.raw_dataset_file

            source_path = self.data_root / Path(source_name).name
            print(f"Loading raw source dataset for sample manifests: {source_path}")
            self.raw_source_data = pd.read_pickle(source_path)
            print("Sample manifest and raw source loaded.")
        elif self.dataset_kind == "window_samples":
            self.data = payload
            if "sensor_cols" in payload:
                self.config.data.set_sensor_cols(
                    include_synthetic_axes=self.include_synthetic_axes,
                    available_sensor_cols=list(payload["sensor_cols"]),
                )
            print("Materialized window samples loaded.")
        else:
            self.data = payload
            print("Raw dataset loaded.")

    def _apply_superclass_mapping(self, df, mapping):
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
            k for k in mapping.keys() if k in df.columns and k not in target_superclasses
        ]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        return df

    def _get_final_target_cols(self):
        unique_superclasses = set(self.config.data.superclass_mapping.values())
        return sorted(list(unique_superclasses))

    def _downsample_interval(self, df, sensor_cols, label_cols):
        factor = max(1, int(self.config.prep.ds_factor))
        keep_cols = [c for c in (sensor_cols + label_cols) if c in df.columns]
        return df.iloc[::factor][keep_cols].reset_index(drop=True)

    def _downsample_sliding_window(self, df, sensor_cols, label_cols):
        win = self.downsample_window_size
        step = self.downsample_window_step

        sensor_cols = [c for c in sensor_cols if c in df.columns]
        label_cols = [c for c in label_cols if c in df.columns]

        sensor_values = df[sensor_cols].to_numpy(dtype=np.float32)
        label_values = df[label_cols].to_numpy(dtype=np.float32)
        n_rows = len(df)
        starts = np.arange(0, n_rows - win + 1, step, dtype=int)

        sensor_ds = np.empty((len(starts), len(sensor_cols)), dtype=np.float32)
        label_ds = (
            np.empty((len(starts), len(label_cols)), dtype=np.float32) if label_cols else None
        )

        for i, start_idx in enumerate(starts):
            end_idx = start_idx + win
            sensor_ds[i] = sensor_values[start_idx:end_idx].mean(axis=0, dtype=np.float32)
            if label_cols:
                label_ds[i] = label_values[end_idx - 1]

        out_df = pd.DataFrame(sensor_ds, columns=sensor_cols)
        if label_cols:
            for col_idx, col_name in enumerate(label_cols):
                out_df[col_name] = label_ds[:, col_idx]

        return out_df.reset_index(drop=True)

    def _maybe_downsample(self, df, sensor_cols, label_cols, split_name):
        method = self.downsample_method
        if method in {"false", "none", "0"}:
            ds_df = df.reset_index(drop=True)
        elif method == "interval":
            ds_df = self._downsample_interval(df, sensor_cols, label_cols)
        elif method == "sliding_window":
            ds_df = self._downsample_sliding_window(df, sensor_cols, label_cols)
        else:
            ds_df = df.reset_index(drop=True)

        print(f"Downsample [{split_name}] with method={method}: {len(df)} -> {len(ds_df)} rows")
        return ds_df

    def _maybe_downsample_window_array(self, X, split_name=None):
        method = self.downsample_method
        if method in {"false", "none", "0"}:
            X_ds = X
        elif method == "interval":
            factor = max(1, int(self.config.prep.ds_factor))
            X_ds = X[:, ::factor, :]
        elif method == "sliding_window":
            win = self.downsample_window_size
            step = self.downsample_window_step
            if X.shape[1] < win:
                return X.astype(np.float32, copy=False)

            windows = sliding_window_view(X, window_shape=win, axis=1)
            windows = windows[:, ::step, :, :]
            X_ds = windows.mean(axis=-1, dtype=np.float32)
        else:
            X_ds = X

        if split_name is not None:
            print(
                f"Downsample [{split_name}] sample windows with method={method}: "
                f"{X.shape[1]} -> {X_ds.shape[1]} steps"
            )
        return X_ds.astype(np.float32, copy=False)

    def _select_subsample_indices(self, total_windows, apply_subsample=True):
        if total_windows <= 0:
            return np.empty((0,), dtype=np.int64)

        if not apply_subsample:
            return np.arange(total_windows, dtype=np.int64)

        method = self.subsample_method
        if method in {"false", "none", "0"}:
            return np.arange(total_windows, dtype=np.int64)

        n_samples = min(max(self.n_train_data_samples, 0), total_windows)
        if n_samples >= total_windows:
            return np.arange(total_windows, dtype=np.int64)

        if method == "interval":
            return np.linspace(0, total_windows - 1, n_samples, dtype=np.int64)

        rng = np.random.RandomState(42)
        if method == "random":
            return rng.choice(total_windows, n_samples, replace=False).astype(np.int64)

        return np.arange(total_windows, dtype=np.int64)

    def _maybe_subsample_window_samples(
        self,
        X,
        y,
        split_name,
        apply_subsample,
        ratio_pre_ds=None,
    ):
        if not apply_subsample:
            return X, y, ratio_pre_ds

        idx = self._select_subsample_indices(len(X), apply_subsample=True)
        if len(idx) == len(X):
            print(f"Subsample [{split_name}] windows: total={len(X)} -> kept={len(X)}")
            return X, y, ratio_pre_ds

        X_sub = X[idx]
        y_sub = y[idx]
        ratio_sub = ratio_pre_ds[idx] if ratio_pre_ds is not None else None
        print(f"Subsample [{split_name}] windows: total={len(X)} -> kept={len(X_sub)}")
        return X_sub, y_sub, ratio_sub

    def _build_train_filter_mask(self, y, label_cols, split_name):
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        n_samples = len(y_arr)
        if n_samples == 0:
            return np.empty((0,), dtype=bool)

        keep_mask = np.ones(n_samples, dtype=bool)
        if self.single_label_only:
            active_counts = np.sum(y_arr, axis=1)
            single_mask = active_counts <= 1
            before = int(np.sum(keep_mask))
            keep_mask &= single_mask
            after = int(np.sum(keep_mask))
            print(
                f"Post-filter [{split_name}] single_label_only: "
                f"{before} -> {after} samples"
            )

        return keep_mask

    def _apply_post_subsample_filters(
        self,
        X,
        y,
        label_cols,
        split_name,
        apply_filters=True,
        ratio_pre_ds=None,
    ):
        before = len(X)
        if not apply_filters:
            return X, y, ratio_pre_ds

        keep_mask = self._build_train_filter_mask(y, label_cols=label_cols, split_name=split_name)
        X = X[keep_mask]
        y = y[keep_mask]
        ratio = ratio_pre_ds[keep_mask] if ratio_pre_ds is not None else None
        after = len(X)
        print(f"Post-subsample filter [{split_name}]: {before} -> {after} samples")
        return X, y, ratio

    def _apply_post_subsample_start_filters(
        self,
        starts,
        y_selected,
        label_cols,
        split_name,
        apply_filters=True,
    ):
        before = len(starts)
        if not apply_filters:
            return starts

        keep_mask = self._build_train_filter_mask(y_selected, label_cols=label_cols, split_name=split_name)
        filtered = starts[keep_mask]
        after = len(filtered)
        print(f"Post-subsample filter [{split_name}] starts: {before} -> {after}")
        return filtered

    def _window_len_after_downsample(self, seq_len):
        method = self.downsample_method
        if method in {"false", "none", "0"}:
            return seq_len
        if method == "interval":
            factor = max(1, int(self.config.prep.ds_factor))
            return 1 + (seq_len - 1) // factor
        if method == "sliding_window":
            if seq_len < self.downsample_window_size:
                return seq_len
            return 1 + (seq_len - self.downsample_window_size) // self.downsample_window_step
        return seq_len

    def _extract_subsampled_windows_batched(
        self,
        df,
        seq_len,
        sensor_cols,
        label_cols,
        split_name,
        apply_subsample=True,
        apply_downsample=True,
        apply_post_subsample_filters=False,
    ):
        sensor_values = df[sensor_cols].to_numpy(dtype=np.float32, copy=False)
        label_values = df[label_cols].to_numpy(dtype=np.int8, copy=False)
        n_labels = len(label_cols)
        total_windows = max(0, len(df) - seq_len + 1)
        starts = self._select_subsample_indices(total_windows, apply_subsample=apply_subsample)
        if apply_post_subsample_filters and len(starts) > 0:
            selected_y = label_values[starts + seq_len - 1]
            starts = self._apply_post_subsample_start_filters(
                starts,
                selected_y,
                label_cols=label_cols,
                split_name=split_name,
                apply_filters=True,
            )

        n_samples = len(starts)
        out_len = self._window_len_after_downsample(seq_len) if apply_downsample else seq_len
        X = np.empty((n_samples, out_len, len(sensor_cols)), dtype=np.float32)
        y = np.empty((n_samples, n_labels), dtype=np.int8)
        ratio_pre_ds = np.empty((n_samples, n_labels), dtype=np.float32)
        if n_samples == 0:
            return X, y, ratio_pre_ds

        base_steps = np.arange(seq_len, dtype=np.int64)
        prefix = np.zeros((len(label_values) + 1, n_labels), dtype=np.int64)
        if len(label_values) > 0:
            prefix[1:] = np.cumsum(label_values, axis=0, dtype=np.int64)
        batch_size = 256
        write_pos = 0

        for start_idx in range(0, n_samples, batch_size):
            batch_starts = starts[start_idx: start_idx + batch_size]
            idx_matrix = batch_starts[:, None] + base_steps[None, :]
            batch_windows = sensor_values[idx_matrix]
            batch_labels = label_values[batch_starts + seq_len - 1]
            if apply_downsample:
                batch_windows = self._maybe_downsample_window_array(batch_windows, split_name=None)
            else:
                batch_windows = batch_windows.astype(np.float32, copy=False)

            batch_len = len(batch_starts)
            X[write_pos: write_pos + batch_len] = batch_windows
            y[write_pos: write_pos + batch_len] = batch_labels
            sums = prefix[batch_starts + seq_len] - prefix[batch_starts]
            ratio_pre_ds[write_pos: write_pos + batch_len] = (
                sums.astype(np.float32) / float(seq_len)
            ).astype(np.float32, copy=False)
            write_pos += batch_len

        action = "selected" if apply_subsample else "all"
        print(f"Subsample-first [{split_name}] windows: total={total_windows} -> {action}={n_samples}")
        return X, y, ratio_pre_ds

    def _fit_and_apply_scaler(self, train_x, val_x, test_x):
        n_channels = train_x.shape[-1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(train_x.reshape(-1, n_channels))

        def transform(split_x):
            original_shape = split_x.shape
            scaled = self.scaler.transform(split_x.reshape(-1, n_channels))
            return scaled.reshape(original_shape).astype(np.float32, copy=False)

        return transform(train_x), transform(val_x), transform(test_x)

    def _prepare_raw_df(self, df):
        df = self._clean(df)
        df = self._add_new_signal(df)
        df = self._apply_superclass_mapping(df, self.config.data.superclass_mapping)
        return df

    def _get_effective_seq_len_for_raw(self):
        if self.downsample_method in {"false", "none", "0"}:
            return int(self.config.prep.original_freq * self.config.prep.seq_len_multiplier)
        return int(self.config.prep.seq_len)

    def _get_processed_source_df(self, source_index, final_target_cols):
        if source_index not in self._source_segment_cache:
            source_df = self.raw_source_data.loc[source_index, "data"]
            processed_df = self._prepare_raw_df(source_df)
            keep_cols = self.config.data.sensor_cols + final_target_cols
            self._source_segment_cache[source_index] = processed_df[keep_cols].reset_index(drop=True)
        return self._source_segment_cache[source_index]

    def _materialize_window_samples(self, sample_df, final_target_cols):
        sample_df = sample_df.reset_index(drop=True)
        if sample_df.empty:
            empty_x = np.empty((0, 0, len(self.config.data.sensor_cols)), dtype=np.float32)
            empty_y = np.empty((0, len(final_target_cols)), dtype=np.int8)
            empty_ratio = np.empty((0, len(final_target_cols)), dtype=np.float32)
            return empty_x, empty_y, empty_ratio

        window_size = int(sample_df.iloc[0]["end_idx"] - sample_df.iloc[0]["start_idx"])
        X = np.empty((len(sample_df), window_size, len(self.config.data.sensor_cols)), dtype=np.float32)
        ratio_pre_ds = np.empty((len(sample_df), len(final_target_cols)), dtype=np.float32)
        ratio_cols = [f"{name}__ratio_pre_ds" for name in final_target_cols]
        has_ratio_cols = all(col in sample_df.columns for col in ratio_cols)
        if has_ratio_cols:
            ratio_pre_ds = sample_df[ratio_cols].to_numpy(dtype=np.float32, copy=False)

        for row_idx, row in enumerate(sample_df.itertuples(index=False)):
            processed_df = self._get_processed_source_df(int(row.source_index), final_target_cols)
            sample_values = processed_df.iloc[int(row.start_idx):int(row.end_idx)][
                self.config.data.sensor_cols
            ].to_numpy(dtype=np.float32, copy=False)
            X[row_idx] = sample_values
            if not has_ratio_cols:
                label_slice = processed_df.iloc[int(row.start_idx):int(row.end_idx)][
                    final_target_cols
                ].to_numpy(dtype=np.float32, copy=False)
                ratio_pre_ds[row_idx] = np.mean(label_slice, axis=0, dtype=np.float32)

        y = sample_df[final_target_cols].to_numpy(dtype=np.int8, copy=False)
        return X, y, ratio_pre_ds

    def _get_data_loaders_from_raw(self):
        test_mask = self.data["experiment"] == self.config.data.test_experiment_id
        validation_mask = self.data["experiment"] == self.config.data.validation_experiment_id

        train_metadata = self.data[~test_mask & ~validation_mask]
        test_metadata = self.data[test_mask]
        validation_metadata = self.data[validation_mask]

        train_df = self._get_merged_data(train_metadata)
        test_df = self._get_merged_data(test_metadata)
        validation_df = self._get_merged_data(validation_metadata)

        train_df = self._prepare_raw_df(train_df)
        test_df = self._prepare_raw_df(test_df)
        validation_df = self._prepare_raw_df(validation_df)

        final_target_cols = self._get_final_target_cols()
        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)

        if self.preprocess_order == "subsample_first":
            train_seq_len = int(self.config.prep.original_freq * self.config.prep.seq_len_multiplier)
            train_x, train_y, train_ratio = self._extract_subsampled_windows_batched(
                train_df,
                train_seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
                "train",
                apply_subsample=True,
                apply_downsample=True,
                apply_post_subsample_filters=True,
            )
            val_x, val_y, val_ratio = self._extract_subsampled_windows_batched(
                validation_df,
                train_seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
                "val",
                apply_subsample=False,
            )
            test_x, test_y, test_ratio = self._extract_subsampled_windows_batched(
                test_df,
                train_seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
                "test",
                apply_subsample=False,
            )
        else:
            train_df = self._maybe_downsample(train_df, self.config.data.sensor_cols, final_target_cols, "train")
            validation_df = self._maybe_downsample(
                validation_df,
                self.config.data.sensor_cols,
                final_target_cols,
                "val",
            )
            test_df = self._maybe_downsample(test_df, self.config.data.sensor_cols, final_target_cols, "test")

            seq_len = self._get_effective_seq_len_for_raw()
            train_x, train_y, train_ratio = self._get_challenge_data_numpy(
                train_df,
                seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
            )
            val_x, val_y, val_ratio = self._get_challenge_data_numpy(
                validation_df,
                seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
            )
            test_x, test_y, test_ratio = self._get_challenge_data_numpy(
                test_df,
                seq_len,
                self.config.data.sensor_cols,
                final_target_cols,
            )

        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        self.last_split_ratio_pre_ds = {
            "train": train_ratio.astype(np.float32, copy=False),
            "val": val_ratio.astype(np.float32, copy=False),
            "test": test_ratio.astype(np.float32, copy=False),
        }
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def _get_data_loaders_from_sample_manifest(self):
        test_mask = self.data["experiment"] == self.config.data.test_experiment_id
        validation_mask = self.data["experiment"] == self.config.data.validation_experiment_id

        train_manifest = self.data[~test_mask & ~validation_mask]
        test_manifest = self.data[test_mask]
        validation_manifest = self.data[validation_mask]

        final_target_cols = self._get_final_target_cols()
        train_x, train_y, train_ratio = self._materialize_window_samples(train_manifest, final_target_cols)
        val_x, val_y, val_ratio = self._materialize_window_samples(validation_manifest, final_target_cols)
        test_x, test_y, test_ratio = self._materialize_window_samples(test_manifest, final_target_cols)

        train_x, train_y, train_ratio = self._maybe_subsample_window_samples(
            train_x,
            train_y,
            split_name="train",
            apply_subsample=True,
            ratio_pre_ds=train_ratio,
        )
        train_x, train_y, train_ratio = self._apply_post_subsample_filters(
            train_x,
            train_y,
            label_cols=final_target_cols,
            split_name="train",
            apply_filters=True,
            ratio_pre_ds=train_ratio,
        )
        train_x = self._maybe_downsample_window_array(train_x, "train")
        val_x = self._maybe_downsample_window_array(val_x, "val")
        test_x = self._maybe_downsample_window_array(test_x, "test")

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        self.last_split_ratio_pre_ds = {
            "train": train_ratio.astype(np.float32, copy=False),
            "val": val_ratio.astype(np.float32, copy=False),
            "test": test_ratio.astype(np.float32, copy=False),
        }
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def _get_data_loaders_from_window_samples(self):
        payload = self.data
        X = payload["X"]
        y = payload["y"]
        experiment = payload["experiment"]
        payload_sensor_cols = list(payload.get("sensor_cols", self.config.data.sensor_cols))
        active_sensor_cols = self.config.data.set_sensor_cols(
            include_synthetic_axes=self.include_synthetic_axes,
            available_sensor_cols=payload_sensor_cols,
        )
        if payload_sensor_cols and len(payload_sensor_cols) == X.shape[-1]:
            sensor_idx = {name: idx for idx, name in enumerate(payload_sensor_cols)}
            keep_idx = [sensor_idx[name] for name in active_sensor_cols if name in sensor_idx]
            if keep_idx and len(keep_idx) != X.shape[-1]:
                X = X[:, :, keep_idx]
        ratio_pre_ds_all = payload.get("label_ratio_pre_ds")
        if ratio_pre_ds_all is None:
            ratio_pre_ds_all = y.astype(np.float32, copy=False)
        ratio_pre_ds_all = np.asarray(ratio_pre_ds_all, dtype=np.float32)
        final_target_cols = list(payload["label_cols"])

        test_mask = experiment == self.config.data.test_experiment_id
        validation_mask = experiment == self.config.data.validation_experiment_id
        train_mask = (~test_mask) & (~validation_mask)

        train_x = X[train_mask]
        train_y = y[train_mask]
        train_ratio = ratio_pre_ds_all[train_mask]
        val_x = X[validation_mask]
        val_y = y[validation_mask]
        val_ratio = ratio_pre_ds_all[validation_mask]
        test_x = X[test_mask]
        test_y = y[test_mask]
        test_ratio = ratio_pre_ds_all[test_mask]

        train_x, train_y, train_ratio = self._maybe_subsample_window_samples(
            train_x,
            train_y,
            split_name="train",
            apply_subsample=True,
            ratio_pre_ds=train_ratio,
        )
        train_x, train_y, train_ratio = self._apply_post_subsample_filters(
            train_x,
            train_y,
            label_cols=final_target_cols,
            split_name="train",
            apply_filters=True,
            ratio_pre_ds=train_ratio,
        )
        train_x = self._maybe_downsample_window_array(train_x, "train")
        val_x = self._maybe_downsample_window_array(val_x, "val")
        test_x = self._maybe_downsample_window_array(test_x, "test")

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        self.last_split_ratio_pre_ds = {
            "train": train_ratio.astype(np.float32, copy=False),
            "val": val_ratio.astype(np.float32, copy=False),
            "test": test_ratio.astype(np.float32, copy=False),
        }
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def get_data_loaders(self):
        print("Starting data preparation...", flush=True)
        self._source_segment_cache = {}
        self.last_split_ratio_pre_ds = {}

        if self.dataset_kind == "raw":
            return self._get_data_loaders_from_raw()
        if self.dataset_kind == "sample_manifest":
            return self._get_data_loaders_from_sample_manifest()
        if self.dataset_kind == "window_samples":
            return self._get_data_loaders_from_window_samples()
        return self._get_data_loaders_from_raw()
