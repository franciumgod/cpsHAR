from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib._stride_tricks_impl import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


class DataHandler:
    def __init__(
        self,
        config,
        downsample_method="false",
        downsample_window_size=40,
        downsample_window_step=20,
    ):
        self.config = config
        self.scaler = None
        self.downsample_method = (
            str(downsample_method).lower() if downsample_method is not None else "false"
        )
        self.downsample_window_size = downsample_window_size
        self.downsample_window_step = downsample_window_step

        self.data_root = Path(r"D:\Code\research_intern\Pal2sim\cpsHAR\data")
        self.local_path = self.data_root / Path(self.config.data.dataset_file).name
        self.data = None
        self.dataset_kind = None
        self.sample_manifest_meta = {}
        self.raw_source_data = None
        self._source_segment_cache = {}

        self._load_data_set()

    def _get_merged_data(self, meta_subset):
        merged = [row["data"] for _, row in meta_subset.iterrows()]
        return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

    def _clean(self, df):
        cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        return df.reset_index(drop=True)

    def _add_new_signal(self, df):
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

    def _get_challenge_data_numpy(self, df, seq_len, sensor_cols, label_cols):
        data_values = df[sensor_cols].to_numpy(dtype=np.float32)
        label_values = df[label_cols].to_numpy(dtype=np.int8)

        X_view = sliding_window_view(data_values, window_shape=seq_len, axis=0)
        y_start_index = seq_len - 1
        y_view = label_values[y_start_index: y_start_index + len(X_view)]

        min_len = min(len(X_view), len(y_view))
        X_view = X_view[:min_len]
        y_view = y_view[:min_len]
        return X_view, y_view

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

        raise ValueError(
            f"Unsupported dataset format in {self.local_path}. "
            "Expected raw metadata dataframe or sample manifest payload."
        )

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
                    self.config.data.sensor_cols = list(payload["sensor_cols"])
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
                self.config.data.sensor_cols = list(payload["sensor_cols"])
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
            raise ValueError(
                f"Unsupported downsample_method='{self.downsample_method}'. "
                "Expected one of: false, interval, sliding_window."
            )

        print(f"Downsample [{split_name}] with method={method}: {len(df)} -> {len(ds_df)} rows")
        return ds_df

    def _maybe_downsample_window_array(self, X, split_name):
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
                raise ValueError(
                    f"Window length {X.shape[1]} is smaller than sliding downsample window {win}."
                )

            windows = sliding_window_view(X, window_shape=win, axis=1)
            windows = windows[:, ::step, :, :]
            X_ds = windows.mean(axis=-1, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported downsample_method='{self.downsample_method}'. "
                "Expected one of: false, interval, sliding_window."
            )

        print(
            f"Downsample [{split_name}] sample windows with method={method}: "
            f"{X.shape[1]} -> {X_ds.shape[1]} steps"
        )
        return X_ds.astype(np.float32, copy=False)

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
            return empty_x, empty_y

        window_size = int(sample_df.iloc[0]["end_idx"] - sample_df.iloc[0]["start_idx"])
        X = np.empty((len(sample_df), window_size, len(self.config.data.sensor_cols)), dtype=np.float32)

        for row_idx, row in enumerate(sample_df.itertuples(index=False)):
            processed_df = self._get_processed_source_df(int(row.source_index), final_target_cols)
            sample_values = processed_df.iloc[int(row.start_idx):int(row.end_idx)][
                self.config.data.sensor_cols
            ].to_numpy(dtype=np.float32, copy=False)
            X[row_idx] = sample_values

        y = sample_df[final_target_cols].to_numpy(dtype=np.int8, copy=False)
        return X, y

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
        train_df = self._maybe_downsample(train_df, self.config.data.sensor_cols, final_target_cols, "train")
        validation_df = self._maybe_downsample(
            validation_df,
            self.config.data.sensor_cols,
            final_target_cols,
            "val",
        )
        test_df = self._maybe_downsample(test_df, self.config.data.sensor_cols, final_target_cols, "test")

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)

        seq_len = self._get_effective_seq_len_for_raw()
        train_x, train_y = self._get_challenge_data_numpy(
            train_df,
            seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )
        val_x, val_y = self._get_challenge_data_numpy(
            validation_df,
            seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )
        test_x, test_y = self._get_challenge_data_numpy(
            test_df,
            seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )

        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def _get_data_loaders_from_sample_manifest(self):
        test_mask = self.data["experiment"] == self.config.data.test_experiment_id
        validation_mask = self.data["experiment"] == self.config.data.validation_experiment_id

        train_manifest = self.data[~test_mask & ~validation_mask]
        test_manifest = self.data[test_mask]
        validation_manifest = self.data[validation_mask]

        final_target_cols = self._get_final_target_cols()
        train_x, train_y = self._materialize_window_samples(train_manifest, final_target_cols)
        val_x, val_y = self._materialize_window_samples(validation_manifest, final_target_cols)
        test_x, test_y = self._materialize_window_samples(test_manifest, final_target_cols)

        train_x = self._maybe_downsample_window_array(train_x, "train")
        val_x = self._maybe_downsample_window_array(val_x, "val")
        test_x = self._maybe_downsample_window_array(test_x, "test")

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def _get_data_loaders_from_window_samples(self):
        payload = self.data
        X = payload["X"]
        y = payload["y"]
        experiment = payload["experiment"]
        final_target_cols = list(payload["label_cols"])

        test_mask = experiment == self.config.data.test_experiment_id
        validation_mask = experiment == self.config.data.validation_experiment_id
        train_mask = (~test_mask) & (~validation_mask)

        train_x = X[train_mask]
        train_y = y[train_mask]
        val_x = X[validation_mask]
        val_y = y[validation_mask]
        test_x = X[test_mask]
        test_y = y[test_mask]

        train_x = self._maybe_downsample_window_array(train_x, "train")
        val_x = self._maybe_downsample_window_array(val_x, "val")
        test_x = self._maybe_downsample_window_array(test_x, "test")

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        train_x, val_x, test_x = self._fit_and_apply_scaler(train_x, val_x, test_x)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

    def get_data_loaders(self):
        print("Starting data preparation...", flush=True)
        self._source_segment_cache = {}

        if self.dataset_kind == "raw":
            return self._get_data_loaders_from_raw()
        if self.dataset_kind == "sample_manifest":
            return self._get_data_loaders_from_sample_manifest()
        if self.dataset_kind == "window_samples":
            return self._get_data_loaders_from_window_samples()

        raise ValueError(f"Unsupported dataset kind: {self.dataset_kind}")
