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

        filename = Path(self.config.data.dataset_file).name
        self.local_path = Path(r"D:\Code\research_intern\Pal2sim\cpsHAR\data") / filename
        self.data = None

        self._load_data_set()

    def _get_merged_data(self, meta_subset):
        merged = [row["data"] for _, row in meta_subset.iterrows()]
        return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

    def _clean(self, df):
        cols_to_drop = ["Error", "Synchronization", "None", "transportation", "container"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")
        return df.reset_index(drop=True)

    def _add_new_signal(self, df):
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
        data_values = df[sensor_cols].values
        label_values = df[label_cols].values

        X_view = sliding_window_view(data_values, window_shape=seq_len, axis=0)

        y_start_index = seq_len - 1
        y_view = label_values[y_start_index: y_start_index + len(X_view)]

        min_len = min(len(X_view), len(y_view))
        X_view = X_view[:min_len]
        y_view = y_view[:min_len]

        return X_view, y_view

    def _load_data_set(self):
        """
        Checks if data exists locally. If not, prints a message.
        Then loads the data into memory.
        """
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.local_path.exists():
            print(f"Dataset not found at {self.local_path}")
        else:
            print(f"Local dataset found: {self.local_path}")

        print("Loading data into memory...")
        df = pd.read_pickle(self.local_path)
        print("Data loaded.")
        self.data = df

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

        for i, s in enumerate(starts):
            e = s + win
            sensor_ds[i] = sensor_values[s:e].mean(axis=0, dtype=np.float32)
            if label_cols:
                label_ds[i] = label_values[e - 1]

        out_df = pd.DataFrame(sensor_ds, columns=sensor_cols)
        if label_cols:
            for col_idx, col_name in enumerate(label_cols):
                out_df[col_name] = label_ds[:, col_idx]

        return out_df.reset_index(drop=True)

    def _maybe_downsample(self, df, sensor_cols, label_cols, split_name):
        method = self.downsample_method
        if method == "interval":
            ds_df = self._downsample_interval(df, sensor_cols, label_cols)
        elif method == "sliding_window":
            ds_df = self._downsample_sliding_window(df, sensor_cols, label_cols)
        print(f"Downsample [{split_name}] with method={method}: {len(df)} -> {len(ds_df)} rows")
        return ds_df

    def get_data_loaders(self):
        print("Starting data preparation...", flush=True)

        test_mask = self.data["experiment"] == self.config.data.test_experiment_id
        validation_mask = self.data["experiment"] == self.config.data.validation_experiment_id

        train_metadata = self.data[~test_mask & ~validation_mask]
        test_metadata = self.data[test_mask]
        validation_metadata = self.data[validation_mask]

        train_df = self._get_merged_data(train_metadata)
        test_df = self._get_merged_data(test_metadata)
        validation_df = self._get_merged_data(validation_metadata)

        train_df = self._clean(train_df)
        test_df = self._clean(test_df)
        validation_df = self._clean(validation_df)

        train_df = self._add_new_signal(train_df)
        test_df = self._add_new_signal(test_df)
        validation_df = self._add_new_signal(validation_df)

        train_df = self._apply_superclass_mapping(train_df, self.config.data.superclass_mapping)
        validation_df = self._apply_superclass_mapping(
            validation_df, self.config.data.superclass_mapping
        )
        test_df = self._apply_superclass_mapping(test_df, self.config.data.superclass_mapping)

        unique_superclasses = set(self.config.data.superclass_mapping.values())
        final_target_cols = sorted(list(unique_superclasses))

        train_df = self._maybe_downsample(
            train_df, self.config.data.sensor_cols, final_target_cols, "train"
        )
        validation_df = self._maybe_downsample(
            validation_df, self.config.data.sensor_cols, final_target_cols, "val"
        )
        test_df = self._maybe_downsample(
            test_df, self.config.data.sensor_cols, final_target_cols, "test"
        )

        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[self.config.data.sensor_cols] = self.scaler.fit_transform(
            train_df[self.config.data.sensor_cols]
        )
        validation_df[self.config.data.sensor_cols] = self.scaler.transform(
            validation_df[self.config.data.sensor_cols]
        )
        test_df[self.config.data.sensor_cols] = self.scaler.transform(
            test_df[self.config.data.sensor_cols]
        )

        train_x, train_y = self._get_challenge_data_numpy(
            train_df,
            self.config.prep.seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )
        val_x, val_y = self._get_challenge_data_numpy(
            validation_df,
            self.config.prep.seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )
        test_x, test_y = self._get_challenge_data_numpy(
            test_df,
            self.config.prep.seq_len,
            self.config.data.sensor_cols,
            final_target_cols,
        )

        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols