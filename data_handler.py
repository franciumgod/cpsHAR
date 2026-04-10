from pathlib import Path
import pandas as pd
import numpy as np
from numpy.lib._stride_tricks_impl import sliding_window_view
from sklearn.preprocessing import MinMaxScaler


def get_merged_data(meta_subset):
    merged = [row["data"] for _, row in meta_subset.iterrows()]
    return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

def clean(df):
    cols_to_drop = ["Error", "Synchronization", "None","transportation","container"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    return df.reset_index(drop=True)

def add_new_signal(df):
    df["Acc.norm"]= np.sqrt(np.asarray(df["Acc.x"])**2+np.asarray(df["Acc.y"])**2+np.asarray(df["Acc.z"])**2)
    df["Gyro.norm"] = np.sqrt(np.asarray(df["Gyro.x"])**2+np.asarray(df["Gyro.y"])**2+np.asarray(df["Gyro.z"])**2)
    return df

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler = None
        filename = Path(self.config.data.dataset_file).name
        self.local_path = Path(f"D:\\Code\\research_intern\\Pal2sim\\cpsHAR\\data") / filename
        self.data = None
        self._load_data_set()

    @staticmethod
    def _get_merged_data(meta_subset):
        merged = [row["data"] for _, row in meta_subset.iterrows()]
        return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

    def _get_challenge_data_numpy(self, df, seq_len, sensor_cols, label_cols):

        data_values = df[sensor_cols].values
        label_values = df[label_cols].values

        X_view = sliding_window_view(data_values, window_shape=seq_len, axis=0)

        y_start_index = seq_len - 1
        y_view = label_values[y_start_index : y_start_index + len(X_view)]

        min_len = min(len(X_view), len(y_view))
        X_view = X_view[:min_len]
        y_view = y_view[:min_len]

        return X_view, y_view

    def _load_data_set(self):
        """
        Checks if data exists locally. If not, downloads it.
        Then loads the data into memory.
        """
        # ensure the ./data/ directory exists
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

        # check if file exists
        if not self.local_path.exists():
            print(f"Dataset not found at {self.local_path}")
        else:
            print(f"Local dataset found: {self.local_path}")

        # load dataset
        try:
            print("Loading data into memory...")
            df = pd.read_pickle(self.local_path)
            print(f"Data loaded.")
            self.data = df
        except Exception as e:
            print(f"Error loading file: {e}")
            raise e

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

        cols_to_drop = [k for k in mapping.keys() if k in df.columns and k not in target_superclasses]

        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        return df

    def get_data_loaders(self):
        print("Starting data preparation...", flush=True)
        #df_final = self.load_data_set()

        test_mask = self.data['experiment'] == self.config.data.test_experiment_id
        validation_mask = self.data['experiment'] == self.config.data.validation_experiment_id
        train_metadata = self.data[~test_mask & ~validation_mask]
        test_metadata = self.data[test_mask]
        validation_metadata = self.data[validation_mask]

        train_df = self._get_merged_data(train_metadata)
        test_df = self._get_merged_data(test_metadata)
        validation_df = self._get_merged_data(validation_metadata)

        train_df = clean(train_df)
        test_df = clean(test_df)
        validation_df = clean(validation_df)

        train_df = add_new_signal(train_df)
        test_df = add_new_signal(test_df)
        validation_df = add_new_signal(validation_df)


        train_df = self._apply_superclass_mapping(train_df, self.config.data.superclass_mapping)
        validation_df = self._apply_superclass_mapping(validation_df, self.config.data.superclass_mapping)
        test_df = self._apply_superclass_mapping(test_df, self.config.data.superclass_mapping)
        unique_superclasses = set(self.config.data.superclass_mapping.values())
        final_target_cols = sorted(list(unique_superclasses))


        self.config.IN_CHANNELS = len(self.config.data.sensor_cols)
        #final_target_cols = self.config.data.label_cols


        # --- SCALING ---
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[self.config.data.sensor_cols] = self.scaler.fit_transform(train_df[self.config.data.sensor_cols])
        validation_df[self.config.data.sensor_cols] = self.scaler.transform(validation_df[self.config.data.sensor_cols])
        test_df[self.config.data.sensor_cols] = self.scaler.transform(test_df[self.config.data.sensor_cols])



        # --- CREATE DATASETS ---
        train_x, train_y = self._get_challenge_data_numpy(train_df, self.config.prep.seq_len, self.config.data.sensor_cols, final_target_cols)
        val_x, val_y = self._get_challenge_data_numpy(validation_df, self.config.prep.seq_len, self.config.data.sensor_cols, final_target_cols)
        test_x, test_y = self._get_challenge_data_numpy(test_df, self.config.prep.seq_len, self.config.data.sensor_cols, final_target_cols)

        return (train_x, train_y), (val_x, val_y), (test_x, test_y), final_target_cols

