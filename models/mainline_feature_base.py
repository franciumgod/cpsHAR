import abc
import gc

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class _ConstantBinaryEstimator:
    def __init__(self, positive_label=1):
        self.positive_label = int(positive_label)
        self.prob_ = 0.0

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        self.prob_ = float(np.mean(y == self.positive_label)) if len(y) > 0 else 0.0
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n = len(X)
        pos = np.full((n,), self.prob_, dtype=np.float32)
        neg = 1.0 - pos
        return np.column_stack([neg, pos]).astype(np.float32, copy=False)


class MainlineFeatureOVRBase(abc.ABC):
    def __init__(
        self,
        classes,
        *,
        model_display_name,
        use_val_in_train=False,
        random_state=42,
        subsample_method="random",
        n_train_data_samples=100_000,
        feature_batch_size=2000,
        pre_downsample_method="false",
        pre_downsample_factor=1,
        pre_downsample_window_size=40,
        pre_downsample_window_step=20,
        signal_combo_map=None,
        sensor_cols=None,
        feature_engineering=True,
        feature_domain="time",
        spectrum_method="rfft",
    ):
        self.classes = classes
        self.model_display_name = str(model_display_name)
        self.use_val_in_train = bool(use_val_in_train)
        self.random_state = int(random_state)
        self.subsample_method = str(subsample_method).lower() if subsample_method is not None else "false"
        self._n_train_data_samples = int(n_train_data_samples)
        self._feature_batch_size = int(feature_batch_size)
        self.pre_downsample_method = (
            str(pre_downsample_method).lower() if pre_downsample_method is not None else "false"
        )
        self.pre_downsample_factor = max(1, int(pre_downsample_factor))
        self.pre_downsample_window_size = int(pre_downsample_window_size)
        self.pre_downsample_window_step = int(pre_downsample_window_step)
        self.signal_combo_map = signal_combo_map or {}
        self.sensor_cols = list(sensor_cols) if sensor_cols is not None else []
        self.feature_engineering = str(feature_engineering).strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        self.feature_domain = str(feature_domain).strip().lower()
        if self.feature_domain not in {"time", "freq", "time_freq"}:
            self.feature_domain = "time"
        self.spectrum_method = self._normalize_spectrum_method(spectrum_method)
        self._input_scaler = None
        self.models = []

    @staticmethod
    def _normalize_spectrum_method(value):
        text = str(value).strip().lower()
        compact = text.replace(" ", "").replace("-", "").replace("_", "")
        if compact in {"rfft", "fft"}:
            return "rfft"
        if compact in {"welch", "welchpsd", "psd"}:
            return "welch_psd"
        if compact in {"stft"}:
            return "stft"
        if compact in {"dwt", "wavelet"}:
            return "dwt"
        return "rfft"

    def set_input_scaler(self, scaler):
        self._input_scaler = scaler

    def _select_windows_for_label(self, X, label_idx):
        if not self.signal_combo_map:
            return X
        class_name = self.classes[label_idx] if label_idx < len(self.classes) else None
        idxs = self.signal_combo_map.get(class_name)
        if not idxs:
            return X
        return X[:, :, idxs]

    def _maybe_downsample_windows(self, X):
        method = self.pre_downsample_method
        if method in {"false", "none", "0"}:
            return np.asarray(X, dtype=np.float32)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            return X

        transposed_input = False
        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
            transposed_input = True

        if method == "interval":
            X_ds = X[:, :: self.pre_downsample_factor, :]
        elif method == "sliding_window":
            win = self.pre_downsample_window_size
            step = self.pre_downsample_window_step
            if X.shape[1] < win:
                X_ds = X
            else:
                windows = sliding_window_view(X, window_shape=win, axis=1)
                windows = windows[:, ::step, :, :]
                X_ds = windows.mean(axis=-1, dtype=np.float32)
        else:
            X_ds = X

        if transposed_input:
            X_ds = np.transpose(X_ds, (0, 2, 1))
        return X_ds.astype(np.float32, copy=False)

    def _extract_time_features(self, X):
        mean = np.mean(X, axis=1, dtype=np.float32)
        std = np.std(X, axis=1, dtype=np.float32)
        max_val = np.max(X, axis=1)
        min_val = np.min(X, axis=1)

        feat_blocks = [mean, std, max_val, min_val]
        if self.feature_engineering:
            rms = np.sqrt(np.mean(X * X, axis=1, dtype=np.float32))
            centered = X - mean[:, None, :]
            rmse = np.sqrt(np.mean(centered * centered, axis=1, dtype=np.float32))

            if X.shape[1] > 1:
                diff_1 = np.diff(X, n=1, axis=1)
                diff_1_mean = np.mean(diff_1, axis=1, dtype=np.float32)
            else:
                diff_1_mean = np.zeros_like(mean, dtype=np.float32)

            lag_blocks = []
            last_idx = X.shape[1] - 1
            for lag in (1, 3, 5, 10):
                lag_idx = max(0, last_idx - lag)
                lag_blocks.append(X[:, lag_idx, :])

            feat_blocks.extend([rms, rmse, diff_1_mean] + lag_blocks)
        return np.hstack(feat_blocks).astype(np.float32, copy=False)

    def _build_rfft_spectrum(self, X):
        spectrum = np.fft.rfft(X, axis=1)
        return np.abs(spectrum).astype(np.float32, copy=False)

    def _build_welch_psd_spectrum(self, X):
        _, t, _ = X.shape
        if t < 4:
            return self._build_rfft_spectrum(X)

        nperseg = min(256, t)
        step = max(1, nperseg // 2)
        starts = list(range(0, t - nperseg + 1, step))
        if not starts:
            starts = [0]

        window = np.hanning(nperseg).astype(np.float32)
        win_norm = float(np.sum(window * window) + 1e-8)
        psd_acc = None

        for s in starts:
            seg = X[:, s : s + nperseg, :] * window[None, :, None]
            fft = np.fft.rfft(seg, axis=1)
            psd = (np.abs(fft) ** 2).astype(np.float32, copy=False) / win_norm
            psd_acc = psd if psd_acc is None else (psd_acc + psd)

        return (psd_acc / float(len(starts))).astype(np.float32, copy=False)

    def _build_stft_spectrum(self, X):
        _, t, _ = X.shape
        if t < 4:
            return self._build_rfft_spectrum(X)

        nperseg = min(256, t)
        step = max(1, nperseg // 2)
        starts = list(range(0, t - nperseg + 1, step))
        if not starts:
            starts = [0]

        window = np.hanning(nperseg).astype(np.float32)
        mags = []
        for s in starts:
            seg = X[:, s : s + nperseg, :] * window[None, :, None]
            fft = np.fft.rfft(seg, axis=1)
            mags.append(np.abs(fft).astype(np.float32, copy=False))

        stacked = np.stack(mags, axis=1)
        return np.mean(stacked, axis=1, dtype=np.float32)

    def _build_dwt_spectrum(self, X):
        n, t, c = X.shape
        if t < 2:
            return np.abs(X).astype(np.float32, copy=False)

        even_t = t - (t % 2)
        paired = X[:, :even_t, :].reshape(n, even_t // 2, 2, c)
        approx = (paired[:, :, 0, :] + paired[:, :, 1, :]) / np.sqrt(2.0)
        detail = (paired[:, :, 0, :] - paired[:, :, 1, :]) / np.sqrt(2.0)
        coeff = np.concatenate([np.abs(approx), np.abs(detail)], axis=1).astype(np.float32, copy=False)

        if t % 2 == 1:
            tail = np.abs(X[:, -1:, :]).astype(np.float32, copy=False)
            coeff = np.concatenate([coeff, tail], axis=1)
        return coeff

    def _build_spectrum(self, X):
        method = self.spectrum_method
        if method == "welch_psd":
            return self._build_welch_psd_spectrum(X)
        if method == "stft":
            return self._build_stft_spectrum(X)
        if method == "dwt":
            return self._build_dwt_spectrum(X)
        return self._build_rfft_spectrum(X)

    def _extract_freq_features(self, X):
        mag = self._build_spectrum(X)

        mean = np.mean(mag, axis=1, dtype=np.float32)
        std = np.std(mag, axis=1, dtype=np.float32)
        max_val = np.max(mag, axis=1)
        min_val = np.min(mag, axis=1)
        energy = np.mean(mag * mag, axis=1, dtype=np.float32)

        freqs = np.linspace(0.0, 1.0, num=mag.shape[1], dtype=np.float32)
        denom = np.sum(mag, axis=1, dtype=np.float32) + 1e-8
        centroid = np.sum(mag * freqs[None, :, None], axis=1, dtype=np.float32) / denom
        spread = np.sqrt(
            np.sum(
                ((freqs[None, :, None] - centroid[:, None, :]) ** 2) * mag,
                axis=1,
                dtype=np.float32,
            ) / denom
        ).astype(np.float32, copy=False)

        return np.hstack([mean, std, max_val, min_val, energy, centroid, spread]).astype(
            np.float32, copy=False
        )

    def _extract_features(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, but got shape {X.shape}")

        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        feat_blocks = []
        if self.feature_domain in {"time", "time_freq"}:
            feat_blocks.append(self._extract_time_features(X))
        if self.feature_domain in {"freq", "time_freq"}:
            feat_blocks.append(self._extract_freq_features(X))
        if not feat_blocks:
            feat_blocks.append(self._extract_time_features(X))
        return np.hstack(feat_blocks).astype(np.float32, copy=False)

    def _extract_features_batched(self, X):
        X = np.asarray(X)
        n_samples = len(X)
        if n_samples == 0:
            return np.empty((0, 0), dtype=np.float32)

        batch_size = max(1, int(self._feature_batch_size))
        feat_batches = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            feat_batches.append(self._extract_features(X[start:end]))
        return np.vstack(feat_batches) if len(feat_batches) > 1 else feat_batches[0]

    def _subsample_train_data(self, X, y):
        method = self.subsample_method
        n_total = len(X)
        if method in {"false", "none", "0"}:
            return X, y

        n_samples = min(self._n_train_data_samples, n_total)
        if n_samples >= n_total:
            return X, y

        rng = np.random.RandomState(self.random_state)
        if method == "random":
            idx = rng.choice(n_total, n_samples, replace=False)
        elif method == "interval":
            idx = np.linspace(0, n_total - 1, n_samples, dtype=int)
        else:
            idx = np.linspace(0, n_total - 1, n_samples, dtype=int)
        return X[idx], y[idx]

    def _prepare_training_arrays(self, train_data, val_data):
        train_X, train_y = train_data[:2]
        val_X, val_y = val_data[:2]

        if self.use_val_in_train:
            base_X = np.concatenate([train_X, val_X], axis=0)
            base_y = np.concatenate([train_y, val_y], axis=0)
            sampled_X, sampled_y = self._subsample_train_data(base_X, base_y)
            split_name = "merged train+val"
        else:
            sampled_X, sampled_y = self._subsample_train_data(train_X, train_y)
            split_name = "train only"

        sampled_X = self._maybe_downsample_windows(sampled_X)
        y_fit_2d = np.asarray(sampled_y)
        if y_fit_2d.ndim == 1:
            y_fit_2d = y_fit_2d.reshape(-1, 1)

        if len(sampled_X) == 0:
            raise ValueError("No training samples available after subsampling.")
        return sampled_X, y_fit_2d, split_name

    @abc.abstractmethod
    def _build_estimator(self, label_idx):
        raise NotImplementedError

    def _fit_estimator(self, estimator, fit_feat, y_train_bin, label_idx):
        estimator.fit(fit_feat, y_train_bin)
        return estimator

    def _predict_proba_for_estimator(self, estimator, feat):
        out = estimator.predict_proba(feat)
        out = np.asarray(out)
        if out.ndim == 1:
            pos = out.astype(np.float32, copy=False)
        elif out.shape[1] == 1:
            pos = out[:, 0].astype(np.float32, copy=False)
        else:
            pos = out[:, -1].astype(np.float32, copy=False)
        return pos

    def train(self, train_data, val_data):
        sampled_X, y_fit_2d, split_name = self._prepare_training_arrays(train_data, val_data)

        base_feat = None
        if not self.signal_combo_map:
            base_feat = self._extract_features_batched(sampled_X)

        feature_dim = base_feat.shape[1] if base_feat is not None else "dynamic"
        print(
            f"Start training multi-label {self.model_display_name} with {split_name}: "
            f"{sampled_X.shape[0]} samples, {feature_dim} features, "
            f"{y_fit_2d.shape[1]} labels | "
            f"feature_domain={self.feature_domain}, spectrum_method={self.spectrum_method}..."
        )

        self.models = []
        for label_idx in range(y_fit_2d.shape[1]):
            y_train_bin = y_fit_2d[:, label_idx].astype(np.int8, copy=False)
            if base_feat is not None:
                fit_feat = base_feat
            else:
                label_X = self._select_windows_for_label(sampled_X, label_idx)
                fit_feat = self._extract_features_batched(label_X)

            if len(np.unique(y_train_bin)) < 2:
                estimator = _ConstantBinaryEstimator().fit(fit_feat, y_train_bin)
                print(
                    f"[OvR] {self.classes[label_idx]} has a single class in the training pool. "
                    f"Using a constant-probability fallback."
                )
            else:
                estimator = self._build_estimator(label_idx)
                estimator = self._fit_estimator(estimator, fit_feat, y_train_bin, label_idx)
            self.models.append(estimator)

        del sampled_X, y_fit_2d, base_feat
        gc.collect()
        print("Training done.")

    def _predict_proba_single(self, test_X):
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        base_feat = None
        if not self.signal_combo_map:
            base_feat = self._extract_features(test_X)

        probs = []
        for label_idx, estimator in enumerate(self.models):
            if base_feat is not None:
                feat = base_feat
            else:
                label_X = self._select_windows_for_label(test_X, label_idx)
                feat = self._extract_features(label_X)
            probs.append(self._predict_proba_for_estimator(estimator, feat))
        return np.column_stack(probs).astype(np.float32, copy=False)

    def predict_proba(self, test_X):
        test_X = self._maybe_downsample_windows(test_X)
        return self._predict_proba_single(test_X)

    def predict(self, test_X):
        probs = self.predict_proba(test_X)
        return (probs >= 0.5).astype(int)
