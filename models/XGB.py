import numpy as np
from xgboost import XGBClassifier
import gc
from numpy.lib._stride_tricks_impl import sliding_window_view


class XGBoostClassifierSK:
    def __init__(
        self,
        classes,
        use_val_in_train=False,
        use_gpu=False,
        random_state=42,
        subsample_method="random",
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stopping_rounds=100,
        n_train_data_samples=100000,
        feature_batch_size=2000,
        pre_downsample_method="false",
        pre_downsample_factor=1,
        pre_downsample_window_size=40,
        pre_downsample_window_step=20,
        signal_combo_map=None,
        sensor_cols=None,
        augment_samples=False,
        augment_target="multilabel",
        augment_noise_ratio=0.03,
        augment_method="jitter",
        mixup_alpha=0.4,
        cutmix_ratio=0.3,
        rotation_plane="xyz",
        rotation_max_degrees=15.0,
        tta_samples=False,
        tta_method="jitter",
        feature_engineering=True,
        use_tsfresh=False,
        feature_domain="time",
        spectrum_method="rfft",
        pos_threshold="",
        ovr_neg_balance=False,
        ovr_pos_neg_ratio=1.0,
        ovr_neg_target_ratio="",
    ):
        self.classes = classes
        self.use_val_in_train = use_val_in_train
        self.random_state = random_state
        self.subsample_method = str(subsample_method).lower() if subsample_method is not None else "false"
        self.early_stopping_rounds = early_stopping_rounds
        self._n_train_data_samples = n_train_data_samples
        self._feature_batch_size = feature_batch_size
        self.pre_downsample_method = (
            str(pre_downsample_method).lower() if pre_downsample_method is not None else "false"
        )
        self.pre_downsample_factor = max(1, int(pre_downsample_factor))
        self.pre_downsample_window_size = int(pre_downsample_window_size)
        self.pre_downsample_window_step = int(pre_downsample_window_step)
        self.signal_combo_map = signal_combo_map or {}
        self.sensor_cols = list(sensor_cols) if sensor_cols is not None else []
        self.rng = np.random.default_rng(random_state)
        self.augment_count = self._parse_augment_count(augment_samples)
        self.augment_target = str(augment_target).strip()
        self.augment_noise_ratio = float(augment_noise_ratio)
        self.augment_methods = self._parse_method_list(augment_method, default="jitter")
        self.mixup_alpha = max(1e-6, float(mixup_alpha))
        self.cutmix_ratio = float(np.clip(float(cutmix_ratio), 0.0, 1.0))
        self.rotation_plane = str(rotation_plane).strip().lower()
        self.rotation_max_degrees = float(rotation_max_degrees)
        self.tta_count = self._parse_augment_count(tta_samples)
        self.tta_methods = self._parse_method_list(tta_method, default="jitter")
        self._input_scaler = None
        self.feature_engineering = str(feature_engineering).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.use_tsfresh = str(use_tsfresh).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.feature_domain = str(feature_domain).strip().lower()
        if self.feature_domain not in {"time", "freq", "time_freq"}:
            self.feature_domain = "time"
        self.spectrum_method = self._normalize_spectrum_method(spectrum_method)
        self._label_index_map = {
            str(name).strip().lower(): idx for idx, name in enumerate(self.classes)
        }
        self.pos_threshold_map = self._parse_pos_threshold_map(pos_threshold)
        self.ovr_neg_balance = str(ovr_neg_balance).strip().lower() in {
            "1", "true", "yes", "y", "on"
        }
        self.ovr_pos_neg_ratio = max(1e-6, float(ovr_pos_neg_ratio))
        self.ovr_neg_target_ratio_map = self._parse_ovr_neg_target_ratio_map(ovr_neg_target_ratio)

        self._model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "objective": "binary:logistic",
            "n_jobs": -1,
            "random_state": random_state,
            "tree_method": "hist",
            "device": "cuda" if use_gpu else "cpu",
            "eval_metric": "logloss",
        }
        self.models = []

    @staticmethod
    def _parse_augment_count(value):
        if value is None:
            return 0
        if isinstance(value, bool):
            return 1 if value else 0
        text = str(value).strip().lower()
        if text in {"", "0", "false", "no", "n", "off", "none", "null"}:
            return 0
        if text in {"1", "true", "yes", "y", "on"}:
            return 1
        try:
            parsed = int(text)
        except ValueError:
            return 0
        return max(0, parsed)

    @staticmethod
    def _parse_method_list(value, default="jitter"):
        text = str(value).strip().lower() if value is not None else ""
        if text in {"", "none", "false", "0", "off", "null"}:
            return []
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            return []
        valid = {"jitter", "scaling", "rotation", "mixup", "cutmix", "smote", "basic"}
        methods = [p for p in parts if p in valid]
        return methods if methods else ([default] if default else [])

    @staticmethod
    def _parse_target_list(value):
        text = str(value).strip() if value is not None else ""
        if text == "":
            return []
        normalized = (
            text.replace(";", ",")
            .replace("|", ",")
            .replace("，", ",")
        )
        out = []
        seen = set()
        for raw in normalized.split(","):
            tok = raw.strip().strip("\"'")
            if not tok:
                continue
            key = tok.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(tok)
        return out

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

    def _parse_pos_threshold_map(self, value):
        text = str(value).strip() if value is not None else ""
        if text.lower() in {"", "none", "false", "0", "off", "null"}:
            return {}

        out = {}
        raw_items = [it.strip() for it in text.replace(";", ",").split(",") if it.strip()]
        for item in raw_items:
            if ":" not in item:
                continue
            label_text, thr_text = item.rsplit(":", 1)
            label_key = label_text.strip().lower()
            if label_key not in self._label_index_map:
                continue
            try:
                thr = float(thr_text.strip())
            except ValueError:
                continue
            out[self._label_index_map[label_key]] = float(np.clip(thr, 0.0, 1.0))
        return out

    def _parse_ovr_neg_target_ratio_map(self, value):
        text = str(value).strip() if value is not None else ""
        if text.lower() in {"", "none", "false", "0", "off", "null"}:
            return {}

        out = {}
        raw_items = [it.strip() for it in text.split(";") if it.strip()]
        for item in raw_items:
            parts = [p.strip() for p in item.split(":")]
            if len(parts) != 3:
                continue
            target_label, neg_label, pct_text = parts
            target_key = target_label.lower()
            target_idx = self._label_index_map.get(target_key)
            if target_idx is None:
                continue
            try:
                pct = float(pct_text)
            except ValueError:
                continue
            out[target_idx] = (neg_label, float(np.clip(pct, 0.0, 100.0)))
        return out

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

        # normalize to (n_samples, time_steps, n_channels)
        transposed_input = False
        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))
            transposed_input = True

        if method == "interval":
            X_ds = X[:, ::self.pre_downsample_factor, :]
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
        return np.hstack(feat_blocks).astype(np.float32)

    def _build_rfft_spectrum(self, X):
        spectrum = np.fft.rfft(X, axis=1)
        return np.abs(spectrum).astype(np.float32, copy=False)

    def _build_welch_psd_spectrum(self, X):
        n, t, _ = X.shape
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
            seg = X[:, s:s + nperseg, :] * window[None, :, None]
            fft = np.fft.rfft(seg, axis=1)
            psd = (np.abs(fft) ** 2).astype(np.float32, copy=False) / win_norm
            psd_acc = psd if psd_acc is None else (psd_acc + psd)

        return (psd_acc / float(len(starts))).astype(np.float32, copy=False)

    def _build_stft_spectrum(self, X):
        n, t, _ = X.shape
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
            seg = X[:, s:s + nperseg, :] * window[None, :, None]
            fft = np.fft.rfft(seg, axis=1)
            mags.append(np.abs(fft).astype(np.float32, copy=False))

        stacked = np.stack(mags, axis=1)  # (n, frames, bins, c)
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

        return np.hstack([mean, std, max_val, min_val, energy, centroid, spread]).astype(np.float32)

    def _extract_tsfresh_features(self, X):
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import EfficientFCParameters

        n, t, c = X.shape
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        ids = np.repeat(np.arange(n, dtype=np.int32), t * c)
        times = np.tile(np.repeat(np.arange(t, dtype=np.int32), c), n)
        kinds = np.tile(np.arange(c, dtype=np.int16), n * t)
        values = X.reshape(-1).astype(np.float32, copy=False)

        long_df = pd.DataFrame(
            {
                "id": ids,
                "time": times,
                "kind": kinds,
                "value": values,
            }
        )
        feat_df = extract_features(
            long_df,
            column_id="id",
            column_sort="time",
            column_kind="kind",
            column_value="value",
            default_fc_parameters=EfficientFCParameters(),
            disable_progressbar=True,
            n_jobs=0,
        )
        feat_df = feat_df.sort_index().fillna(0.0)
        return feat_df.to_numpy(dtype=np.float32, copy=False)

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
        if self.use_tsfresh:
            feat_blocks.append(self._extract_tsfresh_features(X))

        if not feat_blocks:
            feat_blocks.append(self._extract_time_features(X))
        return np.hstack(feat_blocks).astype(np.float32)

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

    def _build_neg_bucket_indices(self, y_2d, neg_idx, target_label_idx):
        buckets = {}
        for idx in neg_idx:
            active = np.where(y_2d[idx] == 1)[0]
            if len(active) > 1:
                bucket = "MULTI_LABEL"
            elif len(active) == 1:
                bucket = str(self.classes[int(active[0])])
            else:
                bucket = "NONE_LABEL"
            buckets.setdefault(bucket, []).append(int(idx))
        return {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}

    def _resolve_bucket_name_for_label(self, raw_name):
        name = str(raw_name).strip().lower()
        if name in {"multi", "multilabel", "multi_label", "multi-label"}:
            return "MULTI_LABEL"
        if name in {"none", "none_label", "nolabel", "no_label"}:
            return "NONE_LABEL"
        mapped_idx = self._label_index_map.get(name)
        if mapped_idx is None:
            return None
        return str(self.classes[mapped_idx])

    @staticmethod
    def _distribute_counts(total_count, keys):
        if total_count <= 0 or not keys:
            return {k: 0 for k in keys}
        base = total_count // len(keys)
        rem = total_count % len(keys)
        out = {}
        for i, key in enumerate(keys):
            out[key] = int(base + (1 if i < rem else 0))
        return out

    def _augment_bucket_deficit(self, X_source, n_needed, rng):
        if n_needed <= 0 or len(X_source) == 0:
            return np.empty((0, X_source.shape[1], X_source.shape[2]), dtype=np.float32)
        src_idx = rng.choice(len(X_source), size=n_needed, replace=True)
        source = np.asarray(X_source[src_idx], dtype=np.float32)
        augmented = self._augment_windows_only(source, "jitter", rng)
        return np.asarray(augmented, dtype=np.float32)

    def _select_ovr_train_subset(self, X, y_2d, ratio_2d, label_idx):
        y_2d = np.asarray(y_2d)
        ratio_2d = np.asarray(ratio_2d, dtype=np.float32)
        if ratio_2d.shape != y_2d.shape:
            ratio_2d = y_2d.astype(np.float32, copy=False)

        pos_idx = np.where(y_2d[:, label_idx] == 1)[0].astype(np.int64)
        neg_idx = np.where(y_2d[:, label_idx] == 0)[0].astype(np.int64)

        thr = self.pos_threshold_map.get(label_idx)
        if thr is not None and len(pos_idx) > 0:
            keep = ratio_2d[pos_idx, label_idx] >= float(thr)
            pos_idx = pos_idx[keep]

        info = {
            "label": str(self.classes[label_idx]),
            "pos_threshold": thr,
            "pos_count_after_threshold": int(len(pos_idx)),
            "neg_count_available": int(len(neg_idx)),
            "ovr_neg_balance": bool(self.ovr_neg_balance),
        }
        explicit_rule = self.ovr_neg_target_ratio_map.get(label_idx)
        neg_mix_only = (explicit_rule is not None) and (not self.ovr_neg_balance)
        info["neg_mix_only"] = bool(neg_mix_only)

        if not self.ovr_neg_balance and not neg_mix_only:
            real_idx = np.concatenate([pos_idx, neg_idx], axis=0).astype(np.int64, copy=False)
            y_real = np.concatenate(
                [
                    np.ones((len(pos_idx),), dtype=np.int8),
                    np.zeros((len(neg_idx),), dtype=np.int8),
                ],
                axis=0,
            )
            info.update(
                {
                    "target_neg_count": int(len(neg_idx)),
                    "selected_neg_real": int(len(neg_idx)),
                    "selected_neg_augmented": 0,
                    "bucket_plan": {},
                }
            )
            return real_idx, y_real, np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int8), info

        if self.ovr_neg_balance:
            target_neg = int(round(len(pos_idx) / float(self.ovr_pos_neg_ratio))) if len(pos_idx) > 0 else 0
            if len(neg_idx) > 0 and target_neg <= 0:
                target_neg = 1
        else:
            target_neg = int(len(neg_idx))
        info["target_neg_count"] = int(target_neg)

        if target_neg <= 0 or len(neg_idx) == 0:
            real_idx = pos_idx.astype(np.int64, copy=False)
            y_real = np.ones((len(real_idx),), dtype=np.int8)
            info.update(
                {
                    "selected_neg_real": 0,
                    "selected_neg_augmented": 0,
                    "bucket_plan": {},
                }
            )
            return real_idx, y_real, np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int8), info

        buckets = self._build_neg_bucket_indices(y_2d, neg_idx, target_label_idx=label_idx)
        bucket_keys = sorted(list(buckets.keys()))
        target_counts = {k: 0 for k in bucket_keys}

        if explicit_rule is not None:
            rule_bucket_raw, pct = explicit_rule
            rule_bucket = self._resolve_bucket_name_for_label(rule_bucket_raw)
            if rule_bucket is None:
                rule_bucket = "MULTI_LABEL"
            if rule_bucket not in target_counts:
                target_counts[rule_bucket] = 0
            explicit_count = int(round(target_neg * (pct / 100.0)))
            explicit_count = min(target_neg, max(0, explicit_count))
            target_counts[rule_bucket] = explicit_count
            remaining = target_neg - explicit_count
            other_keys = [k for k in target_counts.keys() if k != rule_bucket]
            for k, v in self._distribute_counts(remaining, other_keys).items():
                target_counts[k] = target_counts.get(k, 0) + v
        else:
            multi_key = "MULTI_LABEL"
            multi_avail = len(buckets.get(multi_key, np.empty((0,), dtype=np.int64)))
            use_multi = min(target_neg, multi_avail)
            if multi_key not in target_counts:
                target_counts[multi_key] = 0
            target_counts[multi_key] = use_multi
            remaining = target_neg - use_multi
            other_keys = [k for k in target_counts.keys() if k != multi_key]
            for k, v in self._distribute_counts(remaining, other_keys).items():
                target_counts[k] = target_counts.get(k, 0) + v

        rng = np.random.RandomState(self.random_state + 1009 + int(label_idx) * 13)
        real_neg_parts = []
        aug_neg_parts = []
        bucket_plan = {}
        unresolved = 0

        for bucket, tgt in target_counts.items():
            tgt = int(max(0, tgt))
            avail_idx = buckets.get(bucket, np.empty((0,), dtype=np.int64))
            avail_n = int(len(avail_idx))
            if tgt <= 0:
                bucket_plan[bucket] = {"target": 0, "real": 0, "aug": 0, "available": avail_n}
                continue

            real_n = min(avail_n, tgt)
            if real_n > 0:
                if real_n < avail_n:
                    chosen = rng.choice(avail_idx, size=real_n, replace=False).astype(np.int64)
                else:
                    chosen = avail_idx.astype(np.int64, copy=False)
                real_neg_parts.append(chosen)
            else:
                chosen = np.empty((0,), dtype=np.int64)

            deficit = tgt - real_n
            aug_n = 0
            if deficit > 0:
                if real_n > 0:
                    aug_x = self._augment_bucket_deficit(X[chosen], deficit, rng)
                    if len(aug_x) > 0:
                        aug_neg_parts.append(aug_x)
                        aug_n = int(len(aug_x))
                elif avail_n > 0:
                    aug_x = self._augment_bucket_deficit(X[avail_idx], deficit, rng)
                    if len(aug_x) > 0:
                        aug_neg_parts.append(aug_x)
                        aug_n = int(len(aug_x))
                if aug_n < deficit:
                    unresolved += (deficit - aug_n)

            bucket_plan[bucket] = {
                "target": int(tgt),
                "real": int(real_n),
                "aug": int(aug_n),
                "available": int(avail_n),
            }

        if unresolved > 0 and len(neg_idx) > 0:
            fallback_aug = self._augment_bucket_deficit(X[neg_idx], unresolved, rng)
            if len(fallback_aug) > 0:
                aug_neg_parts.append(fallback_aug)
                bucket_plan["FALLBACK_AUG"] = {
                    "target": int(unresolved),
                    "real": 0,
                    "aug": int(len(fallback_aug)),
                    "available": int(len(neg_idx)),
                }

        real_neg_idx = (
            np.concatenate(real_neg_parts, axis=0).astype(np.int64, copy=False)
            if real_neg_parts
            else np.empty((0,), dtype=np.int64)
        )
        aug_neg_x = (
            np.concatenate(aug_neg_parts, axis=0).astype(np.float32, copy=False)
            if aug_neg_parts
            else np.empty((0, X.shape[1], X.shape[2]), dtype=np.float32)
        )

        real_idx = np.concatenate([pos_idx, real_neg_idx], axis=0).astype(np.int64, copy=False)
        y_real = np.concatenate(
            [
                np.ones((len(pos_idx),), dtype=np.int8),
                np.zeros((len(real_neg_idx),), dtype=np.int8),
            ],
            axis=0,
        )
        y_aug = np.zeros((len(aug_neg_x),), dtype=np.int8)

        info.update(
            {
                "selected_neg_real": int(len(real_neg_idx)),
                "selected_neg_augmented": int(len(aug_neg_x)),
                "bucket_plan": bucket_plan,
            }
        )
        return real_idx, y_real, aug_neg_x, y_aug, info

    def _resolve_rotation_indices(self, n_channels):
        name_to_idx = {str(name): i for i, name in enumerate(self.sensor_cols)}
        required = ["Acc.x", "Acc.y", "Acc.z", "Gyro.x", "Gyro.y", "Gyro.z"]
        if all((k in name_to_idx and name_to_idx[k] < n_channels) for k in required):
            acc_idx = np.array([name_to_idx["Acc.x"], name_to_idx["Acc.y"], name_to_idx["Acc.z"]], dtype=np.int64)
            gyro_idx = np.array([name_to_idx["Gyro.x"], name_to_idx["Gyro.y"], name_to_idx["Gyro.z"]], dtype=np.int64)
            acc_norm_idx = name_to_idx["Acc.norm"] if "Acc.norm" in name_to_idx and name_to_idx["Acc.norm"] < n_channels else None
            gyro_norm_idx = name_to_idx["Gyro.norm"] if "Gyro.norm" in name_to_idx and name_to_idx["Gyro.norm"] < n_channels else None
            return acc_idx, gyro_idx, acc_norm_idx, gyro_norm_idx

        if n_channels >= 6:
            acc_norm_idx = 7 if n_channels > 7 else None
            gyro_norm_idx = 8 if n_channels > 8 else None
            return np.array([0, 1, 2], dtype=np.int64), np.array([3, 4, 5], dtype=np.int64), acc_norm_idx, gyro_norm_idx

        return None, None, None, None

    def _build_rotation_matrices(self, n_samples, max_rotation_degrees, rotation_plane):
        rotation_limit = np.deg2rad(float(max_rotation_degrees))
        plane = str(rotation_plane).strip().lower()
        if plane not in {"xyz", "xy", "xz", "yz"}:
            plane = "xyz"

        if plane == "xyz":
            angles = self.rng.uniform(-rotation_limit, rotation_limit, size=(n_samples, 3)).astype(np.float32)
        elif plane == "xy":
            angles = np.zeros((n_samples, 3), dtype=np.float32)
            angles[:, 2] = self.rng.uniform(-rotation_limit, rotation_limit, size=n_samples).astype(np.float32)
        elif plane == "xz":
            angles = np.zeros((n_samples, 3), dtype=np.float32)
            angles[:, 1] = self.rng.uniform(-rotation_limit, rotation_limit, size=n_samples).astype(np.float32)
        else:
            angles = np.zeros((n_samples, 3), dtype=np.float32)
            angles[:, 0] = self.rng.uniform(-rotation_limit, rotation_limit, size=n_samples).astype(np.float32)

        ax, ay, az = angles[:, 0], angles[:, 1], angles[:, 2]
        cos_x, sin_x = np.cos(ax), np.sin(ax)
        cos_y, sin_y = np.cos(ay), np.sin(ay)
        cos_z, sin_z = np.cos(az), np.sin(az)

        matrices = np.empty((n_samples, 3, 3), dtype=np.float32)
        matrices[:, 0, 0] = cos_y * cos_z
        matrices[:, 0, 1] = cos_z * sin_x * sin_y - cos_x * sin_z
        matrices[:, 0, 2] = sin_x * sin_z + cos_x * cos_z * sin_y
        matrices[:, 1, 0] = cos_y * sin_z
        matrices[:, 1, 1] = cos_x * cos_z + sin_x * sin_y * sin_z
        matrices[:, 1, 2] = cos_x * sin_y * sin_z - cos_z * sin_x
        matrices[:, 2, 0] = -sin_y
        matrices[:, 2, 1] = cos_y * sin_x
        matrices[:, 2, 2] = cos_x * cos_y
        return matrices

    def _apply_rotation_3d(self, X):
        X_in = np.asarray(X, dtype=np.float32)
        if X_in.ndim != 3 or len(X_in) == 0:
            return X_in

        n, t, c = X_in.shape
        acc_idx, gyro_idx, acc_norm_idx, gyro_norm_idx = self._resolve_rotation_indices(c)
        if acc_idx is None or gyro_idx is None:
            return X_in

        use_scaler = self._input_scaler is not None and hasattr(self._input_scaler, "inverse_transform") and hasattr(self._input_scaler, "transform")
        if use_scaler:
            flat = X_in.reshape(-1, c)
            unscaled = self._input_scaler.inverse_transform(flat).reshape(n, t, c).astype(np.float32, copy=False)
        else:
            unscaled = X_in.copy()

        rot_matrices = self._build_rotation_matrices(
            n_samples=n,
            max_rotation_degrees=self.rotation_max_degrees,
            rotation_plane=self.rotation_plane,
        )

        acc_data = unscaled[:, :, acc_idx]
        gyro_data = unscaled[:, :, gyro_idx]
        rotated_acc = np.einsum("nij,ntj->nti", rot_matrices, acc_data)
        rotated_gyro = np.einsum("nij,ntj->nti", rot_matrices, gyro_data)

        unscaled[:, :, acc_idx] = rotated_acc
        unscaled[:, :, gyro_idx] = rotated_gyro
        if acc_norm_idx is not None:
            unscaled[:, :, acc_norm_idx] = np.linalg.norm(rotated_acc, axis=2)
        if gyro_norm_idx is not None:
            unscaled[:, :, gyro_norm_idx] = np.linalg.norm(rotated_gyro, axis=2)

        if use_scaler:
            flat = unscaled.reshape(-1, c)
            scaled = self._input_scaler.transform(flat).reshape(n, t, c)
            return np.ascontiguousarray(scaled, dtype=np.float32)
        return np.ascontiguousarray(unscaled, dtype=np.float32)

    def _augment_windows_only(self, base_x, method, rng):
        X = np.asarray(base_x, dtype=np.float32)
        if method == "jitter":
            noise = rng.normal(0.0, self.augment_noise_ratio, size=X.shape).astype(np.float32)
            std = np.std(X, axis=1, keepdims=True).astype(np.float32)
            std = np.maximum(std, 1e-6)
            return X + noise * std

        if method == "scaling":
            scale = rng.normal(1.0, self.augment_noise_ratio, size=(X.shape[0], 1, X.shape[2])).astype(np.float32)
            return X * scale

        if method == "rotation":
            return self._apply_rotation_3d(X)

        if method == "basic":
            out = self._augment_windows_only(X, "jitter", rng)
            out = self._augment_windows_only(out, "scaling", rng)
            out = self._augment_windows_only(out, "rotation", rng)
            return out

        return X

    def _augment_pairwise(self, base_x, base_y, method, rng):
        X = np.asarray(base_x, dtype=np.float32)
        y = np.asarray(base_y)
        n = len(X)
        if n == 0:
            return X, y
        if n == 1:
            return self._augment_windows_only(X, "jitter", rng), y

        perm = rng.permutation(n)
        partner_x = X[perm]
        partner_y = y[perm]

        if method == "mixup":
            lam = rng.beta(self.mixup_alpha, self.mixup_alpha, size=(n, 1, 1)).astype(np.float32)
            out_x = lam * X + (1.0 - lam) * partner_x
            out_y = np.maximum(y, partner_y)
            return out_x.astype(np.float32), out_y

        if method == "cutmix":
            out_x = X.copy()
            t = X.shape[1]
            cut_len = max(1, int(t * self.cutmix_ratio))
            starts = rng.randint(0, max(1, t - cut_len + 1), size=n)
            for i in range(n):
                s = int(starts[i])
                e = s + cut_len
                out_x[i, s:e, :] = partner_x[i, s:e, :]
            out_y = np.maximum(y, partner_y)
            return out_x.astype(np.float32), out_y

        if method == "smote":
            lam = rng.uniform(0.0, 1.0, size=(n, 1, 1)).astype(np.float32)
            out_x = X + lam * (partner_x - X)
            out_y = np.maximum(y, partner_y)
            return out_x.astype(np.float32), out_y

        out_x = self._augment_windows_only(X, method, rng)
        return out_x.astype(np.float32), y

    def _select_augment_mask(self, y_2d):
        targets = self._parse_target_list(self.augment_target)
        if not targets:
            print(
                f"[Augment] Unknown augment_target='{self.augment_target}', "
                "skip augmentation."
            )
            return np.zeros((len(y_2d),), dtype=bool)

        multilabel_alias = {"multilabel", "multi_label", "multi-label", "multi"}
        class_map = {str(name).strip().lower(): idx for idx, name in enumerate(self.classes)}
        mask = np.zeros((len(y_2d),), dtype=bool)
        unknown_targets = []

        for target in targets:
            target_key = target.lower()
            if target_key in multilabel_alias:
                mask = np.logical_or(mask, np.sum(y_2d, axis=1) > 1)
                continue
            class_idx = class_map.get(target_key)
            if class_idx is None:
                unknown_targets.append(target)
                continue
            mask = np.logical_or(mask, y_2d[:, class_idx] == 1)

        if unknown_targets:
            print(
                f"[Augment] Unknown augment_target(s) ignored: {unknown_targets}. "
                f"raw='{self.augment_target}'."
            )
        return mask

    def _augment_train_data(self, X, y_2d):
        if self.augment_count <= 0:
            return X, y_2d
        if len(X) == 0:
            return X, y_2d
        if not self.augment_methods:
            return X, y_2d

        mask = self._select_augment_mask(y_2d)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            print(
                f"[Augment] No samples matched target='{self.augment_target}', "
                "skip augmentation."
            )
            return X, y_2d

        base_x = np.asarray(X[idx], dtype=np.float32)
        base_y = np.asarray(y_2d[idx], dtype=y_2d.dtype)

        rng = np.random.RandomState(self.random_state)
        aug_x_parts = []
        aug_y_parts = []
        rep = int(self.augment_count)
        for rep_idx in range(rep):
            method = self.augment_methods[rep_idx % len(self.augment_methods)]
            if method in {"mixup", "cutmix", "smote"}:
                aug_x_rep, aug_y_rep = self._augment_pairwise(base_x, base_y, method, rng)
            else:
                aug_x_rep = self._augment_windows_only(base_x, method, rng)
                aug_y_rep = base_y
            aug_x_parts.append(np.asarray(aug_x_rep, dtype=np.float32))
            aug_y_parts.append(np.asarray(aug_y_rep, dtype=base_y.dtype))

        aug_x = np.concatenate(aug_x_parts, axis=0)
        aug_y = np.concatenate(aug_y_parts, axis=0)

        out_x = np.concatenate([X.astype(np.float32, copy=False), aug_x], axis=0)
        out_y = np.concatenate([y_2d, aug_y], axis=0)
        print(
            f"[Augment] methods={','.join(self.augment_methods)}, target={self.augment_target}, per_sample={rep}, "
            f"matched={len(base_x)}, added={len(aug_x)}, total={len(out_x)}"
        )
        return out_x, out_y

    def train(self, train_data, val_data):
        if len(train_data) >= 3:
            train_X, train_y, train_ratio = train_data[:3]
        else:
            train_X, train_y = train_data[:2]
            train_ratio = np.asarray(train_y, dtype=np.float32)

        if len(val_data) >= 3:
            val_X, val_y, val_ratio = val_data[:3]
        else:
            val_X, val_y = val_data[:2]
            val_ratio = np.asarray(val_y, dtype=np.float32)

        if self.use_val_in_train:
            base_X = np.concatenate([train_X, val_X], axis=0)
            base_y = np.concatenate([train_y, val_y], axis=0)
            base_ratio = np.concatenate([np.asarray(train_ratio), np.asarray(val_ratio)], axis=0)
            sampled_X, sampled_y = self._subsample_train_data(base_X, base_y)
            sampled_ratio = self._subsample_train_data(base_X, base_ratio)[1]
            sampled_X = self._maybe_downsample_windows(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            ratio_fit_2d = np.asarray(sampled_ratio, dtype=np.float32)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            if ratio_fit_2d.ndim == 1:
                ratio_fit_2d = ratio_fit_2d.reshape(-1, 1)
            split_name = "merged train+val"
            del base_X, base_y, base_ratio, sampled_y, sampled_ratio
        else:
            sampled_X, sampled_y = self._subsample_train_data(train_X, train_y)
            sampled_ratio = self._subsample_train_data(train_X, np.asarray(train_ratio))[1]
            sampled_X = self._maybe_downsample_windows(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            ratio_fit_2d = np.asarray(sampled_ratio, dtype=np.float32)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            if ratio_fit_2d.ndim == 1:
                ratio_fit_2d = ratio_fit_2d.reshape(-1, 1)
            split_name = "train only"

        if sampled_X.shape[0] == 0:
            raise ValueError("No training samples available after subsampling.")

        sampled_X, y_fit_2d = self._augment_train_data(sampled_X, y_fit_2d)
        if ratio_fit_2d.shape[1] != y_fit_2d.shape[1]:
            ratio_fit_2d = y_fit_2d.astype(np.float32, copy=False)
        elif ratio_fit_2d.shape[0] < y_fit_2d.shape[0]:
            extra = y_fit_2d[ratio_fit_2d.shape[0]:].astype(np.float32, copy=False)
            ratio_fit_2d = np.concatenate([ratio_fit_2d, extra], axis=0)
        elif ratio_fit_2d.shape[0] > y_fit_2d.shape[0]:
            ratio_fit_2d = ratio_fit_2d[: y_fit_2d.shape[0]]

        base_feat = None
        if not self.signal_combo_map:
            base_feat = self._extract_features_batched(sampled_X)

        print(
            f"Start training multi-label XGBoost with {split_name}: "
            f"{sampled_X.shape[0]} samples, "
            f"{y_fit_2d.shape[1]} labels | "
            f"feature_domain={self.feature_domain}, spectrum_method={self.spectrum_method}, "
            f"use_tsfresh={self.use_tsfresh}..."
        )

        self.models = []
        for label_idx in range(y_fit_2d.shape[1]):
            train_idx, y_real, aug_neg_x, y_aug, balance_info = self._select_ovr_train_subset(
                sampled_X,
                y_fit_2d,
                ratio_fit_2d,
                label_idx=label_idx,
            )
            print(
                f"[OvR] {balance_info['label']}: pos={balance_info['pos_count_after_threshold']}, "
                f"neg_target={balance_info['target_neg_count']}, "
                f"neg_real={balance_info['selected_neg_real']}, neg_aug={balance_info['selected_neg_augmented']}"
            )
            if balance_info.get("bucket_plan"):
                plan_parts = []
                for bucket_name, bucket_stat in balance_info["bucket_plan"].items():
                    plan_parts.append(
                        f"{bucket_name} tgt={bucket_stat['target']} real={bucket_stat['real']} aug={bucket_stat['aug']}"
                    )
                print(f"[OvR-Buckets] {balance_info['label']}: " + " | ".join(plan_parts))

            model = XGBClassifier(**self._model_params)
            if base_feat is not None:
                fit_feat_real = base_feat[train_idx]
            else:
                label_X_real = self._select_windows_for_label(sampled_X[train_idx], label_idx)
                fit_feat_real = self._extract_features_batched(label_X_real)

            y_train_bin = y_real
            if len(aug_neg_x) > 0:
                if base_feat is not None:
                    fit_feat_aug = self._extract_features_batched(aug_neg_x)
                else:
                    label_X_aug = self._select_windows_for_label(aug_neg_x, label_idx)
                    fit_feat_aug = self._extract_features_batched(label_X_aug)
                fit_feat = np.concatenate([fit_feat_real, fit_feat_aug], axis=0)
                y_train_bin = np.concatenate([y_real, y_aug], axis=0)
            else:
                fit_feat = fit_feat_real

            if len(np.unique(y_train_bin)) < 2:
                print(
                    f"[OvR] {self.classes[label_idx]} has single class after filtering/balancing. "
                    f"Falling back to unbalanced full pool for this label."
                )
                if base_feat is not None:
                    fit_feat = base_feat
                else:
                    label_X_all = self._select_windows_for_label(sampled_X, label_idx)
                    fit_feat = self._extract_features_batched(label_X_all)
                y_train_bin = y_fit_2d[:, label_idx]

            model.fit(fit_feat, y_train_bin, verbose=False)
            self.models.append(model)

        del sampled_X, base_feat, y_fit_2d, ratio_fit_2d, train_X, train_y, val_X, val_y, train_ratio, val_ratio
        gc.collect()
        print("Training done.")

    def _predict_proba_single(self, test_X):
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        base_feat = None
        if not self.signal_combo_map:
            base_feat = self._extract_features(test_X)

        probs = []
        for label_idx, model in enumerate(self.models):
            if base_feat is not None:
                feat = base_feat
            else:
                label_X = self._select_windows_for_label(test_X, label_idx)
                feat = self._extract_features(label_X)
            per_label_prob = model.predict_proba(feat)
            probs.append(per_label_prob[:, 1])
        return np.column_stack(probs)

    def _build_tta_views(self, test_X):
        views = [np.asarray(test_X, dtype=np.float32)]
        if self.tta_count <= 0 or not self.tta_methods:
            return views
        rng = np.random.RandomState(self.random_state + 123)
        for i in range(int(self.tta_count)):
            method = self.tta_methods[i % len(self.tta_methods)]
            if method in {"mixup", "cutmix", "smote"}:
                aug_x, _ = self._augment_pairwise(views[0], np.zeros((len(views[0]), 1), dtype=np.int8), method, rng)
                views.append(np.asarray(aug_x, dtype=np.float32))
            else:
                views.append(np.asarray(self._augment_windows_only(views[0], method, rng), dtype=np.float32))
        return views

    def predict(self, test_X):
        probs = self.predict_proba(test_X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, test_X):
        test_X = self._maybe_downsample_windows(test_X)
        views = self._build_tta_views(test_X)
        probs_list = [self._predict_proba_single(view) for view in views]
        if len(probs_list) == 1:
            return probs_list[0]
        return np.mean(np.stack(probs_list, axis=0), axis=0)

    def get_feature_importance(self, importance_type="gain"):
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        per_label_scores = {}
        for label_idx, model in enumerate(self.models):
            booster = model.get_booster()
            per_label_scores[self.classes[label_idx]] = booster.get_score(importance_type=importance_type)
        return per_label_scores
