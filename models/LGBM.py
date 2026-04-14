import gc
import numpy as np
from lightgbm import LGBMClassifier
from numpy.lib._stride_tricks_impl import sliding_window_view


class LightGBMClassifierSK:
    def __init__(
        self,
        classes,
        use_val_in_train=False,
        subsample_method="random",
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        objective="binary",
        boosting_type="lgbt",
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        drop_rate=0.1,
        max_drop=None,
        skip_drop=None,
        n_train_data_samples=100_000,
        feature_batch_size=2000,
        min_child_samples=30,
        min_data_in_leaf=10,
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
        feature_engineering=False,
    ):
        self.classes = classes
        self.use_val_in_train = use_val_in_train
        self.subsample_method = str(subsample_method).lower() if subsample_method is not None else "false"
        self.random_state = random_state
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
        normalized_boosting_type = str(boosting_type).lower()
        if normalized_boosting_type == "lgbt":
            normalized_boosting_type = "gbdt"

        self._model_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "random_state": random_state,
            "reg_lambda": reg_lambda,
            "objective": objective,
            "n_jobs": -1,
            "verbose":-1,
            # key params
            "boosting_type": normalized_boosting_type,
            "max_depth": max_depth,
            "num_leaves": num_leaves,

            # regularization params
            "subsample": subsample,
            "min_child_samples": min_child_samples,
            "min_data_in_leaf": min_data_in_leaf,
        }
        if normalized_boosting_type == "dart":
            self._model_params["drop_rate"] = drop_rate
            if max_drop is not None:
                self._model_params["max_drop"] = max_drop
            if skip_drop is not None:
                self._model_params["skip_drop"] = skip_drop

        self.models = []
        self._estimator_cls = LGBMClassifier

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

    def _extract_features(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            return np.asarray(X, dtype=np.float32)

        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

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

    def _apply_rotation_3d(self, X, rng):
        X_rot = np.asarray(X, dtype=np.float32).copy()
        n, _, c = X_rot.shape
        if c < 3:
            return X_rot

        angles = rng.uniform(-15.0, 15.0, size=(n, 3)).astype(np.float32) * (np.pi / 180.0)
        for i in range(n):
            ax, ay, az = angles[i]
            cx, sx = np.cos(ax), np.sin(ax)
            cy, sy = np.cos(ay), np.sin(ay)
            cz, sz = np.cos(az), np.sin(az)
            rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
            ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
            rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
            r = rz @ ry @ rx

            X_rot[i, :, :3] = X_rot[i, :, :3] @ r.T
            if c >= 6:
                X_rot[i, :, 3:6] = X_rot[i, :, 3:6] @ r.T
        return X_rot

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
            return self._apply_rotation_3d(X, rng)

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
        target = self.augment_target.strip().lower()
        if target in {"multilabel", "multi_label", "multi-label", "multi"}:
            return np.sum(y_2d, axis=1) > 1

        class_map = {str(name).strip().lower(): idx for idx, name in enumerate(self.classes)}
        class_idx = class_map.get(target)
        if class_idx is None:
            print(
                f"[Augment] Unknown augment_target='{self.augment_target}', "
                "skip augmentation."
            )
            return np.zeros((len(y_2d),), dtype=bool)
        return y_2d[:, class_idx] == 1

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
        train_X, train_y = train_data
        val_X, val_y = val_data

        if self.use_val_in_train:
            base_X = np.concatenate([train_X, val_X], axis=0)
            base_y = np.concatenate([train_y, val_y], axis=0)
            sampled_X, sampled_y = self._subsample_train_data(base_X, base_y)
            sampled_X = self._maybe_downsample_windows(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "merged train+val"
            del base_X, base_y, sampled_y
        else:
            sampled_X, sampled_y = self._subsample_train_data(train_X, train_y)
            sampled_X = self._maybe_downsample_windows(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "train only"

        sampled_X, y_fit_2d = self._augment_train_data(sampled_X, y_fit_2d)

        base_feat = None
        if not self.signal_combo_map:
            base_feat = self._extract_features_batched(sampled_X)

        feature_dim = base_feat.shape[1] if base_feat is not None else "dynamic"
        print(
            f"Start training multi-label LightGBM with {split_name}: "
            f"{sampled_X.shape[0]} samples, {feature_dim} features, "
            f"{y_fit_2d.shape[1]} labels..."
        )

        self.models = []
        for label_idx in range(y_fit_2d.shape[1]):
            model = self._estimator_cls(**self._model_params)
            if base_feat is not None:
                fit_feat = base_feat
            else:
                label_X = self._select_windows_for_label(sampled_X, label_idx)
                fit_feat = self._extract_features_batched(label_X)
            model.fit(fit_feat, y_fit_2d[:, label_idx])
            self.models.append(model)

        del sampled_X, base_feat, y_fit_2d, train_X, train_y, val_X, val_y
        gc.collect()
        print("Training done.")

    def _predict_proba_single(self, test_X):
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
