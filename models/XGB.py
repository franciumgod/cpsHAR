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

    def _maybe_downsample_windows(self, X):
        method = self.pre_downsample_method
        if method in {"false", "none", "0"}:
            return np.asarray(X, dtype=np.float32)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D input for downsampling, but got shape {X.shape}")

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
                raise ValueError(
                    f"Window length {X.shape[1]} is smaller than downsample window {win}."
                )
            windows = sliding_window_view(X, window_shape=win, axis=1)
            windows = windows[:, ::step, :, :]
            X_ds = windows.mean(axis=-1, dtype=np.float32)
        else:
            raise ValueError(
                f"Unsupported pre_downsample_method='{self.pre_downsample_method}'. "
                "Expected one of: false, interval, sliding_window."
            )

        if transposed_input:
            X_ds = np.transpose(X_ds, (0, 2, 1))
        return X_ds.astype(np.float32, copy=False)

    def _extract_features(self, X):
        """
        Input:
            X: shape = (n_samples, time_steps, n_channels)
               or (n_samples, n_channels, time_steps)

        Output:
            shape = (n_samples, n_features_2d)
        """
        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 3:
            raise ValueError(f"Expected 3D input, but got shape {X.shape}")

        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        mean = np.mean(X, axis=1, dtype=np.float32)
        std = np.std(X, axis=1, dtype=np.float32)
        max_val = np.max(X, axis=1)
        min_val = np.min(X, axis=1)

        features = np.hstack([mean, std, max_val, min_val]).astype(np.float32)
        return features

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
            raise ValueError(
                f"Unsupported subsample_method='{self.subsample_method}'. "
                f"Expected one of: false, random, interval."
            )
        return X[idx], y[idx]

    def train(self, train_data, val_data):
        train_X, train_y = train_data
        val_X, val_y = val_data

        if self.use_val_in_train:
            base_X = np.concatenate([train_X, val_X], axis=0)
            base_y = np.concatenate([train_y, val_y], axis=0)
            sampled_X, sampled_y = self._subsample_train_data(base_X, base_y)
            sampled_X = self._maybe_downsample_windows(sampled_X)
            X_fit_feat = self._extract_features_batched(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "merged train+val"
            del base_X, base_y, sampled_X, sampled_y
        else:
            sampled_X, sampled_y = self._subsample_train_data(train_X, train_y)
            sampled_X = self._maybe_downsample_windows(sampled_X)
            X_fit_feat = self._extract_features_batched(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "train only"

        if X_fit_feat.shape[0] == 0:
            raise ValueError("No training samples available after subsampling.")

        print(
            f"Start training multi-label XGBoost with {split_name}: "
            f"{X_fit_feat.shape[0]} samples, {X_fit_feat.shape[1]} features, "
            f"{y_fit_2d.shape[1]} labels..."
        )

        self.models = []
        for label_idx in range(y_fit_2d.shape[1]):
            model = XGBClassifier(**self._model_params)
            model.fit(X_fit_feat, y_fit_2d[:, label_idx], verbose=False)
            self.models.append(model)

        del X_fit_feat, y_fit_2d, train_X, train_y, val_X, val_y
        gc.collect()
        print("Training done.")

    def predict(self, test_X):
        test_X = self._maybe_downsample_windows(test_X)
        X_features = self._extract_features(test_X)
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        per_label_preds = [model.predict(X_features) for model in self.models]
        predictions = np.column_stack(per_label_preds).astype(int)
        return predictions

    def predict_proba(self, test_X):
        test_X = self._maybe_downsample_windows(test_X)
        X_features = self._extract_features(test_X)
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        probs = []
        for model in self.models:
            per_label_prob = model.predict_proba(X_features)
            probs.append(per_label_prob[:, 1])
        return np.column_stack(probs)

    def get_feature_importance(self, importance_type="gain"):
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        per_label_scores = {}
        for label_idx, model in enumerate(self.models):
            booster = model.get_booster()
            per_label_scores[self.classes[label_idx]] = booster.get_score(importance_type=importance_type)
        return per_label_scores
