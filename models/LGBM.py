import gc
import numpy as np
from lightgbm import LGBMClassifier


class LightGBMClassifierSK:
    def __init__(
        self,
        classes,
        use_val_in_train=False,
        subsample_method="random",
        random_state=42,
        n_estimators=1000,
        learning_rate=0.05,
        objective="multiclass",
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
        n_train_data_samples=100000,
        feature_batch_size=2000,
        min_child_samples=30,
        min_data_in_leaf=10,
    ):
        self.classes = classes
        self.use_val_in_train = use_val_in_train
        self.subsample_method = str(subsample_method).lower() if subsample_method is not None else "false"
        self.random_state = random_state
        self._n_train_data_samples = n_train_data_samples
        self._feature_batch_size = feature_batch_size
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

    def _extract_features(self, X):
        X = np.asarray(X, dtype=np.float32)

        if X.shape[1] < X.shape[2]:
            X = np.transpose(X, (0, 2, 1))

        mean = np.mean(X, axis=1, dtype=np.float32)
        std = np.std(X, axis=1, dtype=np.float32)
        max_val = np.max(X, axis=1)
        min_val = np.min(X, axis=1)

        return np.hstack([mean, std, max_val, min_val]).astype(np.float32)

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
        return X[idx], y[idx]

    def train(self, train_data, val_data):
        train_X, train_y = train_data
        val_X, val_y = val_data

        if self.use_val_in_train:
            base_X = np.concatenate([train_X, val_X], axis=0)
            base_y = np.concatenate([train_y, val_y], axis=0)
            sampled_X, sampled_y = self._subsample_train_data(base_X, base_y)
            X_fit_feat = self._extract_features_batched(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "merged train+val"
            del base_X, base_y, sampled_X, sampled_y
        else:
            sampled_X, sampled_y = self._subsample_train_data(train_X, train_y)
            X_fit_feat = self._extract_features_batched(sampled_X)
            y_fit_2d = np.asarray(sampled_y)
            if y_fit_2d.ndim == 1:
                y_fit_2d = y_fit_2d.reshape(-1, 1)
            split_name = "train only"

        print(
            f"Start training multi-label LightGBM with {split_name}: "
            f"{X_fit_feat.shape[0]} samples, {X_fit_feat.shape[1]} features, "
            f"{y_fit_2d.shape[1]} labels..."
        )

        self.models = []
        for label_idx in range(y_fit_2d.shape[1]):
            model = self._estimator_cls(**self._model_params)
            model.fit(X_fit_feat, y_fit_2d[:, label_idx])
            self.models.append(model)

        del X_fit_feat, y_fit_2d, train_X, train_y, val_X, val_y
        gc.collect()
        print("Training done.")

    def predict(self, test_X):
        X_features = self._extract_features(test_X)

        per_label_preds = [model.predict(X_features) for model in self.models]
        return np.column_stack(per_label_preds).astype(int)

    def predict_proba(self, test_X):
        X_features = self._extract_features(test_X)

        probs = []
        for model in self.models:
            per_label_prob = model.predict_proba(X_features)
            probs.append(per_label_prob[:, 1])
        return np.column_stack(probs)
