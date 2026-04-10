import numpy as np
from xgboost import XGBClassifier
import gc


class XGBoostClassifierSK:
    def __init__(
        self,
        classes,
        use_val_in_train=False,
        use_gpu=False,
        random_state=42,
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
        feature_batch_size=20000,
    ):
        self.classes = classes
        self.use_val_in_train = use_val_in_train
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self._n_train_data_samples = n_train_data_samples
        self._feature_batch_size = feature_batch_size

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

    def _prepare_training_data(self, sub_train_X, sub_train_y, val_X, val_y):
        y_train_2d = np.asarray(sub_train_y)
        if y_train_2d.ndim == 1:
            y_train_2d = y_train_2d.reshape(-1, 1)

        train_feat = self._extract_features_batched(sub_train_X)

        if self.use_val_in_train:
            y_val_2d = np.asarray(val_y)
            if y_val_2d.ndim == 1:
                y_val_2d = y_val_2d.reshape(-1, 1)
            val_feat = self._extract_features_batched(val_X)
            X_fit_feat = np.vstack([train_feat, val_feat])
            y_fit_2d = np.vstack([y_train_2d, y_val_2d])
            del val_feat, y_val_2d
        else:
            X_fit_feat = train_feat
            y_fit_2d = y_train_2d

        del train_feat
        gc.collect()
        return X_fit_feat, y_fit_2d

    def _subsample_train_data(self, X, y):
        n_samples = min(self._n_train_data_samples, len(X))
        idx = np.random.RandomState(self.random_state).choice(len(X), n_samples, replace=False)
        return X[idx], y[idx]

    def train(self, train_data, val_data):
        train_X, train_y = train_data
        val_X, val_y = val_data

        sub_train_X, sub_train_y = self._subsample_train_data(train_X, train_y)

        X_fit_feat, y_fit_2d = self._prepare_training_data(sub_train_X, sub_train_y, val_X, val_y)
        split_name = "merged train+val" if self.use_val_in_train else "train only"
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

        del X_fit_feat, y_fit_2d, sub_train_X, sub_train_y, train_X, train_y, val_X, val_y
        gc.collect()
        print("Training done.")

    def predict(self, test_X):
        X_features = self._extract_features(test_X)
        if not self.models:
            raise RuntimeError("Model has not been trained yet.")

        per_label_preds = [model.predict(X_features) for model in self.models]
        predictions = np.column_stack(per_label_preds).astype(int)
        return predictions

    def predict_proba(self, test_X):
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
