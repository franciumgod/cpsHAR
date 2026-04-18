import copy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from tabm import TabM

from models.mainline_feature_base import MainlineFeatureOVRBase


class _TabMBinaryEstimator:
    def __init__(
        self,
        *,
        random_state=42,
        device="cpu",
        max_epochs=40,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=8,
        validation_fraction=0.15,
        arch_type="tabm",
        k=32,
        d_block=512,
        n_blocks=3,
        dropout=0.1,
    ):
        self.random_state = int(random_state)
        self.device = str(device)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.arch_type = str(arch_type)
        self.k = int(k)
        self.d_block = int(d_block)
        self.n_blocks = int(n_blocks)
        self.dropout = float(dropout)

        self.scaler_ = None
        self.model_ = None
        self.device_ = None

    def _resolve_device(self):
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_model(self, n_features):
        return TabM.make(
            n_num_features=int(n_features),
            d_out=1,
            k=self.k,
            arch_type=self.arch_type,
            d_block=self.d_block,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        )

    @staticmethod
    def _sigmoid_average(logits):
        if logits.ndim == 3:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
        if probs.ndim == 2:
            probs = probs.mean(dim=1)
        return probs

    def _make_loader(self, X, y=None, shuffle=False):
        x_tensor = torch.from_numpy(X.astype(np.float32, copy=False))
        if y is None:
            dataset = TensorDataset(x_tensor)
        else:
            y_tensor = torch.from_numpy(y.astype(np.float32, copy=False))
            dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        self.scaler_ = StandardScaler()
        X_all = self.scaler_.fit_transform(X).astype(np.float32, copy=False)

        stratify = y if len(np.unique(y)) > 1 and np.min(np.bincount(y.astype(np.int64))) >= 2 else None
        use_val = self.validation_fraction > 0.0 and len(X_all) >= 20 and len(np.unique(y)) > 1
        if use_val:
            X_train, X_val, y_train, y_val = train_test_split(
                X_all,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify,
            )
        else:
            X_train, y_train = X_all, y
            X_val, y_val = None, None

        self.device_ = self._resolve_device()
        torch.manual_seed(self.random_state)
        self.model_ = self._build_model(X_train.shape[1]).to(self.device_)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False) if X_val is not None else None

        best_state = copy.deepcopy(self.model_.state_dict())
        best_val_loss = float("inf")
        bad_epochs = 0

        for _ in range(self.max_epochs):
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(x_batch).squeeze(-1)
                target = y_batch[:, None].expand(-1, logits.shape[1])
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

            if val_loader is None:
                best_state = copy.deepcopy(self.model_.state_dict())
                continue

            self.model_.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)
                    logits = self.model_(x_batch).squeeze(-1)
                    target = y_batch[:, None].expand(-1, logits.shape[1])
                    val_loss = criterion(logits, target)
                    val_losses.append(float(val_loss.detach().cpu()))

            mean_val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
            if mean_val_loss + 1e-6 < best_val_loss:
                best_val_loss = mean_val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_.transform(X).astype(np.float32, copy=False)
        loader = self._make_loader(X_scaled, y=None, shuffle=False)

        probs = []
        self.model_.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device_)
                logits = self.model_(x_batch)
                batch_prob = self._sigmoid_average(logits).detach().cpu().numpy()
                probs.append(batch_prob.astype(np.float32, copy=False))

        pos = np.concatenate(probs, axis=0) if probs else np.empty((0,), dtype=np.float32)
        neg = 1.0 - pos
        return np.column_stack([neg, pos]).astype(np.float32, copy=False)


class TabMClassifierSK(MainlineFeatureOVRBase):
    def __init__(
        self,
        classes,
        *,
        use_val_in_train=False,
        use_gpu=False,
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
        max_epochs=40,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=8,
        validation_fraction=0.15,
        arch_type="tabm",
        k=32,
        d_block=512,
        n_blocks=3,
        dropout=0.1,
    ):
        super().__init__(
            classes,
            model_display_name="TabM",
            use_val_in_train=use_val_in_train,
            random_state=random_state,
            subsample_method=subsample_method,
            n_train_data_samples=n_train_data_samples,
            feature_batch_size=feature_batch_size,
            pre_downsample_method=pre_downsample_method,
            pre_downsample_factor=pre_downsample_factor,
            pre_downsample_window_size=pre_downsample_window_size,
            pre_downsample_window_step=pre_downsample_window_step,
            signal_combo_map=signal_combo_map,
            sensor_cols=sensor_cols,
            feature_engineering=feature_engineering,
            feature_domain=feature_domain,
            spectrum_method=spectrum_method,
        )
        self._model_params = {
            "random_state": int(random_state),
            "device": "cuda" if use_gpu else "cpu",
            "max_epochs": int(max_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "patience": int(patience),
            "validation_fraction": float(validation_fraction),
            "arch_type": str(arch_type),
            "k": int(k),
            "d_block": int(d_block),
            "n_blocks": int(n_blocks),
            "dropout": float(dropout),
        }

    def _build_estimator(self, label_idx):
        return _TabMBinaryEstimator(**self._model_params)
