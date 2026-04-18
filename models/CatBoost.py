from catboost import CatBoostClassifier

from models.mainline_feature_base import MainlineFeatureOVRBase


class CatBoostClassifierSK(MainlineFeatureOVRBase):
    def __init__(
        self,
        classes,
        *,
        use_val_in_train=False,
        use_gpu=False,
        random_state=42,
        subsample_method="random",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
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
        super().__init__(
            classes,
            model_display_name="CatBoost",
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
            "iterations": int(n_estimators),
            "learning_rate": float(learning_rate),
            "depth": int(max_depth),
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": -1,
            "random_seed": int(random_state),
            "task_type": "GPU" if use_gpu else "CPU",
            "rsm": float(colsample_bytree),
            "l2_leaf_reg": float(reg_lambda),
        }
        if 0.0 < float(subsample) < 1.0:
            self._model_params["bootstrap_type"] = "Bernoulli"
            self._model_params["subsample"] = float(subsample)

    def _build_estimator(self, label_idx):
        return CatBoostClassifier(**self._model_params)

    def _fit_estimator(self, estimator, fit_feat, y_train_bin, label_idx):
        estimator.fit(fit_feat, y_train_bin, verbose=False)
        return estimator
