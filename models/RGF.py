from rgf.sklearn import RGFClassifier

from models.mainline_feature_base import MainlineFeatureOVRBase


class RGFClassifierSK(MainlineFeatureOVRBase):
    def __init__(
        self,
        classes,
        *,
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
        max_leaf=1000,
        algorithm="RGF",
        reg_depth=1.0,
        l2=0.1,
        learning_rate=0.5,
        min_samples_leaf=10,
    ):
        super().__init__(
            classes,
            model_display_name="RGF",
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
            "max_leaf": int(max_leaf),
            "algorithm": str(algorithm),
            "loss": "Log",
            "reg_depth": float(reg_depth),
            "l2": float(l2),
            "learning_rate": float(learning_rate),
            "min_samples_leaf": int(min_samples_leaf),
            "n_jobs": -1,
            "verbose": 0,
        }

    def _build_estimator(self, label_idx):
        return RGFClassifier(**self._model_params)
