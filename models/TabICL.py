import sys
import types

from models.mainline_feature_base import MainlineFeatureOVRBase


def _import_tabicl_classifier_for_classification():
    # TabICL's classification stack imports the regression-only quantile module,
    # which drags numba into import time. In cpsHAR we only use classification,
    # so we stub that module to keep import latency practical.
    module_name = "tabicl.model.quantile_dist"
    if module_name not in sys.modules:
        stub = types.ModuleType(module_name)

        class QuantileToDistribution:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "QuantileToDistribution is unavailable in the cpsHAR TabICL shim. "
                    "This shim is classification-only."
                )

        stub.QuantileToDistribution = QuantileToDistribution
        sys.modules[module_name] = stub

    from tabicl.sklearn.classifier import TabICLClassifier

    return TabICLClassifier


class TabICLClassifierSK(MainlineFeatureOVRBase):
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
        n_estimators=8,
        batch_size=8,
        kv_cache=False,
        model_path=None,
        allow_auto_download=False,
        checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
        device=None,
        verbose=False,
    ):
        super().__init__(
            classes,
            model_display_name="TabICL",
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
        resolved_device = device if device is not None else ("cuda" if use_gpu else "cpu")
        self._model_params = {
            "n_estimators": int(n_estimators),
            "batch_size": int(batch_size) if batch_size is not None else None,
            "kv_cache": kv_cache,
            "model_path": model_path,
            "allow_auto_download": bool(allow_auto_download),
            "checkpoint_version": str(checkpoint_version),
            "device": resolved_device,
            "random_state": int(random_state),
            "verbose": bool(verbose),
        }

    def _build_estimator(self, label_idx):
        TabICLClassifier = _import_tabicl_classifier_for_classification()

        return TabICLClassifier(**self._model_params)

    def _fit_estimator(self, estimator, fit_feat, y_train_bin, label_idx):
        try:
            estimator.fit(fit_feat, y_train_bin)
            return estimator
        except Exception as exc:
            raise RuntimeError(
                f"TabICL fit failed for label '{self.classes[label_idx]}'. "
                f"If the TabICL checkpoint is not cached locally, provide --tabicl_model_path "
                f"or enable --tabicl_allow_auto_download True in a network-enabled session. "
                f"Original error: {exc}"
            ) from exc
