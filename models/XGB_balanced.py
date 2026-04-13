from models.XGB import XGBoostClassifierSK
from utils.balanced_sampling_ext import (
    format_balanced_sampling_report,
    select_balanced_multilabel_subset,
)


class XGBoostClassifierBalancedSK(XGBoostClassifierSK):
    def __init__(
        self,
        classes,
        balance_keep_zero_ratio=True,
        balance_verbose=True,
        **kwargs,
    ):
        super().__init__(classes, **kwargs)
        self.balance_keep_zero_ratio = balance_keep_zero_ratio
        self.balance_verbose = balance_verbose
        self.last_balance_report = None

    def _subsample_train_data(self, X, y):
        if self.subsample_method != "balanced":
            self.last_balance_report = None
            return super()._subsample_train_data(X, y)

        selected_indices, report = select_balanced_multilabel_subset(
            y=y,
            n_samples=self._n_train_data_samples,
            class_names=self.classes,
            random_state=self.random_state,
            keep_zero_ratio=self.balance_keep_zero_ratio,
        )
        self.last_balance_report = report

        if self.balance_verbose:
            print(
                format_balanced_sampling_report(
                    report,
                    class_names=self.classes,
                    title="Balanced train subsampling",
                )
            )

        return X[selected_indices], y[selected_indices]
