import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestClassifierSK:
    def __init__(self, classes):
        self.classes = classes
        self.model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        self._n_train_data_samples = 10000

    def _extract_features(self, X):
        mean = np.mean(X, axis=1)
        std = np.std(X, axis=1)
        max_val = np.max(X, axis=1)
        min_val = np.min(X, axis=1)
        squared = np.square(X)
        rms = np.sqrt(np.mean(squared, axis=1))


        return np.hstack([mean, std, max_val, min_val, rms])

    def train(self, train_data, val_data):

        # RandomForest doesn't need a separate validation set for hyperparameter tuning -> use it in training as well
        full_train_set = np.concatenate([train_data[0], val_data[0]], axis=0)
        full_train_labels = np.concatenate([train_data[1], val_data[1]], axis=0)
        train_data = (full_train_set, full_train_labels)

        # we train on a sample of training data since time windows overlap
        n_samples = min(self._n_train_data_samples, len(full_train_set))
        train_data_sample_idx = np.random.choice(len(full_train_set), n_samples, replace=False)

        X_train = train_data[0][train_data_sample_idx]
        y_train = train_data[1][train_data_sample_idx]


        # Feature Engineering (3D -> 2D)
        X_features = self._extract_features(X_train)


        print(f"Start training with {X_features.shape[0]} Samples and {X_features.shape[1]} Features...")
        self.model.fit(X_features, y_train)
        print("Training done.")

    def predict(self, test_X):
        X_features = self._extract_features(test_X)
        predictions = self.model.predict(X_features)
        return predictions