import numpy as np
from numcompute.tree import DecisionTreeClassifier


# ======================================================================
# Bagging Classifier
# ======================================================================

class BaggingClassifier:

    def __init__(self, n_estimators=10, max_depth=5,
                 min_samples_split=2, criterion='gini'):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.criterion         = criterion

        self._trees   = []
        self._classes = None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _make_tree(self):
        
        return DecisionTreeClassifier(
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            criterion         = self.criterion,
            max_features      = 'sqrt',   # random feature subsets per split
        )

    def _bootstrap(self, X, y):
        
        n = len(y)
        idx = np.random.choice(n, size=n, replace=True)
        return X[idx], y[idx]

    def _majority_vote(self, all_preds):
    
        n_samples = all_preds.shape[1]
        votes = np.empty(n_samples, dtype=all_preds.dtype)

        for i in range(n_samples):
            classes, counts = np.unique(all_preds[:, i], return_counts=True)
            votes[i] = classes[np.argmax(counts)]

        return votes

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
    
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._classes = np.unique(y)
        self._trees   = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y)
            tree = self._make_tree()
            tree.fit(X_sample, y_sample)
            self._trees.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk):
    
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        # Update known classes
        if self._classes is None:
            self._classes = np.unique(y_chunk)
        else:
            self._classes = np.unique(np.concatenate([self._classes, y_chunk]))

        # Create trees on first chunk
        if not self._trees:
            self._trees = [self._make_tree() for _ in range(self.n_estimators)]

        # Each tree gets its own bootstrap sample of this chunk
        for tree in self._trees:
            X_sample, y_sample = self._bootstrap(X_chunk, y_chunk)
            tree.partial_fit(X_sample, y_sample)

        return self

    def predict(self, X):
    
        if not self._trees:
            raise ValueError("BaggingClassifier has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        all_preds = np.array([tree.predict(X) for tree in self._trees])
        return self._majority_vote(all_preds)

    def predict_proba(self, X):
    
        if not self._trees:
            raise ValueError("BaggingClassifier has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        n_classes = len(self._classes)
        proba_sum = np.zeros((len(X), n_classes))

        for tree in self._trees:
            preds = tree.predict(X)
            for i, cls in enumerate(self._classes):
                proba_sum[preds == cls, i] += 1.0

        return proba_sum / self.n_estimators

    def __repr__(self):
        return (
            f"BaggingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth})"
        )


# ======================================================================
# Boosting Classifier  (AdaBoost)
# ======================================================================

class BoostingClassifier:

    def __init__(self, n_estimators=50, max_depth=1, learning_rate=1.0):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate

        self._trees        = []    # trained trees
        self._tree_weights = []    # how much each tree's vote counts
        self._classes      = None

    # ------------------------------------------------------------------ #
    # AdaBoost training logic
    # ------------------------------------------------------------------ #

    def _adaboost_round(self, X, y, sample_weights):

        # Step 1: Train tree using weighted sampling
        # Since our tree doesn't natively accept sample weights,
        # we simulate it by sampling rows according to their weights
        n = len(y)
        idx = np.random.choice(n, size=n, replace=True, p=sample_weights)
        X_weighted, y_weighted = X[idx], y[idx]

        tree = DecisionTreeClassifier(max_depth=self.max_depth)
        tree.fit(X_weighted, y_weighted)

        # Step 2: Compute weighted error on full dataset
        preds = tree.predict(X)
        wrong = (preds != y).astype(float)
        weighted_error = np.dot(sample_weights, wrong)

        # Clip to avoid log(0) or division by zero
        weighted_error = np.clip(weighted_error, 1e-10, 1 - 1e-10)

        # Step 3: Alpha — how trustworthy is this tree?
        # Higher alpha = tree made fewer mistakes
        alpha = self.learning_rate * 0.5 * np.log(
            (1.0 - weighted_error) / weighted_error
        )

        # Step 4: Update sample weights
        # Misclassified samples get multiplied by e^alpha (boosted)
        # Correct samples get multiplied by e^-alpha (shrunk)
        new_weights = sample_weights * np.exp(alpha * (2 * wrong - 1))

        # Step 5: Renormalise so weights sum to 1
        new_weights /= new_weights.sum()

        return tree, alpha, new_weights

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._classes      = np.unique(y)
        self._trees        = []
        self._tree_weights = []

        n = len(y)
        # Start with uniform weights — every sample equally important
        weights = np.ones(n) / n

        for _ in range(self.n_estimators):
            tree, alpha, weights = self._adaboost_round(X, y, weights)
            self._trees.append(tree)
            self._tree_weights.append(alpha)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        # Update known classes
        if self._classes is None:
            self._classes = np.unique(y_chunk)
        else:
            self._classes = np.unique(np.concatenate([self._classes, y_chunk]))

        n = len(y_chunk)
        weights = np.ones(n) / n   # uniform weights for this chunk

        # Add one new tree per partial_fit call
        tree, alpha, _ = self._adaboost_round(X_chunk, y_chunk, weights)
        self._trees.append(tree)
        self._tree_weights.append(alpha)

        return self

    def predict(self, X):
        if not self._trees:
            raise ValueError("BoostingClassifier has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        n_samples  = len(X)
        n_classes  = len(self._classes)

        # Accumulate weighted votes per class
        class_scores = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self._trees, self._tree_weights):
            preds = tree.predict(X)
            for i, cls in enumerate(self._classes):
                class_scores[preds == cls, i] += alpha

        # Pick the class with the highest total weighted vote
        best_class_idx = np.argmax(class_scores, axis=1)
        return self._classes[best_class_idx]

    def predict_proba(self, X):
       
        if not self._trees:
            raise ValueError("BoostingClassifier has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        n_samples = len(X)
        n_classes = len(self._classes)

        class_scores = np.zeros((n_samples, n_classes))

        for tree, alpha in zip(self._trees, self._tree_weights):
            preds = tree.predict(X)
            for i, cls in enumerate(self._classes):
                class_scores[preds == cls, i] += alpha

        # Softmax to turn scores into probabilities
        exp_scores = np.exp(class_scores - class_scores.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def __repr__(self):
        return (
            f"BoostingClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"learning_rate={self.learning_rate})"
        )


# ======================================================================
# EnsembleClassifier  — unified interface
# ======================================================================

class EnsembleClassifier:

    def __init__(self, method='bagging', **kwargs):
        if method == 'bagging':
            self._model = BaggingClassifier(**kwargs)
        elif method == 'boosting':
            self._model = BoostingClassifier(**kwargs)
        else:
            raise ValueError("method must be 'bagging' or 'boosting'")

        self.method = method

    def fit(self, X, y):
        """Fit the ensemble on the full dataset."""
        self._model.fit(X, y)
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Incrementally update the ensemble on a new chunk."""
        self._model.partial_fit(X_chunk, y_chunk)
        return self

    def predict(self, X):
        """Predict class labels."""
        return self._model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities."""
        return self._model.predict_proba(X)

    def __repr__(self):
        return f"EnsembleClassifier(method='{self.method}', model={self._model})"