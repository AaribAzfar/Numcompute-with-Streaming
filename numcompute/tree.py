import numpy as np


# ======================================================================
# Impurity Functions
# ======================================================================

def _gini(y):

    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def _entropy(y):

    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # Avoid log(0) by only computing where prob > 0
    return -np.sum(probs * np.log2(probs + 1e-12))


# ======================================================================
# Tree Node
# ======================================================================

class _Node:

    def __init__(self):
        # Split info (internal nodes)
        self.feature   = None   # which column to split on
        self.threshold = None   # split value: go left if X[feature] <= threshold

        # Children
        self.left  = None       # _Node for samples <= threshold
        self.right = None       # _Node for samples >  threshold

        # Leaf info
        self.is_leaf    = False
        self.prediction = None  # majority class at this leaf


# ======================================================================
# Decision Tree Classifier
# ======================================================================

class DecisionTreeClassifier:

    def __init__(self, max_depth=5, min_samples_split=2,
                 criterion='gini', max_features=None):
        self.max_depth        = max_depth
        self.min_samples_split = min_samples_split
        self.criterion        = criterion
        self.max_features     = max_features

        self._root      = None   # the root _Node after fitting
        self._classes   = None   # all class labels seen during fit
        self._X_buffer  = None   # accumulated data for partial_fit
        self._y_buffer  = None

    # ------------------------------------------------------------------ #
    # Impurity selector
    # ------------------------------------------------------------------ #

    def _impurity(self, y):
        if self.criterion == 'gini':
            return _gini(y)
        elif self.criterion == 'entropy':
            return _entropy(y)
        else:
            raise ValueError("criterion must be 'gini' or 'entropy'")

    # ------------------------------------------------------------------ #
    # Feature subset (for Random Forest)
    # ------------------------------------------------------------------ #

    def _n_features_to_try(self, n_features):
        
        mf = self.max_features
        if mf is None:
            return n_features
        elif mf == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        elif mf == 'log2':
            return max(1, int(np.log2(n_features)))
        elif isinstance(mf, float):
            return max(1, int(mf * n_features))
        elif isinstance(mf, int):
            return max(1, min(mf, n_features))
        else:
            raise ValueError("max_features must be None, 'sqrt', 'log2', int, or float")

    # ------------------------------------------------------------------ #
    # Finding the best split
    # ------------------------------------------------------------------ #

    def _best_split(self, X, y):

        n_samples, n_features = X.shape
        best_gain      = -1.0
        best_feature   = None
        best_threshold = None

        parent_impurity = self._impurity(y)

        # Randomly pick which features to try (key for Random Forest)
        k = self._n_features_to_try(n_features)
        feature_indices = np.random.choice(n_features, size=k, replace=False)

        for feature in feature_indices:
            # Each unique value in this column is a candidate threshold
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # Skip if one side is empty
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Weighted impurity after the split
                n = len(y)
                weighted_impurity = (
                    len(y_left)  / n * self._impurity(y_left) +
                    len(y_right) / n * self._impurity(y_right)
                )

                gain = parent_impurity - weighted_impurity

                if gain > best_gain:
                    best_gain      = gain
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ------------------------------------------------------------------ #
    # Tree building (recursive)
    # ------------------------------------------------------------------ #

    def _build(self, X, y, depth):

        node = _Node()

        # --- Leaf conditions ---
        pure      = len(np.unique(y)) == 1
        too_small = len(y) < self.min_samples_split
        too_deep  = depth >= self.max_depth

        if pure or too_small or too_deep:
            node.is_leaf    = True
            node.prediction = self._majority_class(y)
            return node

        # --- Find best split ---
        feature, threshold = self._best_split(X, y)

        # If no valid split was found, make a leaf
        if feature is None:
            node.is_leaf    = True
            node.prediction = self._majority_class(y)
            return node

        # --- Split and recurse ---
        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        node.feature   = feature
        node.threshold = threshold
        node.left      = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right     = self._build(X[right_mask], y[right_mask], depth + 1)

        return node

    def _majority_class(self, y):
        """Return the most common label in y."""
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    # ------------------------------------------------------------------ #
    # Predicting one sample
    # ------------------------------------------------------------------ #

    def _predict_one(self, x, node):
        """Walk the tree for a single sample x, return leaf prediction."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
        
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._classes  = np.unique(y)
        self._X_buffer = X.copy()
        self._y_buffer = y.copy()
        self._root     = self._build(X, y, depth=0)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        # Grow the buffer
        if self._X_buffer is None:
            self._X_buffer = X_chunk
            self._y_buffer = y_chunk
        else:
            self._X_buffer = np.vstack([self._X_buffer, X_chunk])
            self._y_buffer = np.concatenate([self._y_buffer, y_chunk])

        # Rebuild tree on all data seen so far
        self._classes = np.unique(self._y_buffer)
        self._root    = self._build(self._X_buffer, self._y_buffer, depth=0)

        return self

    def predict(self, X):
        if self._root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() or partial_fit() first.")

        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self._root) for x in X])

    def predict_proba(self, X):
        
        preds  = self.predict(X)
        n      = len(preds)
        n_cls  = len(self._classes)
        proba  = np.zeros((n, n_cls))

        for i, cls in enumerate(self._classes):
            proba[preds == cls, i] = 1.0

        return proba

    def __repr__(self):
        return (
            f"DecisionTreeClassifier("
            f"max_depth={self.max_depth}, "
            f"criterion='{self.criterion}', "
            f"max_features={self.max_features})"
        )


# ======================================================================
# Random Forest Classifier
# ======================================================================

class RandomForestClassifier:
    
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 criterion='gini', max_features='sqrt', bootstrap=True):
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.criterion         = criterion
        self.max_features      = max_features
        self.bootstrap         = bootstrap

        self._trees   = []     # list of fitted DecisionTreeClassifiers
        self._classes = None   # all unique classes seen

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _make_tree(self):
       
        return DecisionTreeClassifier(
            max_depth         = self.max_depth,
            min_samples_split = self.min_samples_split,
            criterion         = self.criterion,
            max_features      = self.max_features,
        )

    def _bootstrap_sample(self, X, y):
       
        n = len(y)
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices]

    def _majority_vote(self, all_preds):
        
        n_samples = all_preds.shape[1]
        votes = np.empty(n_samples, dtype=all_preds.dtype)

        for i in range(n_samples):
            col = all_preds[:, i]
            classes, counts = np.unique(col, return_counts=True)
            votes[i] = classes[np.argmax(counts)]

        return votes

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self, X, y):
       
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self._classes  = np.unique(y)
        self._trees    = []

        for _ in range(self.n_estimators):
            tree = self._make_tree()

            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            tree.fit(X_sample, y_sample)
            self._trees.append(tree)

        return self

    def partial_fit(self, X_chunk, y_chunk):
        
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        self._classes = np.unique(
            np.concatenate([self._classes, y_chunk])
        ) if self._classes is not None else np.unique(y_chunk)

        # If no trees yet, create them
        if not self._trees:
            self._trees = [self._make_tree() for _ in range(self.n_estimators)]

        for tree in self._trees:
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X_chunk, y_chunk)
            else:
                X_sample, y_sample = X_chunk, y_chunk

            tree.partial_fit(X_sample, y_sample)

        return self

    def predict(self, X):

        if not self._trees:
            raise ValueError("Forest has not been fitted yet. Call fit() or partial_fit() first.")

        X = np.asarray(X, dtype=float)

        # Collect each tree's predictions: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self._trees])

        return self._majority_vote(all_preds)

    def predict_proba(self, X):
        
        if not self._trees:
            raise ValueError("Forest has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        n_classes = len(self._classes)

        # Average probability from each tree
        proba_sum = np.zeros((len(X), n_classes))

        for tree in self._trees:
            proba_sum += tree.predict_proba(X)

        return proba_sum / self.n_estimators

    def __repr__(self):
        return (
            f"RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"criterion='{self.criterion}', "
            f"max_features={self.max_features})"
        )