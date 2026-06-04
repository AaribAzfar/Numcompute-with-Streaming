import numpy as np
from collections import deque

# Validation Helper
def _validate_inputs(y_true, y_pred=None):
    """
    Validate inputs for metrics.

    - Ensures arrays are numeric
    - Ensures same shape for y_true and y_pred
    """
    if y_true is None:
        raise ValueError("y_true cannot be None")

    y_true = np.asarray(y_true)

    if not np.issubdtype(y_true.dtype, np.number):
        raise ValueError("y_true must be numeric")

    if y_pred is not None:
        y_pred = np.asarray(y_pred)

        if not np.issubdtype(y_pred.dtype, np.number):
            raise ValueError("y_pred must be numeric")

        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

    return y_true, y_pred


# For Classification Metrics 

# Accuracy
def accuracy(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(y_true == y_pred)


# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))  # all unique labels
    n = len(classes)

    cm = np.zeros((n, n), dtype=int)

    # Rows = true labels, cols = predicted labels
    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = np.sum((y_true == c1) & (y_pred == c2))

    return cm


# Precision
def precision(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        raise ValueError("No positive predictions; precision undefined")

    return tp / (tp + fp)


# Recall
def recall(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # Assumes binary classification (0 and 1)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        raise ValueError("No actual positives; recall undefined")

    return tp / (tp + fn)


# F1 Score
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        raise ValueError("Precision and Recall are zero; F1 undefined")

    return 2 * p * r / (p + r)


# Regression Metrics

def mse(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


# Root Mean Squared Error (RMSE)
def rmse(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# Mean Absolute Error (MAD / MAE)
def mad(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


# Mean Absolute Percentage Error (MAPE)
def mape(y_true, y_pred):
    y_true, y_pred = _validate_inputs(y_true, y_pred)

    # avoid division by zero
    non_zero = y_true != 0

    if not np.any(non_zero):
        raise ValueError("All y_true values are zero; MAPE undefined")

    y_true = y_true[non_zero]
    y_pred = y_pred[non_zero]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# For ROC Curve (Binary) 

def roc_curve(y_true, y_scores):
    
    #y_scores: probability scores (not labels)
    
    if y_true is None or y_scores is None:
        raise ValueError("Inputs cannot be None")

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    if y_true.shape != y_scores.shape:
        raise ValueError("y_true and y_scores must have the same shape")

    # EnsureS binary labels
    unique = np.unique(y_true)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("y_true must contain only binary labels (0 and 1)")

    # Sort by scores descending
    desc = np.argsort(-y_scores)
    y_true = y_true[desc]

    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)

    tp_total = tp[-1]
    fp_total = fp[-1]

    if tp_total == 0 or fp_total == 0:
        raise ValueError("ROC undefined: need both positive and negative samples")

    tpr = tp / tp_total
    fpr = fp / fp_total

    return fpr, tpr


# For AUC 

def auc(fpr, tpr):

    # Trapezoidal rule
    
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    if fpr.shape != tpr.shape:
        raise ValueError("fpr and tpr must have the same shape")

    if len(fpr) < 2:
        raise ValueError("At least two points required to compute AUC")

    return np.trapezoid(tpr, fpr)

class StreamingMetrics:

    def __init__(self, window=None):
        self.window = window
        self.reset()
 
    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
 
    def reset(self):
        
        # Cumulative counts for accuracy
        self._total_correct = 0
        self._total_seen = 0
 
        # Cumulative TP/FP/FN for binary precision/recall/f1
        self._cum_tp = 0
        self._cum_fp = 0
        self._cum_fn = 0
 
        # Cumulative confusion matrix
        # Stored as a dict {(true_class, pred_class): count} so it can
        # grow when new classes appear in later chunks.
        self._cm_counts = {}
        self._cm_classes = np.array([], dtype=int)
 
        # AUC: accumulate all scores and labels seen so far
        self._all_y_true = []
        self._all_y_scores = []
 
        # Per-chunk history (for plotting via visualise.py)
        self.history = {
            "accuracy":  [],
            "precision": [],
            "recall":    [],
            "f1":        [],
            "auc":       [],
        }
 
        # Rolling window: store (y_true_chunk, y_pred_chunk) deques
        if self.window is not None:
            self._window_true  = deque(maxlen=self.window)
            self._window_pred  = deque(maxlen=self.window)
            self._window_scores = deque(maxlen=self.window)
        else:
            self._window_true  = None
            self._window_pred  = None
            self._window_scores = None
 
        return self
 
    def update(self, y_true_chunk, y_pred_chunk, y_scores_chunk=None):
       
        y_true_chunk, y_pred_chunk = _validate_inputs(y_true_chunk, y_pred_chunk)
 
        # ---- Accuracy accumulation ------------------------------------
        self._total_correct += np.sum(y_true_chunk == y_pred_chunk)
        self._total_seen    += len(y_true_chunk)
 
        # ---- Binary TP/FP/FN accumulation ----------------------------
        self._cum_tp += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 1)))
        self._cum_fp += int(np.sum((y_true_chunk == 0) & (y_pred_chunk == 1)))
        self._cum_fn += int(np.sum((y_true_chunk == 1) & (y_pred_chunk == 0)))
 
        # ---- Confusion matrix accumulation ---------------------------
        self._update_confusion_matrix(y_true_chunk, y_pred_chunk)
 
        # ---- AUC score accumulation ----------------------------------
        if y_scores_chunk is not None:
            y_scores_chunk = np.asarray(y_scores_chunk)
            if y_scores_chunk.shape != y_true_chunk.shape:
                raise ValueError("y_scores_chunk must have the same shape as y_true_chunk")
            self._all_y_true.append(y_true_chunk)
            self._all_y_scores.append(y_scores_chunk)
 
        # ---- Rolling window ------------------------------------------
        if self.window is not None:
            self._window_true.append(y_true_chunk)
            self._window_pred.append(y_pred_chunk)
            if y_scores_chunk is not None:
                self._window_scores.append(y_scores_chunk)
 
        # ---- Per-chunk history for plotting --------------------------
        chunk_acc = float(np.mean(y_true_chunk == y_pred_chunk))
        self.history["accuracy"].append(chunk_acc)
 
        # Only record precision/recall/f1 if binary labels present
        binary_labels = set(np.unique(y_true_chunk)).issubset({0, 1})
        if binary_labels:
            try:
                self.history["precision"].append(precision(y_true_chunk, y_pred_chunk))
                self.history["recall"].append(recall(y_true_chunk, y_pred_chunk))
                self.history["f1"].append(f1(y_true_chunk, y_pred_chunk))
            except ValueError:
                # Edge chunk with no positives etc. — record NaN
                self.history["precision"].append(float("nan"))
                self.history["recall"].append(float("nan"))
                self.history["f1"].append(float("nan"))
 
        if y_scores_chunk is not None and binary_labels:
            try:
                fpr_, tpr_ = roc_curve(y_true_chunk, y_scores_chunk)
                self.history["auc"].append(float(auc(fpr_, tpr_)))
            except ValueError:
                self.history["auc"].append(float("nan"))
 
        return self
 
    def result(self):
        
        if self._total_seen == 0:
            raise ValueError("No data has been seen yet. Call update() first.")
 
        out = {
            "n_samples": self._total_seen,
            "accuracy":  self._total_correct / self._total_seen,
        }
 
        # Precision / recall / f1 from cumulative TP/FP/FN
        tp, fp, fn = self._cum_tp, self._cum_fp, self._cum_fn
 
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        out["recall"]    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
 
        p_val = out["precision"]
        r_val = out["recall"]
        if not (np.isnan(p_val) or np.isnan(r_val)) and (p_val + r_val) > 0:
            out["f1"] = 2 * p_val * r_val / (p_val + r_val)
        else:
            out["f1"] = float("nan")
 
        # Confusion matrix as ndarray
        out["confusion_matrix"] = self._build_confusion_matrix()
 
        # Cumulative AUC over all accumulated scores
        if self._all_y_true:
            all_true   = np.concatenate(self._all_y_true)
            all_scores = np.concatenate(self._all_y_scores)
            try:
                fpr_, tpr_ = roc_curve(all_true, all_scores)
                out["auc"] = float(auc(fpr_, tpr_))
            except ValueError:
                out["auc"] = float("nan")
 
        return out
 
    def result_rolling(self):
        
        if self.window is None:
            raise ValueError("No window set. Pass window=N to StreamingMetrics().")
 
        if not self._window_true:
            raise ValueError("No data in rolling window yet.")
 
        y_true_all = np.concatenate(list(self._window_true))
        y_pred_all = np.concatenate(list(self._window_pred))
 
        out = {
            "window":    self.window,
            "n_samples": len(y_true_all),
            "accuracy":  float(np.mean(y_true_all == y_pred_all)),
        }
 
        tp = int(np.sum((y_true_all == 1) & (y_pred_all == 1)))
        fp = int(np.sum((y_true_all == 0) & (y_pred_all == 1)))
        fn = int(np.sum((y_true_all == 1) & (y_pred_all == 0)))
 
        out["precision"] = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        out["recall"]    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
 
        p_val = out["precision"]
        r_val = out["recall"]
        if not (np.isnan(p_val) or np.isnan(r_val)) and (p_val + r_val) > 0:
            out["f1"] = 2 * p_val * r_val / (p_val + r_val)
        else:
            out["f1"] = float("nan")
 
        out["confusion_matrix"] = confusion_matrix(y_true_all, y_pred_all)
 
        if self._window_scores:
            all_scores = np.concatenate(list(self._window_scores))
            try:
                fpr_, tpr_ = roc_curve(y_true_all, all_scores)
                out["auc"] = float(auc(fpr_, tpr_))
            except ValueError:
                out["auc"] = float("nan")
 
        return out
 
    # ------------------------------------------------------------------ #
    # Confusion matrix helpers
    # ------------------------------------------------------------------ #
 
    def _update_confusion_matrix(self, y_true, y_pred):
    
        chunk_classes = np.unique(np.concatenate([y_true, y_pred]))
 
        # Expand known class list
        self._cm_classes = np.unique(
            np.concatenate([self._cm_classes, chunk_classes])
        ).astype(int)
 
        # Accumulate counts
        for c_true in chunk_classes:
            for c_pred in chunk_classes:
                key = (int(c_true), int(c_pred))
                count = int(np.sum((y_true == c_true) & (y_pred == c_pred)))
                self._cm_counts[key] = self._cm_counts.get(key, 0) + count
 
    def _build_confusion_matrix(self):
       
        classes = self._cm_classes
        n = len(classes)
        cm = np.zeros((n, n), dtype=int)
 
        for i, c_true in enumerate(classes):
            for j, c_pred in enumerate(classes):
                cm[i, j] = self._cm_counts.get((int(c_true), int(c_pred)), 0)
 
        return cm
 
    def confusion_matrix_result(self):
        
        if self._total_seen == 0:
            raise ValueError("No data seen yet.")
        return self._build_confusion_matrix(), self._cm_classes.copy()
 
    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #
 
    def __repr__(self):
        return (
            f"StreamingMetrics(window={self.window}, "
            f"chunks_seen={len(self.history['accuracy'])}, "
            f"n_samples={self._total_seen})"
        )

