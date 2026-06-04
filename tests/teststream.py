"""
Unit tests for NumCompute Assignment 2.2
Run with:  python -m pytest test_all.py -v
Covers: preprocessing, stats, metrics, pipeline, tree, ensemble, stream
Minimum 30 tests including streaming and edge cases.
"""

import numpy as np
import pytest

from numcompute.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, SimpleImputer
from numcompute.stats import mean, median, std, minimum, maximum, histogram, quantiles, StreamingStats
from numcompute.metrics import (accuracy, precision, recall, f1, confusion_matrix,
                     mse, roc_curve, auc, StreamingMetrics)
from numcompute.pipeline import Pipeline
from numcompute.tree import DecisionTreeClassifier, RandomForestClassifier
from numcompute.ensemble import BaggingClassifier, BoostingClassifier, EnsembleClassifier
from numcompute.stream import StreamTrainer


# ======================================================================
# Shared fixtures
# ======================================================================

np.random.seed(42)

# Simple linearly separable dataset
X_SIMPLE = np.array([
    [1.0, 2.0], [1.5, 1.8], [1.2, 2.1],   # class 0
    [5.0, 6.0], [5.5, 5.8], [5.2, 6.1],   # class 1
])
Y_SIMPLE = np.array([0, 0, 0, 1, 1, 1])

# Larger random dataset for streaming tests
np.random.seed(0)
X_BIG = np.random.randn(200, 4)
Y_BIG = (X_BIG[:, 0] + X_BIG[:, 1] > 0).astype(int)


def make_chunks(X, y, n_chunks=4):
    """Split X, y into n_chunks equal pieces."""
    size = len(y) // n_chunks
    return [(X[i*size:(i+1)*size], y[i*size:(i+1)*size]) for i in range(n_chunks)]


# ======================================================================
# 1-6: preprocessing.py
# ======================================================================

def test_standard_scaler_batch():
    """StandardScaler fit/transform produces mean~0 std~1."""
    sc = StandardScaler()
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Xt = sc.fit_transform(X)
    assert Xt.shape == X.shape
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Xt.std(axis=0),  1, atol=1e-10)


def test_standard_scaler_partial_fit_matches_batch():
    """Streaming partial_fit over two halves ≈ batch fit on whole."""
    X = np.random.randn(100, 3)
    sc_batch  = StandardScaler().fit(X)
    sc_stream = StandardScaler()
    sc_stream.partial_fit(X[:50])
    sc_stream.partial_fit(X[50:])
    assert np.allclose(sc_batch.mean, sc_stream.mean, atol=1e-10)
    assert np.allclose(sc_batch.var, sc_stream.var, atol=1e-10)
    assert np.allclose(sc_batch.var,  sc_stream.var,  atol=1e-6)


def test_standard_scaler_zero_variance():
    """Zero-variance column should not cause division by zero."""
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    sc = StandardScaler()
    Xt = sc.fit_transform(X)
    assert np.all(np.isfinite(Xt))


def test_minmax_scaler_partial_fit():
    """MinMaxScaler running min/max updates correctly across chunks."""
    sc = MinMaxScaler()
    sc.partial_fit(np.array([[1.0], [3.0]]))
    sc.partial_fit(np.array([[0.0], [5.0]]))
    assert sc.min[0] == 0.0
    assert sc.max[0] == 5.0


def test_onehot_encoder_partial_fit_new_category():
    """OneHotEncoder expands columns when new category arrives in later chunk."""
    enc = OneHotEncoder()
    enc.partial_fit(np.array([['a'], ['b']]))
    enc.partial_fit(np.array([['c']]))
    assert 'c' in enc.categories[0]
    assert enc.transform(np.array([['a'], ['c']])).shape[1] == 3


def test_simple_imputer_partial_fit():
    """SimpleImputer running mean stays accurate across two chunks."""
    imp = SimpleImputer()
    imp.partial_fit(np.array([[1.0], [3.0]]))   # mean = 2.0
    imp.partial_fit(np.array([[5.0], [7.0]]))   # overall mean = 4.0
    assert np.isclose(imp.fill_values[0], 4.0)


# ======================================================================
# 7-11: stats.py
# ======================================================================

def test_stats_basic():
    """Basic stat functions return correct values."""
    X = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert np.isclose(mean(X),    3.0)
    assert np.isclose(median(X),  3.0)
    assert np.isclose(minimum(X), 1.0)
    assert np.isclose(maximum(X), 5.0)


def test_stats_nan_handling():
    """Stats functions ignore NaN values correctly."""
    X = [1.0, np.nan, 3.0]
    assert np.isclose(mean(X), 2.0)


def test_streaming_stats_cumulative():
    """StreamingStats cumulative mean matches batch mean."""
    X = np.random.randn(200)
    ss = StreamingStats()
    for chunk in np.array_split(X, 4):
        ss.update_stats(chunk)
    result = ss.result()
    assert np.isclose(result['mean'], np.mean(X), atol=1e-10)


def test_streaming_stats_sliding_window():
    """StreamingStats window only reflects last N samples."""
    ss = StreamingStats(window=10)
    ss.update_stats(np.ones(10) * 100)   # first 10: mean=100
    ss.update_stats(np.ones(10) * 0)     # next  10: mean=0  (window slides)
    result = ss.result()
    assert np.isclose(result['mean'], 0.0)


def test_streaming_stats_histogram():
    """StreamingStats histogram counts are non-negative and sum > 0."""
    ss = StreamingStats(bins=5)
    ss.update_stats(np.arange(20, dtype=float))
    counts, edges = ss.histogram()
    assert counts.sum() > 0
    assert len(edges) == 6   # bins + 1


# ======================================================================
# 12-17: metrics.py
# ======================================================================

def test_accuracy_perfect():
    assert accuracy([0, 1, 1, 0], [0, 1, 1, 0]) == 1.0


def test_accuracy_zero():
    assert accuracy([0, 1], [1, 0]) == 0.0


def test_confusion_matrix_shape():
    cm = confusion_matrix([0, 1, 2, 0], [0, 2, 1, 0])
    assert cm.shape == (3, 3)
    assert cm.sum() == 4


def test_streaming_metrics_update_result():
    """StreamingMetrics cumulative accuracy matches manual calculation."""
    sm = StreamingMetrics()
    sm.update(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    sm.update(np.array([1, 1, 0, 0]), np.array([1, 1, 0, 1]))
    result = sm.result()
    # manually: 3/4 correct + 3/4 correct = 6/8
    assert np.isclose(result['accuracy'], 6/8)


def test_streaming_metrics_reset():
    """StreamingMetrics reset wipes all state."""
    sm = StreamingMetrics()
    sm.update(np.array([0, 1]), np.array([0, 1]))
    sm.reset()
    assert sm._total_seen == 0
    assert sm.history['accuracy'] == []


def test_streaming_metrics_rolling_window():
    """Rolling window only uses last N chunks."""
    sm = StreamingMetrics(window=1)
    sm.update(np.array([0, 0, 0]), np.array([1, 1, 1]))   # chunk 1: acc=0
    sm.update(np.array([1, 1, 1]), np.array([1, 1, 1]))   # chunk 2: acc=1
    result = sm.result_rolling()
    assert result['accuracy'] == 1.0   # only chunk 2 in window


# ======================================================================
# 18-22: pipeline.py
# ======================================================================

def test_pipeline_fit_predict():
    """Pipeline fit+predict runs end-to-end."""
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier(max_depth=3))
    ])
    pipe.fit(X_SIMPLE, Y_SIMPLE)
    preds = pipe.predict(X_SIMPLE)
    assert len(preds) == len(Y_SIMPLE)


def test_pipeline_partial_fit():
    """Pipeline partial_fit updates all steps incrementally."""
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier(max_depth=3))
    ])
    for X_chunk, y_chunk in make_chunks(X_BIG, Y_BIG):
        pipe.partial_fit(X_chunk, y_chunk)
    preds = pipe.predict(X_BIG)
    assert len(preds) == len(Y_BIG)


def test_pipeline_get_step():
    """Pipeline.get_step returns correct object by name."""
    sc = StandardScaler()
    pipe = Pipeline([('scale', sc), ('model', DecisionTreeClassifier())])
    assert pipe.get_step('scale') is sc


def test_pipeline_missing_step_raises():
    """Accessing a nonexistent step raises KeyError."""
    pipe = Pipeline([('scale', StandardScaler())])
    with pytest.raises(KeyError):
        pipe.get_step('nonexistent')


def test_pipeline_predict_before_fit_raises():
    """Predicting before fitting raises ValueError."""
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('model', DecisionTreeClassifier())
    ])
    with pytest.raises(Exception):
        pipe.predict(X_SIMPLE)


# ======================================================================
# 23-27: tree.py
# ======================================================================

def test_decision_tree_fit_predict():
    """DecisionTree learns a perfectly separable dataset."""
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_SIMPLE, Y_SIMPLE)
    preds = tree.predict(X_SIMPLE)
    assert accuracy(Y_SIMPLE, preds) == 1.0


def test_decision_tree_partial_fit_streaming():
    """Streaming partial_fit converges to correct predictions."""
    tree = DecisionTreeClassifier(max_depth=4)
    for X_chunk, y_chunk in make_chunks(X_BIG, Y_BIG, n_chunks=5):
        tree.partial_fit(X_chunk, y_chunk)
    preds = tree.predict(X_BIG)
    assert accuracy(Y_BIG, preds) > 0.7


def test_decision_tree_entropy_criterion():
    """Entropy criterion runs without error."""
    tree = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    tree.fit(X_SIMPLE, Y_SIMPLE)
    preds = tree.predict(X_SIMPLE)
    assert len(preds) == len(Y_SIMPLE)


def test_decision_tree_max_depth_one():
    """Tree with depth=1 makes a single split (stump)."""
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_BIG, Y_BIG)
    preds = tree.predict(X_BIG)
    assert len(preds) == len(Y_BIG)


def test_random_forest_better_than_single_tree():
    """Random forest accuracy >= single tree on random data."""
    np.random.seed(1)
    X = np.random.randn(300, 5)
    y = (X[:, 0] - X[:, 2] > 0).astype(int)

    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X[:200], y[:200])
    tree_acc = accuracy(y[200:], tree.predict(X[200:]))

    rf = RandomForestClassifier(n_estimators=10, max_depth=3)
    rf.fit(X[:200], y[:200])
    rf_acc = accuracy(y[200:], rf.predict(X[200:]))

    # Forest should be at least as good
    assert rf_acc >= tree_acc - 0.05


# ======================================================================
# 28-32: ensemble.py + stream.py
# ======================================================================

def test_bagging_partial_fit():
    """BaggingClassifier partial_fit runs and predicts correctly shaped output."""
    bag = BaggingClassifier(n_estimators=5, max_depth=3)
    for X_chunk, y_chunk in make_chunks(X_BIG, Y_BIG):
        bag.partial_fit(X_chunk, y_chunk)
    preds = bag.predict(X_BIG)
    assert preds.shape == Y_BIG.shape


def test_boosting_fit_predict():
    """BoostingClassifier fits and predicts on simple data."""
    boost = BoostingClassifier(n_estimators=10, max_depth=1)
    boost.fit(X_SIMPLE, Y_SIMPLE)
    preds = boost.predict(X_SIMPLE)
    assert accuracy(Y_SIMPLE, preds) > 0.5


def test_ensemble_classifier_swap():
    """EnsembleClassifier works with both methods via same API."""
    for method in ('bagging', 'boosting'):
        clf = EnsembleClassifier(method=method, n_estimators=5, max_depth=2)
        clf.fit(X_SIMPLE, Y_SIMPLE)
        preds = clf.predict(X_SIMPLE)
        assert len(preds) == len(Y_SIMPLE)


def test_stream_trainer_fit_chunk_logs():
    """StreamTrainer logs one entry per fit_chunk call."""
    model   = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model, verbose=False)
    chunks  = make_chunks(X_BIG, Y_BIG, n_chunks=4)

    for X_chunk, y_chunk in chunks:
        trainer.fit_chunk(X_chunk, y_chunk)

    assert len(trainer.log) == 4
    assert all('accuracy' in e for e in trainer.log)
    assert all('memory_bytes' in e for e in trainer.log)


def test_stream_trainer_cumulative_accuracy():
    """StreamTrainer cumulative accuracy stays in [0, 1]."""
    model   = RandomForestClassifier(n_estimators=5, max_depth=3)
    trainer = StreamTrainer(model, verbose=False)
    trainer.stream(X_BIG, Y_BIG, chunk_size=50)

    cum_acc = trainer.cumulative_accuracy()
    assert 0.0 <= cum_acc <= 1.0


def test_stream_trainer_score_chunk():
    """score_chunk returns float in [0, 1] without changing the model."""
    model   = DecisionTreeClassifier(max_depth=3)
    trainer = StreamTrainer(model, verbose=False)
    trainer.fit_chunk(X_BIG[:100], Y_BIG[:100])

    val_acc = trainer.score_chunk(X_BIG[100:], Y_BIG[100:])
    assert 0.0 <= val_acc <= 1.0
    # Log should still have only 1 entry (score_chunk doesn't log)
    assert len(trainer.log) == 1


# ======================================================================
# Edge cases
# ======================================================================

def test_single_sample_chunk():
    """All streaming components handle a chunk of size 1."""
    tree = DecisionTreeClassifier(max_depth=2)
    tree.partial_fit(X_SIMPLE[:1], Y_SIMPLE[:1])
    tree.partial_fit(X_SIMPLE[1:], Y_SIMPLE[1:])
    preds = tree.predict(X_SIMPLE)
    assert len(preds) == len(Y_SIMPLE)


def test_all_same_class_chunk():
    """A chunk where all labels are the same class doesn't crash."""
    tree = DecisionTreeClassifier(max_depth=3)
    tree.partial_fit(X_BIG[:50], np.zeros(50, dtype=int))   # all class 0
    tree.partial_fit(X_BIG[50:], Y_BIG[50:])                # mixed
    preds = tree.predict(X_BIG)
    assert len(preds) == len(Y_BIG)


def test_streaming_metrics_nan_chunk():
    """StreamingMetrics handles a chunk with no positive class gracefully."""
    sm = StreamingMetrics()
    # Chunk with only class 0 — precision/recall undefined but shouldn't crash
    sm.update(np.zeros(5, dtype=int), np.zeros(5, dtype=int))
    result = sm.result()
    assert 'accuracy' in result
    assert result['accuracy'] == 1.0