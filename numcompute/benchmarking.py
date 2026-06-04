"""
benchmark.py
============
Compares single DecisionTree vs RandomForest vs Bagging vs Boosting
under streaming conditions.

Metrics tracked per model:
    - Per-chunk accuracy
    - Cumulative accuracy
    - Time per chunk (seconds)
    - Memory footprint (KB)

Run:  python benchmark.py
"""

import numpy as np
import time

from tree import DecisionTreeClassifier, RandomForestClassifier
from ensemble import BaggingClassifier, BoostingClassifier
from stream import StreamTrainer
from visualise import compare_models, plot_metric_over_time


# ======================================================================
# Dataset
# ======================================================================

def make_dataset(n=500, n_features=6, seed=42):
    """
    Generate a synthetic binary classification dataset.
    Classes are separated by the sum of the first two features.
    """
    np.random.seed(seed)
    X = np.random.randn(n, n_features)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)
    return X, y


# ======================================================================
# Benchmark runner
# ======================================================================

def run_benchmark(chunk_size=50):
    X, y = make_dataset(n=500)

    models = {
        "DecisionTree":   DecisionTreeClassifier(max_depth=4),
        "RandomForest":   RandomForestClassifier(n_estimators=10, max_depth=4),
        "Bagging":        BaggingClassifier(n_estimators=10, max_depth=4),
        "Boosting":       BoostingClassifier(n_estimators=10, max_depth=1),
    }

    trainers = {
        name: StreamTrainer(model, verbose=False)
        for name, model in models.items()
    }

    # ---- Stream each model through the same chunks -------------------
    for name, trainer in trainers.items():
        trainer.stream(X, y, chunk_size=chunk_size)

    # ---- Print summary table -----------------------------------------
    print("\n" + "=" * 65)
    print(f"{'Model':<18}  {'Cum Acc':>8}  {'Avg Time(s)':>11}  {'Final Mem(KB)':>13}")
    print("-" * 65)

    for name, trainer in trainers.items():
        cum_acc   = trainer.cumulative_accuracy()
        avg_time  = np.mean(trainer.time_history())
        final_mem = trainer.memory_history()[-1]
        print(f"{name:<18}  {cum_acc:>8.4f}  {avg_time:>11.4f}  {final_mem:>13.1f}")

    print("=" * 65)

    # ---- Plots -------------------------------------------------------
    # 1. Accuracy comparison: Tree vs Forest
    compare_models(
        trainers["DecisionTree"].accuracy_history(),
        trainers["RandomForest"].accuracy_history(),
        labels=("Decision Tree", "Random Forest"),
        title="Streaming Accuracy: Tree vs Random Forest",
        ylabel="Accuracy",
        save_path="benchmark_tree_vs_forest.png"
    )

    # 2. Accuracy comparison: Bagging vs Boosting
    compare_models(
        trainers["Bagging"].accuracy_history(),
        trainers["Boosting"].accuracy_history(),
        labels=("Bagging", "Boosting"),
        title="Streaming Accuracy: Bagging vs Boosting",
        ylabel="Accuracy",
        save_path="benchmark_bagging_vs_boosting.png"
    )

    # 3. Memory growth: all models
    _plot_all_memory(trainers)

    # 4. Time per chunk: all models
    _plot_all_time(trainers)

    print("\nPlots saved:")
    print("  benchmark_tree_vs_forest.png")
    print("  benchmark_bagging_vs_boosting.png")
    print("  benchmark_memory.png")
    print("  benchmark_time.png")

    return trainers


# ======================================================================
# Multi-model memory + time plots
# ======================================================================

def _plot_all_memory(trainers):
    """Plot memory (KB) over chunks for all models on one chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    for (name, trainer), color in zip(trainers.items(), colors):
        mem = trainer.memory_history()
        ax.plot(range(1, len(mem) + 1), mem, label=name, color=color, linewidth=2)

    ax.set_title("Model Memory Footprint Over Streaming Chunks")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Memory (KB)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_memory.png", bbox_inches='tight')
    plt.close()


def _plot_all_time(trainers):
    """Plot time per chunk (seconds) for all models on one chart."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']

    for (name, trainer), color in zip(trainers.items(), colors):
        times = trainer.time_history()
        ax.plot(range(1, len(times) + 1), times, label=name,
                color=color, linewidth=2, marker='o', markersize=3)

    ax.set_title("Time per Chunk (seconds)")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("benchmark_time.png", bbox_inches='tight')
    plt.close()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("Running streaming benchmark  (chunk_size=50, n=500)")
    run_benchmark(chunk_size=50)