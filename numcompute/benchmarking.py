import numpy as np
import matplotlib.pyplot as plt

from tree import DecisionTreeClassifier, RandomForestClassifier
from ensemble import BaggingClassifier, BoostingClassifier
from stream import StreamTrainer
from visualise import (
    plot_metric_over_time,              # single metric over chunks
    compare_models,                     # two models side by side
    plot_predictions_vs_ground_truth,   # preds vs truth on test set
    plot_cumulative_accuracy,           # per-chunk + cumulative on one chart
    plot_memory_over_time,              # memory footprint over chunks
)


# ======================================================================
# Dataset
# ======================================================================

def make_dataset(n=600, n_features=6, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, n_features)
    y = ((X[:, 0] + X[:, 1] - X[:, 2] + 0.5 * X[:, 3]) > 0).astype(int)
    return X, y


# ======================================================================
# Main benchmark
# ======================================================================

def run_benchmark(chunk_size=50):

    X, y = make_dataset()

    # Hold out last 100 samples as a fixed test set
    X_train, y_train = X[:-100], y[:-100]
    X_test,  y_test  = X[-100:], y[-100:]

    models = {
        "DecisionTree (base)":      DecisionTreeClassifier(max_depth=4),
        "RandomForest (ensemble)":  RandomForestClassifier(n_estimators=10, max_depth=4),
        "Bagging (ensemble)":       BaggingClassifier(n_estimators=10,      max_depth=4),
        "Boosting (ensemble)":      BoostingClassifier(n_estimators=10,     max_depth=1),
    }

    trainers = {
        name: StreamTrainer(model, verbose=False)
        for name, model in models.items()
    }

    # Stream all models through the same chunks
    for name, trainer in trainers.items():
        trainer.stream(X_train, y_train, chunk_size=chunk_size)

    # --- Results table -----------------------------------------------
    print("\n" + "=" * 72)
    print("BASE MODEL vs ENSEMBLE -- Streaming Benchmark")
    print("=" * 72)
    print(f"Train: {len(y_train)} samples  |  Test: {len(y_test)} samples  |  "
          f"Features: {X.shape[1]}  |  chunk_size: {chunk_size}")
    print()
    print(f"{'Model':<28}  {'Cum Acc':>8}  {'Avg Time(s)':>11}  "
          f"{'Final Mem(KB)':>13}  {'vs Base':>8}")
    print("-" * 72)

    base_acc = trainers["DecisionTree (base)"].cumulative_accuracy()

    for name, trainer in trainers.items():
        cum_acc   = trainer.cumulative_accuracy()
        avg_time  = np.mean(trainer.time_history())
        final_mem = trainer.memory_history()[-1]
        diff      = cum_acc - base_acc
        diff_str  = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        tag       = "  <- base" if "(base)" in name else ""
        print(f"{name:<28}  {cum_acc:>8.4f}  {avg_time:>11.5f}  "
              f"{final_mem:>13.1f}  {diff_str:>8}{tag}")

    print("=" * 72)
    print()
    print("Interpretation:")
    print("  Cum Acc   -- overall accuracy across all chunks (higher = better)")
    print("  Avg Time  -- average seconds to fit one chunk (lower = faster)")
    print("  Final Mem -- KB used by model after all chunks (lower = lighter)")
    print("  vs Base   -- accuracy gain/loss compared to the single tree")
    print()

    # Shorthand
    tree_t   = trainers["DecisionTree (base)"]
    forest_t = trainers["RandomForest (ensemble)"]
    bag_t    = trainers["Bagging (ensemble)"]
    boost_t  = trainers["Boosting (ensemble)"]

    # ================================================================
    # Plots via visualise.py
    # ================================================================

    # 1. plot_metric_over_time
    #    Base model accuracy chunk by chunk
    plot_metric_over_time(
        tree_t.accuracy_history(),
        title="Decision Tree (base) -- Per-Chunk Accuracy",
        ylabel="Accuracy",
        save_path="bench_1_tree_accuracy.png"
    )

    # 2. compare_models  (base vs best ensemble)
    compare_models(
        tree_t.accuracy_history(),
        forest_t.accuracy_history(),
        labels=("Decision Tree (base)", "Random Forest (ensemble)"),
        title="Base vs Ensemble -- Per-Chunk Accuracy",
        ylabel="Accuracy",
        save_path="bench_2_tree_vs_forest.png"
    )

    # 3. compare_models  (bagging vs boosting)
    compare_models(
        bag_t.accuracy_history(),
        boost_t.accuracy_history(),
        labels=("Bagging (ensemble)", "Boosting (ensemble)"),
        title="Bagging vs Boosting -- Per-Chunk Accuracy",
        ylabel="Accuracy",
        save_path="bench_3_bagging_vs_boosting.png"
    )

    # 4. plot_cumulative_accuracy  (one per model)
    plot_cumulative_accuracy(
        tree_t.log,
        title="Decision Tree (base) -- Cumulative Accuracy",
        save_path="bench_4_tree_cumulative.png"
    )
    plot_cumulative_accuracy(
        forest_t.log,
        title="Random Forest (ensemble) -- Cumulative Accuracy",
        save_path="bench_5_forest_cumulative.png"
    )

    # 5. plot_memory_over_time  (base vs ensemble memory cost)
    plot_memory_over_time(
        tree_t.log,
        title="Decision Tree (base) -- Memory Over Chunks",
        save_path="bench_6_tree_memory.png"
    )
    plot_memory_over_time(
        forest_t.log,
        title="Random Forest (ensemble) -- Memory Over Chunks",
        save_path="bench_7_forest_memory.png"
    )

    # 6. plot_predictions_vs_ground_truth  (all models on test set)
    for name, trainer in trainers.items():
        y_pred = trainer.model.predict(X_test)
        slug   = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_predictions_vs_ground_truth(
            y_test, y_pred,
            title=f"{name} -- Predictions vs Ground Truth (test set)",
            save_path=f"bench_8_preds_{slug}.png"
        )

    # ================================================================
    # Multi-model charts (raw matplotlib)
    # These put all 4 models on one chart for direct comparison --
    # beyond what visualise.py's two-model compare_models covers.
    # ================================================================
    _plot_all_cumulative(trainers)
    _plot_time_and_memory(trainers)

    print("Plots saved (visualise.py functions used where possible):")
    print("  bench_1_tree_accuracy.png       -- plot_metric_over_time")
    print("  bench_2_tree_vs_forest.png      -- compare_models")
    print("  bench_3_bagging_vs_boosting.png -- compare_models")
    print("  bench_4_tree_cumulative.png     -- plot_cumulative_accuracy")
    print("  bench_5_forest_cumulative.png   -- plot_cumulative_accuracy")
    print("  bench_6_tree_memory.png         -- plot_memory_over_time")
    print("  bench_7_forest_memory.png       -- plot_memory_over_time")
    print("  bench_8_preds_*.png             -- plot_predictions_vs_ground_truth")
    print("  bench_9_all_cumulative.png      -- raw matplotlib (all 4 models)")
    print("  bench_10_time_and_memory.png    -- raw matplotlib (bar charts)")

    return trainers


# ======================================================================
# Multi-model plots (raw matplotlib)
# ======================================================================

COLORS = {
    "DecisionTree (base)":     "steelblue",
    "RandomForest (ensemble)": "tomato",
    "Bagging (ensemble)":      "seagreen",
    "Boosting (ensemble)":     "darkorange",
}
STYLES = {
    "DecisionTree (base)":     "--",
    "RandomForest (ensemble)": "-",
    "Bagging (ensemble)":      "-",
    "Boosting (ensemble)":     "-",
}


def _plot_all_cumulative(trainers):
    """All four models on one cumulative accuracy chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, trainer in trainers.items():
        cum = trainer.cumulative_accuracy_history()
        ax.plot(range(1, len(cum) + 1), cum,
                label=name, color=COLORS[name],
                linestyle=STYLES[name], linewidth=2.5)
    ax.set_title("Cumulative Accuracy: Base vs All Ensembles")
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Cumulative Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bench_9_all_cumulative.png", bbox_inches='tight')
    plt.close()


def _plot_time_and_memory(trainers):
    """Bar charts: avg time and final memory for all four models."""
    names     = list(trainers.keys())
    avg_times = [np.mean(t.time_history()) for t in trainers.values()]
    final_mem = [t.memory_history()[-1]    for t in trainers.values()]
    colors    = [COLORS[n] for n in names]
    short     = ["Tree\n(base)", "RandomForest\n(ensemble)",
                 "Bagging\n(ensemble)", "Boosting\n(ensemble)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(short, avg_times, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_title("Avg Time per Chunk (seconds)")
    ax1.set_ylabel("Seconds")
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(short, final_mem, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title("Final Memory Footprint (KB)")
    ax2.set_ylabel("KB")
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle("Cost of Ensembling vs Base Model", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("bench_10_time_and_memory.png", bbox_inches='tight')
    plt.close()


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    print("Running benchmark: base model vs ensembles under streaming...")
    run_benchmark(chunk_size=50)