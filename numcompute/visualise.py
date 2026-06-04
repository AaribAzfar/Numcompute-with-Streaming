import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# Internal helper
# ======================================================================

def _save_or_show(save_path):
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ======================================================================
# Required plots (from spec)
# ======================================================================

def plot_metric_over_time(metric_values, title="Metric over Chunks",ylabel="Value", save_path=None):
    
    metric_values = np.asarray(metric_values, dtype=float) 
    chunks = np.arange(1, len(metric_values) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(chunks, metric_values, marker='o', linewidth=2,
            markersize=4, color='steelblue', label=ylabel)

    # Dashed line for the overall mean
    mean_val = np.nanmean(metric_values)
    ax.axhline(mean_val, linestyle='--', color='grey', linewidth=1,
               label=f"Mean: {mean_val:.4f}")

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),title="Model Comparison", ylabel="Value", save_path=None):
   
    metric1 = np.asarray(metric1, dtype=float)
    metric2 = np.asarray(metric2, dtype=float)

    # Use the shorter length if they differ
    n = min(len(metric1), len(metric2))
    chunks = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(chunks, metric1[:n], marker='o', linewidth=2,
            markersize=4, color='steelblue', label=labels[0])
    ax.plot(chunks, metric2[:n], marker='s', linewidth=2,
            markersize=4, color='tomato', label=labels[1])

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)


def plot_predictions_vs_ground_truth(y_true, y_pred,title="Predictions vs Ground Truth",save_path=None):
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n      = len(y_true)
    idx    = np.arange(n)
    correct = y_true == y_pred

    fig, ax = plt.subplots(figsize=(10, 4))

    # Green dots = correct, red dots = wrong
    ax.scatter(idx[correct],  y_true[correct],  color='green',
               label='Correct',   alpha=0.6, s=20, zorder=3)
    ax.scatter(idx[~correct], y_true[~correct], color='red',
               label='Wrong (true)',  alpha=0.6, s=20, zorder=3)
    ax.scatter(idx[~correct], y_pred[~correct], color='orange',
               label='Wrong (pred)',  alpha=0.6, s=20, marker='x', zorder=4)

    accuracy = np.mean(correct)
    ax.set_title(f"{title}  —  Accuracy: {accuracy:.4f}")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Class label")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)


# ======================================================================
# Extra useful plots
# ======================================================================

def plot_confusion_matrix(cm, classes=None, title="Confusion Matrix",  save_path=None):

    cm = np.asarray(cm, dtype=int)
    n  = cm.shape[0]

    if classes is None:
        classes = [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    # Write count numbers inside each cell
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    _save_or_show(save_path)


def plot_cumulative_accuracy(trainer_log, title="Cumulative Accuracy",save_path=None):

    chunks   = [e["chunk"]        for e in trainer_log]
    acc      = [e["accuracy"]     for e in trainer_log]
    cum_acc  = [e["cum_accuracy"] for e in trainer_log]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(chunks, acc,     marker='o', linewidth=1.5, markersize=3,
            color='steelblue', alpha=0.6, label="Per-chunk accuracy")
    ax.plot(chunks, cum_acc, linewidth=2,
            color='navy', label="Cumulative accuracy")

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)


def plot_memory_over_time(trainer_log, title="Model Memory Usage",save_path=None):

    chunks = [e["chunk"]                    for e in trainer_log]
    mem_kb = [e["memory_bytes"] / 1024      for e in trainer_log]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.fill_between(chunks, mem_kb, alpha=0.3, color='darkorange')
    ax.plot(chunks, mem_kb, color='darkorange', linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Memory (KB)")
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)


def plot_roc_curve(fpr, tpr, auc_score=None, title="ROC Curve",save_path=None):

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    label = "ROC curve"
    if auc_score is not None:
        label += f" (AUC = {auc_score:.4f})"

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr, tpr, color='steelblue', linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey',
            linewidth=1, label="Random classifier")

    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_or_show(save_path)