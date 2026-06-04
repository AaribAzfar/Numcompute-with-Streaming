import numpy as np
import time


# ======================================================================
# StreamTrainer
# ======================================================================

class StreamTrainer:

    def __init__(self, model, verbose=True):
        self.model   = model
        self.verbose = verbose

        # ---- Log: one entry per chunk --------------------------------
        # Each entry is a dict with keys:
        #   chunk, n_samples, accuracy, time_s, memory_bytes
        self.log = []

        # ---- Running totals for cumulative accuracy ------------------
        self._total_correct = 0
        self._total_seen    = 0
        self._chunk_count   = 0

    # ------------------------------------------------------------------ #
    # Core methods
    # ------------------------------------------------------------------ #

    def fit_chunk(self, X_chunk, y_chunk):
        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        self._chunk_count += 1

        # Time the fit
        t_start = time.perf_counter()
        self.model.partial_fit(X_chunk, y_chunk)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        mem     = self._model_memory_bytes()

        # Score this chunk immediately after fitting
        y_pred   = self.model.predict(X_chunk)
        n        = len(y_chunk)
        correct  = int(np.sum(y_pred == y_chunk))
        acc      = correct / n

        # Update cumulative totals
        self._total_correct += correct
        self._total_seen    += n

        # Log this chunk
        entry = {
            "chunk":        self._chunk_count,
            "n_samples":    n,
            "accuracy":     acc,
            "cum_accuracy": self._total_correct / self._total_seen,
            "time_s":       elapsed,
            "memory_bytes": mem,
        }
        self.log.append(entry)

        if self.verbose:
            self._print_entry(entry)

        return self

    def score_chunk(self, X_chunk, y_chunk):
        if not self._is_fitted():
            raise ValueError("Model has not been fitted yet. Call fit_chunk() first.")

        X_chunk = np.asarray(X_chunk, dtype=float)
        y_chunk = np.asarray(y_chunk)

        y_pred = self.model.predict(X_chunk)
        acc    = float(np.mean(y_pred == y_chunk))

        if self.verbose:
            print(f"  [score_chunk] chunk={self._chunk_count}  "
                  f"val_accuracy={acc:.4f}  n={len(y_chunk)}")

        return acc

    # ------------------------------------------------------------------ #
    # Streaming over a full dataset split into chunks
    # ------------------------------------------------------------------ #

    def stream(self, X, y, chunk_size=100, X_val=None, y_val=None):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n = len(y)
        starts = range(0, n, chunk_size)

        for start in starts:
            end     = min(start + chunk_size, n)
            X_chunk = X[start:end]
            y_chunk = y[start:end]

            self.fit_chunk(X_chunk, y_chunk)

            # Optionally score on validation set after each chunk
            if X_val is not None and y_val is not None:
                val_acc = self.score_chunk(
                    np.asarray(X_val, dtype=float),
                    np.asarray(y_val)
                )
                # Add val_accuracy into the last log entry
                self.log[-1]["val_accuracy"] = val_acc

        return self

    # ------------------------------------------------------------------ #
    # Summary methods
    # ------------------------------------------------------------------ #

    def cumulative_accuracy(self):
       
        if self._total_seen == 0:
            raise ValueError("No chunks have been trained on yet.")
        return self._total_correct / self._total_seen

    def summary(self):
        
        if not self.log:
            print("No chunks logged yet.")
            return

        print(f"\n{'='*65}")
        print(f"{'Chunk':>6}  {'N':>6}  {'Acc':>8}  {'Cum Acc':>8}  "
              f"{'Time(s)':>8}  {'Mem(KB)':>8}")
        print(f"{'-'*65}")

        for e in self.log:
            val_str = ""
            if "val_accuracy" in e:
                val_str = f"  val={e['val_accuracy']:.4f}"
            print(
                f"{e['chunk']:>6}  "
                f"{e['n_samples']:>6}  "
                f"{e['accuracy']:>8.4f}  "
                f"{e['cum_accuracy']:>8.4f}  "
                f"{e['time_s']:>8.4f}  "
                f"{e['memory_bytes']/1024:>8.1f}"
                f"{val_str}"
            )

        print(f"{'='*65}")
        print(f"Cumulative accuracy: {self.cumulative_accuracy():.4f}  |  "
              f"Total samples: {self._total_seen}  |  "
              f"Chunks: {self._chunk_count}")
        print()

    # ------------------------------------------------------------------ #
    # Log accessors — for feeding into visualise.py
    # ------------------------------------------------------------------ #

    def accuracy_history(self):
        return [e["accuracy"] for e in self.log]

    def cumulative_accuracy_history(self):
        return [e["cum_accuracy"] for e in self.log]

    def memory_history(self):
    
        return [e["memory_bytes"] / 1024 for e in self.log]

    def time_history(self):
    
        return [e["time_s"] for e in self.log]

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _is_fitted(self):
    
        return self._chunk_count > 0

    def _model_memory_bytes(self):
    
        total = 0
        model = self.model

        # If this is a Pipeline, inspect the final step (the model)
        if hasattr(model, 'steps'):
            _, model = model.steps[-1]

        # Trees stored in a list (BaggingClassifier, RandomForestClassifier)
        if hasattr(model, '_trees'):
            for tree in model._trees:
                total += self._tree_bytes(tree)

        # Single decision tree
        elif hasattr(model, '_root'):
            total += self._tree_bytes(model)

        # Boosting: also count tree weights
        if hasattr(model, '_tree_weights') and model._tree_weights:
            total += np.asarray(model._tree_weights).nbytes

        return total

    def _tree_bytes(self, tree):

        total = 0

        if hasattr(tree, '_X_buffer') and tree._X_buffer is not None:
            total += tree._X_buffer.nbytes

        if hasattr(tree, '_y_buffer') and tree._y_buffer is not None:
            total += tree._y_buffer.nbytes

        return total

    def _print_entry(self, e):
        """Print one log entry to stdout."""
        print(
            f"[chunk {e['chunk']:>3}]  "
            f"n={e['n_samples']:>4}  "
            f"acc={e['accuracy']:.4f}  "
            f"cum_acc={e['cum_accuracy']:.4f}  "
            f"time={e['time_s']:.4f}s  "
            f"mem={e['memory_bytes']/1024:.1f}KB"
        )

    def __repr__(self):
        return (
            f"StreamTrainer("
            f"model={type(self.model).__name__}, "
            f"chunks_seen={self._chunk_count}, "
            f"cum_accuracy={self.cumulative_accuracy() if self._total_seen > 0 else 'n/a'})"
        )