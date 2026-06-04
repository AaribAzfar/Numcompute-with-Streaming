from typing import Union, Sequence, Optional, Dict, Any
import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]  # Accepts Python lists, tuples, or NumPy arrays of numbers


# Validation Helper

def _validate_array(
    X: ArrayLike,
    allow_empty: bool = False
) -> np.ndarray:
    """
    Validates and converts input into a NumPy array.

    - Ensures input is not None
    - Ensures input is numeric
    - Optionally allows empty arrays
    """
    if X is None:
        raise ValueError("Input cannot be None")

    arr = np.asarray(X)  # Convert input to NumPy array

    if not np.issubdtype(arr.dtype, np.number):  # Checks if array contains numeric data
        raise ValueError("Input must be numeric")

    if not allow_empty and arr.size == 0:   # Prevent empty arrays unless explicitly allowed
        raise ValueError("Input array cannot be empty")

    return arr


# Basic Statistics

# Mean
def mean(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute mean while ignoring NaN values
    X = _validate_array(X)
    return np.nanmean(X, axis=axis)


# Median
def median(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute median while ignoring NaN values
    X = _validate_array(X)
    return np.nanmedian(X, axis=axis)


# Standard Deviation
def std(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute standard deviation while ignoring NaNs
    X = _validate_array(X)
    return np.nanstd(X, axis=axis)


# Minimum
def minimum(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute minimum value while ignoring NaNs
    X = _validate_array(X)
    return np.nanmin(X, axis=axis)


# Maximum
def maximum(X: ArrayLike, axis: Optional[int] = None) -> float:  # Compute maximum value while ignoring NaNs
    X = _validate_array(X)
    return np.nanmax(X, axis=axis)


# Histogram 
def histogram(
    X: ArrayLike,
    bins: int = 10,
    range: Optional[tuple] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram of the data.

    - bins: number of bins (must be positive integer)
    - range: lower and upper range of bins
    """
    X = _validate_array(X, allow_empty=True)  # Allows empty arrays (NumPy can handle this case)

    if not isinstance(bins, int) or bins <= 0:  # Validate bins input
        raise ValueError("bins must be a positive integer")

    return np.histogram(X, bins=bins, range=range)


# Quantiles
def quantiles(
    X: ArrayLike,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Compute percentiles (0–100 scale).

    - q can be a single value or a list of percentiles
    - NaN values are ignored
    """
    X = _validate_array(X)

    q_arr = np.asarray(q)  # Convert q into NumPy array for uniform handling

    if np.any((q_arr < 0) | (q_arr > 100)):  # Ensures if all percentile values are within valid range
        raise ValueError("q must be between 0 and 100")

    return np.nanpercentile(X, q_arr, axis=axis)


# Describe
def describe(
    X: ArrayLike,
    axis: int = 0
) -> Dict[str, Any]:
    """
    Return a summary of basic statistics:
    mean, standard deviation, min, max.
    """
    X = _validate_array(X)

    return {
        "mean": mean(X, axis),
        "std": std(X, axis),
        "min": minimum(X, axis),
        "max": maximum(X, axis),
    }

class StreamingStats:
   
    def __init__(self, bins=10, window=None):
        if not isinstance(bins, int) or bins <= 0:
            raise ValueError("bins must be a positive integer")
        self.bins = bins
        self.window = window
        self.reset()
 
    def reset(self):
        """Clear all accumulated state."""
        self._n = 0          # total samples seen
        self._mean = None    # running mean (per feature or scalar)
        self._M2 = None      # running sum of squared deviations (Welford)
        self._min = None     # running min
        self._max = None     # running max
 
        # Histogram: fixed bin edges set on first chunk, counts accumulated
        self._bin_edges = None
        self._bin_counts = None
 
        # Sliding window buffer (flat 1D or 2D depending on input)
        self._window_buffer = []
 
        return self
 
    def update_stats(self, X_chunk):
        
        X_chunk = _validate_array(X_chunk)
        X_flat = X_chunk.flatten()
        X_valid = X_flat[~np.isnan(X_flat)]  # drop NaNs
 
        if X_valid.size == 0:
            return self  # nothing to update
 
        # ---- Sliding window ------------------------------------------
        if self.window is not None:
            self._window_buffer.extend(X_valid.tolist())
            # Keep only the last `window` samples
            if len(self._window_buffer) > self.window:
                self._window_buffer = self._window_buffer[-self.window:]
            # Recompute stats from the window buffer directly
            buf = np.array(self._window_buffer)
            self._n    = len(buf)
            self._mean = np.mean(buf)
            self._M2   = np.var(buf) * len(buf)   # M2 = var * n
            self._min  = np.min(buf)
            self._max  = np.max(buf)
            self._update_histogram(buf)
            return self
 
        # ---- Cumulative: Welford's online algorithm -------------------
        for x in X_valid:
            self._n += 1
            if self._mean is None:
                self._mean = 0.0
                self._M2   = 0.0
            delta      = x - self._mean
            self._mean += delta / self._n
            delta2     = x - self._mean
            self._M2  += delta * delta2
 
        # Running min / max
        chunk_min = np.min(X_valid)
        chunk_max = np.max(X_valid)
        self._min = chunk_min if self._min is None else min(self._min, chunk_min)
        self._max = chunk_max if self._max is None else max(self._max, chunk_max)
 
        # Update histogram with this chunk
        self._update_histogram(X_valid)
 
        return self
 
    def _update_histogram(self, X_valid):
        
        if self._bin_edges is None:
            # First chunk: set fixed bin edges from its range
            lo, hi = np.min(X_valid), np.max(X_valid)
            if lo == hi:
                lo, hi = lo - 0.5, hi + 0.5
            self._bin_edges  = np.linspace(lo, hi, self.bins + 1)
            self._bin_counts = np.zeros(self.bins, dtype=int)
 
        counts, _ = np.histogram(X_valid, bins=self._bin_edges)
        self._bin_counts += counts
 
    def result(self):
        
        if self._n == 0:
            raise ValueError("No data seen yet. Call update_stats() first.")
 
        var = self._M2 / self._n if self._n > 0 else 0.0
 
        return {
            "n":    self._n,
            "mean": self._mean,
            "var":  var,
            "std":  np.sqrt(var),
            "min":  self._min,
            "max":  self._max,
        }
 
    def histogram(self):
       
        if self._bin_edges is None:
            raise ValueError("No data seen yet. Call update_stats() first.")
        return self._bin_counts.copy(), self._bin_edges.copy()
 
    def quantiles(self, q):
        
        if self.window is not None:
            data = np.array(self._window_buffer)
        else:
            raise ValueError(
                "Exact quantiles require storing all data. "
                "Use window mode or call quantiles() on the full dataset."
            )
        q_arr = np.asarray(q)
        if np.any((q_arr < 0) | (q_arr > 100)):
            raise ValueError("q must be between 0 and 100")
        return np.nanpercentile(data, q_arr)
 
    def __repr__(self):
        return (
            f"StreamingStats(bins={self.bins}, window={self.window}, "
            f"n_seen={self._n})"
        )