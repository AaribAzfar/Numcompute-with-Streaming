![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-Used-orange)

# NumCompute

NumCompute is a modular machine learning framework developed from scratch using only Python, NumPy, and Matplotlib.

The project supports end-to-end machine learning workflows, including data ingestion, preprocessing, streaming learning, decision trees, random forests, evaluation metrics, pipeline abstraction, and real-time visualisation. All components are designed to support incremental updates through streaming data, enabling both batch and online learning scenarios while maintaining clean APIs, numerical stability, and efficient vectorised computation.

## 🚀 Features

- 📥 **Data I/O**
  - CSV reader with missing value handling

- 🧼 **Preprocessing**
  - StandardScaler, MinMaxScaler
  - OneHotEncoder for categorical data
  - Streaming

- 🔍 **Sorting & Searching**
  - Stable sorting, multi-key sorting
  - Top-k using `argpartition`
  - Quickselect (k-th smallest)
  - Binary search

Supports:

- fit()
- transform()
- partial_fit()

for streaming workflows.

- 🏆 **Ranking**
  - Ranking with tie handling (average, dense, ordinal)
  - Percentile computation

- 📊 **Statistics**
  - Mean, variance, std (NaN-safe)
  - Quantiles
  - Streaming

- 🌳 **Decision Trees**
- DecisionTreeClassifier
- Gini impurity splitting
- Configurable depth
- Incremental updates via partial_fit()

- 🌲 **Ensemble Learning**
- Random Forest Classifier
- Bootstrap sampling
- Majority voting
- Multiple decision trees

- 📏 **Metrics**
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix, MSE
  - Streaming

- ⚡ **Optimisation**
  - Finite-difference gradients
  - Jacobian estimation

- 🔗 **Pipeline API**
  - Transformer-based design
  - Sequential pipelines and feature unions
  - Streaming

- 📡 **Streaming Learning**
- Chunk-based training
- Online model updates
- Online metric tracking
- StreamTrainer abstraction

- 📈 **Visualisation**
- Accuracy over time
- Error over time
- Model comparison plots
- Streaming metric visualisation

- ⏱ **Benchmarking**
  - Compared Tree and ensemble

## 📁 Project Structure

```bash
NumCompute/
├── numcompute/ # Core library
├── benchmark/ # benchmark
├── tests/ # Unit tests
├── demo/ # Demo scripts / notebook
├── README.md
├── pyproject.toml`
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/AaribAzfar/Numcompute-with-Streaming.git
cd NumCompute
```

Install the package:

```bash
pip install -e .
```

Run the benchmark to get the results

```bash
python -m demo.
```

🧪 Testing

Run tests:

```bash
pip install pytest
pytest tests/
```

## 🧠 Design Principles

- Vectorisation-first: Avoid Python loops where possible
- Numerical Stability: Handles NaNs, overflow/underflow
- Modular API: Consistent `fit`, `transform`, `predict` , `partial_fit`interface
- Streaming Compatibility: Core modules support incremental learning
- Reusability: Components work independently and in pipelines

## 👥 Authors

- Sheikh Muhammad Aarib Azfar
