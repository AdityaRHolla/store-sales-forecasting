# 🏢 Corporación Favorita - High-Performance Scalable Demand Forecasting Engine

An end-to-end time-series machine learning system built to predict multi-million row grocery item demands across parallel store departments. Implemented using an isolated per-family architecture to eliminate high-volume variance skewing.

## 🛠️ System Architecture & Framework
* **Data Ingestion & Memory Optimization:** Implemented smart data type downcasting, shrinking a 3-million-row Pandas grid from its raw size down to a high-performance **148.8 MB** footprint.
* **Leak-Free Validation Strategy:** Configured a strict chronological rolling horizon validation timeline to completely mirror true test-set parameters and prevent temporal data leakage.
* **Isolated Multi-Model Loops:** Deployed 33 separate, specialized gradient boosting models (`LightGBM`) to isolate massive-volume staple items from flat-line low-volume tracking groups.

## 📁 Engineering Framework Directory
* `src/data_loader.py`: Timeline gap repair, holiday calendar filters, and downcasting optimization.
* `src/features.py`: Trigonometric cyclical time loops, 21/28 day strict weekly matching lags, and store-family identifier shortcuts.
* `src/train.py`: Chronological window splits and adaptive per-family scale training workflows.

## 📈 Metric Optimisation Ledger
* `Initial Baseline Model`: 0.5986 RMSLE
* `Added Historical Safe Horizons`: 0.4388 RMSLE
* `Isolated Per-Family Modeling`: 0.4024 RMSLE
* **Current Best Submission Score:** **0.42658** (Live Kaggle Public Leaderboard)
