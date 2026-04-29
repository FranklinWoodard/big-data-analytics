# Big Data Analytics

Projects from CS469 — Big Data Analytics. Each demonstrates a different big data concept using Python.

---

## Projects

### TF-IDF Theme Extraction
Extracts themes from 14,640 airline tweets using term frequency–inverse document frequency. Built with scikit-learn and pandas. Outputs the original dataset with a `top_themes` column appended.

**Concepts:** TF-IDF, n-gram analysis (unigram + bigram), sublinear term frequency scaling

### Bitcoin Key-Value Store
A key-value store of Bitcoin and blockchain concepts. Demonstrates dictionary operations — insert, lookup, update, delete — analogous to MapReduce patterns on Hadoop.

**Concepts:** Key-value data model, MapReduce, distributed storage patterns

### KNN Price Direction Predictor
Predicts whether Bitcoin's next-period closing price will move up (1) or down (0) using K-Nearest Neighbors on historical OHLCV data (January 2012 – April 2026). Uses temporal train/test split to prevent data leakage.

**Concepts:** KNN classification, feature engineering, temporal data splitting, predictive analytics

### CDC Health Data Pipeline
A full 5-stage data pipeline processing CDC PLACES County Health Data (2022 release) across 3,143 U.S. counties and 28 health metrics.

| Stage | Description |
|-------|-------------|
| Ingest | Load CSV from local path |
| Storage | Structured pandas DataFrame |
| Cleansing | Handle missing values, drop CI columns, normalize types |
| Preprocessing | Derive composite Health Burden Index, state-level aggregations |
| Utilization | Descriptive statistics report + visualizations |

**Concepts:** ETL pipeline design, data cleansing, composite index construction, exploratory data analysis

---

## Stack
Python, pandas, scikit-learn, matplotlib

## Author
Franklin Woodard — [GitHub](https://github.com/FranklinWoodard)
