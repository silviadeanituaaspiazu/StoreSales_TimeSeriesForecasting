# 🛒 Store Sales Time Series Forecasting

An end-to-end machine learning project to predict store sales for Corporación Favorita (Ecuador).

This project goes beyond standard Jupyter Notebook experimentation by wrapping the entire data science lifecycle in a production-ready MLOps architecture. By prioritizing testability and CI/CD from day one, it ensures that the EDA and modeling phases are built on a highly reliable data foundation.

## 🎯 Project Lifecycle

1. **Robust Data Pipeline (Completed):** Modularized data preprocessing with automated end-to-end testing to prevent data leakage, handle missing values, and avoid merge explosions.
2. **Exploratory Data Analysis (WIP):** Deep dive into seasonality, store clustering, holiday impacts, and long-term sales trends.
3. **Feature Engineering (WIP):** Implementation of dynamic lag variables, rolling windows, and domain-specific time features.
4. **Modeling & Evaluation (WIP):** Algorithm selection (e.g., XGBoost, LightGBM), strict time-series cross-validation, and RMSLE optimization.

## 🛠️ Engineering Approach & Stack

- **Dependency Management:** [Poetry](https://python-poetry.org/) (guarantees deterministic builds via `poetry.lock`).
- **Data Pipeline:** Modularized `src-layout` separating pure logic (transformations) from I/O operations (reading/writing).
- **Automated Testing:** `pytest` suite simulating edge cases (missing dates, merge explosions, data type casting).
- **Code Quality:** Automated formatting enforced by `Black`.
- **CI/CD:** GitHub Actions workflow triggers on every push/PR to run tests and format checks on ephemeral Linux runners.

## 🚀 Running the Pipeline

This project uses Poetry for environment management. To reproduce the environment and run the data pipeline:

```bash
# 1. Install dependencies
poetry install

# 2. Run the test suite (validates data logic and edge cases)
poetry run pytest -v

# 3. Generate the processed master table (.parquet)
poetry run python src/storesales/preprocessing.py
```

📊 Exploratory Data Analysis (EDA)
[WIP: Key findings on seasonality, trend analysis, and holiday impacts will be documented here.]

🧬 Feature Engineering
[WIP: Implementation of lag variables, rolling windows, and time-based features.]

🤖 Modeling & Experimentation
[WIP: Model selection, time-series cross-validation strategy, and RMSLE evaluation.]