import pandas as pd
import numpy as np
from storesales.preprocessing import build_features


def test_preprocessing_main_pipeline():
    dates = pd.date_range(start="2014-01-01", periods=10)

    train_raw = pd.DataFrame(
        {
            "id": range(10),
            "date": dates,
            "store_nbr": 1,
            "family": "PRODUCE",
            "sales": [10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            "onpromotion": 0,
        }
    )

    oil_raw = pd.DataFrame(
        {
            "date": dates,
            "dcoilwtico": [
                90.0,
                91.0,
                np.nan,
                93.0,
                94.0,
                95.0,
                96.0,
                97.0,
                98.0,
                99.0,
            ],
        }
    )

    stores_raw = pd.DataFrame(
        {
            "store_nbr": [1],
            "city": ["Quito"],
            "state": ["Pichincha"],
            "type": ["D"],
            "cluster": [13],
        }
    )

    transactions_raw = pd.DataFrame(
        {"date": dates, "store_nbr": 1, "transactions": 500}
    )

    holidays_raw = pd.DataFrame(
        {
            "date": [pd.to_datetime("2014-01-05")],
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "description": ["Día Festivo Inventado"],
            "transferred": [False],
        }
    )

    df_final = build_features(
        train_raw=train_raw,
        stores_raw=stores_raw,
        holidays_raw=holidays_raw,
        transactions_raw=transactions_raw,
        oil_raw=oil_raw,
        lags=[1, 3],
        drop_lag_nan=True,
    )

    # 1. Row count check (No join explosion)
    assert len(df_final) <= len(train_raw), "Too many rows."

    # 2. Missing values check
    cols_to_check = [
        "store_nbr",
        "family",
        "date",
        "sales",
        "onpromotion",
        "sales_missing",
        "weekday",
        "month",
        "week",
        "year",
        "is_int_sales",
        "city",
        "state",
        "store_type",
        "store_cluster",
        "is_holiday",
        "is_earthquake_manabi",
        "is_closed",
        "is_drastic_drop",
        "is_commercial_spike",
        "is_regular_holiday",
        "dcoilwtico",
    ]
    cols_with_nans = (
        df_final[cols_to_check].columns[df_final[cols_to_check].isna().any()].tolist()
    )
    assert (
        len(cols_with_nans) == 0
    ), f"Pipeline failed! Missing values found in: {cols_with_nans}"

    # 3. Data type checks
    assert (
        df_final["store_nbr"].dtype.name == "category"
    ), "store_nbr lost its category dtype after joins."
    assert (
        df_final["family"].dtype.name == "category"
    ), "family lost its category dtype after joins."

    # 4. Residual columns check (Cleanup)
    garbage_columns = [
        col for col in df_final.columns if col.endswith("_x") or col.endswith("_y")
    ]
    assert (
        len(garbage_columns) == 0
    ), f"Duplicate merge columns found: {garbage_columns}"
