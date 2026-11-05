import pandas as pd
import numpy as np
import os

def prep_train_df(df: pd.DataFrame, lags: list[int], drop_lag_nan: bool = True) -> pd.DataFrame:
    """Preprocesses the main sales dataset by parsing dates, casting types,
    generating temporal and lag features and getting the sales types for each family.
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    df["date"] = pd.to_datetime(df["date"])
    df["onpromotion"] = df["onpromotion"].astype("category")
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["family"] = df["family"].astype("category")
    df["weekday"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week
    df["year"] = df["date"].dt.year

    df = prep_train_get_lags(df, lags)

    if drop_lag_nan:
        max_lag = max(lags)
        df = df.dropna(subset=[f"sales_lag{max_lag}"]).reset_index(drop=True)

    # Gets sales types for each family
    is_decimal = df["sales"] % 1 != 0
    family_has_decimals = is_decimal.groupby(df["family"], observed=True).any()
    df["is_int_sales"] = df["family"].map(~family_has_decimals)

    return df


def prep_train_get_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Computes raw and log-transformed sales lags grouped by store and product family."""
    for i in lags:
        df[f"sales_lag{i}"] = df.groupby(["family", "store_nbr"])["sales"].shift(i)
        df[f"sales_lag{i}_log"] = np.log1p(df[f"sales_lag{i}"])
    return df


def prep_oil_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the oil dataset by converting dates, sorting, and filling missing dates and values."""

    df["date"] = pd.to_datetime(df["date"])

    calendar = pd.date_range(start=df["date"].min(), end=df["date"].max())
    df_calendar = pd.DataFrame({"date": calendar})

    df_complete = df_calendar.merge(df, on="date", how="left")

    df_complete = df_complete.sort_values("date").reset_index(drop=True)
    df_complete["dcoilwtico"] = df_complete["dcoilwtico"].ffill().bfill()
    return df_complete


def prep_stores_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the main sales dataset by casting types"""

    df["store_nbr"] = df["store_nbr"].astype("category")
    df["cluster"] = df["cluster"].astype("category")
    df["type"] = df["type"].astype("category")
    df = df.rename(columns={"type": "store_type", "cluster": "store_cluster"})

    return df


def prep_transactions_df(transactions: pd.DataFrame) -> pd.DataFrame:
    """Add transactions data"""

    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions["store_nbr"] = transactions["store_nbr"].astype("category")

    return transactions


def prep_holiday_df(holidays_events: pd.DataFrame) -> pd.DataFrame:
    """Add holiday_events data"""

    holidays_events["date"] = pd.to_datetime(holidays_events["date"])
    holidays_events[holidays_events["transferred"] == False]

    return holidays_events


def prep_joins(
    df_sales: pd.DataFrame,
    df_stores: pd.DataFrame,
    df_holiday_events: pd.Dataframe,
    df_oil: pd.DataFrame,
    df_transactions: pd.Dataframe,
) -> pd.DataFrame:
    """Joins all the datasets"""

    # Joins the sales history dataset with the stores dataset
    df = df_sales.merge(df_stores, how="left", on=["store_nbr"])

    # Joins with the holidays dataset
    city_holidays = df_holiday_events[df_holiday_events["locale"] == "Local"]
    national_holidays = df_holiday_events[df_holiday_events["locale"] == "National"]
    regional_holidays = df_holiday_events[df_holiday_events["locale"] == "Regional"]

    df = df.merge(
        city_holidays[["date", "locale_name"]],
        how="left",
        left_on=["date", "city"],
        right_on=["date", "locale_name"],
    )
    df = df.merge(
        regional_holidays[["date", "locale_name"]],
        how="left",
        left_on=["date", "state"],
        right_on=["date", "locale_name"],
    )
    df = df.merge(
        national_holidays[["date", "locale_name"]],
        how="left",
        left_on=["date"],
        right_on=["date"],
    )

    holiday_cols = ["locale_name_x", "locale_name_y", "locale_name"]
    df["is_holiday"] = df[holiday_cols].notna().any(axis=1).astype(int)

    df = df.drop(columns=holiday_cols)

    # Joins with oil dataset
    df = df.merge(df_oil, on="date", how="left")

    # Joins with transactions dataset
    df = df.merge(df_transactions, how="left", on=["store_nbr", "date"])

    df = df.sort_values(["family", "store_nbr", "date"])
    df["store_nbr"] = df["store_nbr"].astype("category")

    return df

def export_to_parquet(df: pd.DataFrame, filepath: str) -> None:
    carpeta_destino = os.path.dirname(filepath)
    os.makedirs(carpeta_destino, exist_ok=True)
    df.to_parquet(filepath, index=False)
    
    print(f"Data saved at: {filepath}")