import pandas as pd
import numpy as np
import os

PATH_SALES_DATA = "data/raw/train.csv"
PATH_STORES_DATA = "data/raw/stores.csv"
PATH_HOLIDAYS_DATA = "data/raw/holidays_events.csv"
PATH_TRANSACTIONS_DATA = "data/raw/transactions.csv"
PATH_OIL_DATA = "data/raw/oil.csv"

PATH_OUTPUT = "data/processed/df_train.parquet"

LAGS = [15, 16, 17, 18, 19, 20, 30, 60, 90, 364, 365]


def prep_train_df(
    df: pd.DataFrame, lags: list[int], drop_lag_nan: bool = True
) -> pd.DataFrame:
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

    df = fulfill_sales(df)

    df = prep_train_get_lags(df, lags)

    if drop_lag_nan:
        max_lag = max(lags)
        df = df.dropna(subset=[f"sales_lag{max_lag}"]).reset_index(drop=True)

    # Gets sales types for each family
    is_decimal = df["sales"] % 1 != 0
    family_has_decimals = is_decimal.groupby(df["family"], observed=True).any()
    df["is_int_sales"] = df["family"].map(~family_has_decimals)

    return df


def fulfill_sales(df: pd.DataFrame):
    """Fulfill days without sales"""

    df_stores = pd.DataFrame({"store_nbr": df["store_nbr"].unique()})
    df_families = pd.DataFrame({"family": df["family"].unique()})
    df_dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max())})

    df_calendar = df_stores.merge(df_families, how="cross").merge(df_dates, how="cross")

    df = df_calendar.merge(df, on=["store_nbr", "family", "date"], how="left")

    df["sales_missing"] = df["sales"].isna().astype(int)

    df["sales"] = df["sales"].fillna(0)
    df["onpromotion"] = df["onpromotion"].fillna(0)
    return df


def prep_train_get_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Computes sales lags grouped by store and product family."""

    df = df.sort_values(["family", "store_nbr", "date"]).reset_index(drop=True)
    for i in lags:
        df[f"sales_lag{i}"] = df.groupby(["family", "store_nbr"])["sales"].shift(i)
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
    holidays_events = holidays_events[holidays_events["transferred"] == False]

    return holidays_events


def prep_joins(
    df_sales: pd.DataFrame,
    df_stores: pd.DataFrame,
    df_holiday_events: pd.DataFrame,
    df_oil: pd.DataFrame,
    df_transactions: pd.DataFrame,
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

    df["store_nbr"] = df["store_nbr"].astype("category")
    df = df.sort_values(["family", "store_nbr", "date"])

    return df


def export_to_parquet(df: pd.DataFrame, filepath: str) -> None:
    folder = os.path.dirname(filepath)
    os.makedirs(folder, exist_ok=True)
    df.to_parquet(filepath, index=False)

    print(f"Data saved at: {filepath}")


def build_features(
    train_raw,
    stores_raw,
    holidays_raw,
    transactions_raw,
    oil_raw,
    lags: list[int],
    drop_lag_nan: bool = True,
):
    print("Processing data...")
    train = prep_train_df(train_raw, lags, drop_lag_nan)
    stores = prep_stores_df(stores_raw)
    holidays = prep_holiday_df(holidays_raw)
    transactions = prep_transactions_df(transactions_raw)
    oil = prep_oil_df(oil_raw)

    print("Joining tables...")
    df = prep_joins(train, stores, holidays, oil, transactions)
    return df


def main():
    print("Initializing data preparation...")

    train_raw = pd.read_csv(PATH_SALES_DATA)
    stores_raw = pd.read_csv(PATH_STORES_DATA)
    holidays_raw = pd.read_csv(PATH_HOLIDAYS_DATA)
    transactions_raw = pd.read_csv(PATH_TRANSACTIONS_DATA)
    oil_raw = pd.read_csv(PATH_OIL_DATA)

    df_final = build_features(
        train_raw,
        stores_raw,
        holidays_raw,
        transactions_raw,
        oil_raw,
        LAGS,
        drop_lag_nan=True,
    )

    export_to_parquet(df_final, PATH_OUTPUT)
    print("Success!")


if __name__ == "__main__":
    main()
