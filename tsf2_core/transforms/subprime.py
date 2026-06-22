"""SARIMAX subprime feature engineering transforms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsf2_core.constants import (
    OIL_PRICE_CUTOFF,
    OIL_REFERENCE_DATE,
    ONE_HOT_INT_COLUMNS,
)


def apply_subprime_transformations(datasets: dict) -> pd.DataFrame:
    """Apply subprime transformations: merge, impute, encode. No Box-Cox or HMV."""
    print("\n[TRANSFORM] Applying SARIMAX subprime transformations...")

    stores = datasets["stores"]
    holidays_events = datasets["holidays_events"]
    oil = datasets["oil"]
    transactions = datasets["transactions"]
    train = datasets["train"].drop("id", axis=1)

    print("  - Compressing integer and float columns...")
    for df in (stores, holidays_events, oil, transactions, train):
        for col in df:
            if df[col].dtype == "int64":
                if df[col].max() <= np.iinfo(np.int16).max:
                    if df[col].max() < np.iinfo(np.int8).max:
                        df[col] = df[col].astype("int8")
                    else:
                        df[col] = df[col].astype("int16")
            if df[col].dtype == "float64" and df[col].max() <= np.finfo(np.float32).max:
                df[col] = df[col].astype("float32")

    print("  - Merging datasets and imputing missing values...")
    oil.rename(columns={"dcoilwtico": "oilprice"}, inplace=True)

    train = train.merge(oil, how="left", on="date")
    train["oilprice"] = train["oilprice"].bfill()
    train = train.merge(transactions, how="left", on=["date", "store_nbr"])
    train = train.merge(stores, how="left", on="store_nbr")
    train = train.merge(holidays_events, how="left", on="date")
    train.fillna({"transactions": 0}, inplace=True)
    train.rename(columns={"type_x": "store_type", "type_y": "holiday_type"}, inplace=True)

    print("  - Engineering features...")
    train.loc[train["transferred"] == True, "holiday_type"] = (
        "Transferred" + train["holiday_type"]
    )
    train["ntl_holiday"] = (train["locale"] == "National").astype("int8")
    train["rgnl_holiday"] = (
        (train["locale"] == "Regional") & (train["locale_name"] == train["state"])
    ).astype("int8")
    train["lcl_holiday"] = (
        (train["locale"] == "Local") & (train["locale_name"] == train["city"])
    ).astype("int8")

    median_onpromotion = train.loc[train["onpromotion"] > 0, "onpromotion"].median()
    train["exists_promotion"] = train["onpromotion"].apply(lambda x: 1 if x > 0 else 0).astype(
        "int8"
    )
    train["onpromotion"] = train["onpromotion"].apply(
        lambda x: x if x > 0 else median_onpromotion
    )

    median_transactions = transactions.loc[
        transactions["transactions"] > 0, "transactions"
    ].median()
    transactions["exists_transaction"] = transactions["transactions"].apply(
        lambda x: 1 if x > 0 else 0
    ).astype("int8")
    transactions["transactions"] = transactions["transactions"].apply(
        lambda x: x if x > 0 else median_transactions
    )
    train = train.drop("transactions", axis=1)
    train = train.merge(transactions, how="left", on=["date", "store_nbr"])
    train.fillna({"transactions": 0, "exists_transaction": 0}, inplace=True)

    oil["oil_price_status"] = oil["oilprice"].apply(
        lambda x: 1 if x < OIL_PRICE_CUTOFF else 0
    )
    median_lowoilprice = oil.loc[
        (oil["date"] < OIL_REFERENCE_DATE) & (oil["oilprice"] <= OIL_PRICE_CUTOFF),
        "oilprice",
    ].median()
    oil["low_oil_price"] = oil["oilprice"].apply(
        lambda x: x if x <= OIL_PRICE_CUTOFF else median_lowoilprice
    )
    median_highoilprice = oil.loc[
        (oil["date"] < OIL_REFERENCE_DATE) & (oil["oilprice"] > OIL_PRICE_CUTOFF),
        "oilprice",
    ].median()
    oil["high_oil_price"] = oil["oilprice"].apply(
        lambda x: x if x > OIL_PRICE_CUTOFF else median_highoilprice
    )
    oil = oil.drop("oilprice", axis=1)

    train = train.drop("oilprice", axis=1)
    train = train.merge(oil, how="left", on="date")
    train["oil_price_status"] = train["oil_price_status"].bfill()
    train["low_oil_price"] = train["low_oil_price"].bfill()
    train["high_oil_price"] = train["high_oil_price"].bfill()

    print("  - One-hot encoding categorical features...")
    train.loc[
        (train["ntl_holiday"] == 0)
        & (train["rgnl_holiday"] == 0)
        & (train["lcl_holiday"] == 0),
        "holiday_type",
    ] = np.nan
    train = pd.get_dummies(train, columns=["holiday_type", "store_type"])
    train = train.reindex(columns=train.columns.union(ONE_HOT_INT_COLUMNS), fill_value=0)
    train[ONE_HOT_INT_COLUMNS] = train[ONE_HOT_INT_COLUMNS].astype("int8")
    train = train.drop(["locale", "locale_name", "transferred"], axis=1)

    print(f"✓ Subprime transformations complete. Dataset: {len(train)} rows")
    return train
