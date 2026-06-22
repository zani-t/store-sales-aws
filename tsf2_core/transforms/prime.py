"""SARIMAX prime transforms (Box-Cox and holiday mean variation)."""

from __future__ import annotations

import pandas as pd
from scipy.stats import boxcox


def fit_prime_transform(data: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """Fit Box-Cox lambdas and compute HMV values (retraining / bootstrap path)."""
    print("\n[TRANSFORM] Applying SARIMAX prime transformations (fit)...")

    train = data.copy()
    lmbda_sales = boxcox(train.loc[train["sales"] > 0, "sales"])[1]
    lmbda_onpromotion = boxcox(train.loc[train["onpromotion"] > 0, "onpromotion"])[1]
    lmbda_transactions = boxcox(train.loc[train["transactions"] > 0, "transactions"])[1]

    train["onpromotion"] = boxcox(train["onpromotion"] + 0.01, lmbda_onpromotion)
    train["transactions"] = boxcox(train["transactions"] + 0.01, lmbda_transactions)
    train["sales"] = boxcox(train["sales"] + 0.01, lmbda_sales)

    print("  - Computing holiday mean variations...")
    ma = train[["date", "sales"]].groupby(["date"]).agg({"sales": "mean"})
    ma = pd.DataFrame(
        ma.rolling(window=15, min_periods=1).mean().values, columns=["ma15"]
    ).set_index(ma.index)
    train = train.merge(ma, how="left", on="date")
    train["hmv"] = 0.0
    hmvs: dict = {}
    for holiday in train["description"].unique():
        df = train.loc[
            train["description"] == holiday, ["date", "ma15", "sales"]
        ].groupby(["date", "ma15"], as_index=False).agg(sales=("sales", "mean"))
        hmv = float((df["sales"] - df["ma15"]).mean())
        hmvs[holiday] = hmv
        train.loc[train["description"] == holiday, "hmv"] = (
            (
                (train["ntl_holiday"] == 1)
                | (train["rgnl_holiday"] == 1)
                | (train["lcl_holiday"] == 1)
            ).astype("int8")
            * hmv
        )

    train = train.drop(["description", "ma15"], axis=1)
    lambdas = {
        "lmbda_sales": float(lmbda_sales),
        "lmbda_onpromotion": float(lmbda_onpromotion),
        "lmbda_transactions": float(lmbda_transactions),
    }
    print(f"✓ Prime transformations complete. Final dataset: {len(train)} rows")
    return train, lambdas, hmvs


def apply_prime_transform(
    data: pd.DataFrame,
    lambdas: dict,
    hmvs: dict,
    *,
    rolling_window: int = 15,
    min_periods: int = 15,
) -> pd.DataFrame:
    """Apply stored Box-Cox lambdas and HMV values (evaluation path)."""
    print("\n[TRANSFORM] Applying SARIMAX prime transformations (apply)...")

    train = data.copy()
    train["onpromotion"] = boxcox(train["onpromotion"] + 0.01, lambdas["lmbda_onpromotion"])
    train["transactions"] = boxcox(train["transactions"] + 0.01, lambdas["lmbda_transactions"])
    train["sales"] = boxcox(train["sales"] + 0.01, lambdas["lmbda_sales"])

    ma_col = f"ma{rolling_window}"
    ma = train[["date", "sales"]].groupby(["date"]).agg({"sales": "mean"})
    ma = pd.DataFrame(
        ma.rolling(window=rolling_window, min_periods=min_periods).mean().values,
        columns=[ma_col],
    ).set_index(ma.index)
    train = train.merge(ma, how="left", on="date")
    train["hmv"] = 0.0
    for holiday in train["description"].unique():
        df = train.loc[
            train["description"] == holiday, ["date", ma_col, "sales"]
        ].groupby(["date", ma_col], as_index=False).agg(sales=("sales", "mean"))
        hmv = hmvs.get(holiday, float((df["sales"] - df[ma_col]).mean()))
        train.loc[train["description"] == holiday, "hmv"] = (
            (
                (train["ntl_holiday"] == 1)
                | (train["rgnl_holiday"] == 1)
                | (train["lcl_holiday"] == 1)
            ).astype("int8")
            * hmv
        )

    train = train.drop(["description", ma_col], axis=1)
    print(f"✓ Prime transformations complete. Final dataset: {len(train)} rows")
    return train
