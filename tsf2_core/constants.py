"""Canonical constants for data processing and S3 layout."""

MARKER = "_COMPLETE"

SIGNIFICANT_EXOG = ["hmv", "exists_promotion", "exists_transaction"]

MODEL_FILENAME = "model.joblib"

DATASET_NAMES = ["holidays_events", "oil", "stores", "train", "transactions"]
REQUIRED_DAILY_DATASETS = ("holidays_events", "oil", "train", "transactions")

OIL_PRICE_CUTOFF = 71.5
OIL_REFERENCE_DATE = "2017-08-15"
TWO_YEAR_CUTOFF = "2015-08-15"

PERIOD_MAP = {
    0.25: "2017-05-15",
    0.5: "2017-02-15",
    1: "2016-08-15",
    1.5: "2016-02-15",
    2.5: "2015-02-15",
    3.5: "2014-02-15",
    4: "2013-01-01",
}

NON_TWO_YEAR_FAMILIES = {
    "BABY CARE": 1.5,
    "BOOKS": 0.5,
    "LAWN AND GARDEN": 0.5,
    "LIQUOR,WINE,BEER": 1,
    "MAGAZINES": 1.5,
    "AUTOMOTIVE": 4,
    "BEAUTY": 4,
    "BREAD/BAKERY": 4,
    "CLEANING": 4,
    "DAIRY": 3.5,
    "DELI": 4,
    "EGGS": 4,
    "FROZEN FOODS": 4,
    "GROCERY I": 4,
    "GROCERY II": 4,
    "LINGERIE": 4,
    "MEATS": 4,
    "PERSONAL CARE": 4,
    "POULTRY": 3.5,
    "PREPARED FOODS": 4,
    "SEAFOOD": 2.5,
    "SCHOOL AND OFFICE SUPPLIES": 1,
}

TWO_YEAR_FAMILIES = {
    "BEVERAGES",
    "CELEBRATION",
    "HARDWARE",
    "HOME AND KITCHEN I",
    "HOME AND KITCHEN II",
    "HOME APPLIANCES",
    "HOME CARE",
    "LADIESWEAR",
    "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS",
    "PRODUCE",
}

NON_TWO_YEAR_STORES = {21: 1, 22: 1.5, 25: 0.5, 42: 1.5, 52: 0.25, 53: 1}
TWO_YEAR_STORES = {*range(1, 21), 23, 24, *range(26, 42), *range(43, 52), 54}

EXOG_FEATURES = {
    feature: "mean"
    for feature in [
        "sales",
        "onpromotion",
        "transactions",
        "ntl_holiday",
        "rgnl_holiday",
        "lcl_holiday",
        "hmv",
        "exists_promotion",
        "exists_transaction",
        "oil_price_status",
        "low_oil_price",
        "high_oil_price",
        "holiday_type_Additional",
        "holiday_type_Bridge",
        "holiday_type_Event",
        "holiday_type_Holiday",
        "holiday_type_Transfer",
        "holiday_type_TransferredHoliday",
        "holiday_type_Work Day",
    ]
}

ONE_HOT_INT_COLUMNS = [
    "holiday_type_Additional",
    "holiday_type_Bridge",
    "holiday_type_Event",
    "holiday_type_Holiday",
    "holiday_type_Transfer",
    "holiday_type_TransferredHoliday",
    "holiday_type_Work Day",
    "store_type_A",
    "store_type_B",
    "store_type_C",
    "store_type_D",
    "store_type_E",
]

# S3 prefixes
RAW_HISTORICAL_PREFIX = "raw/historical/"
SUBPRIME_HISTORICAL_PREFIX = "processed/sarimax-subprime/historical/"
PRIME_HISTORICAL_PREFIX = "processed/sarimax-prime/historical/"
TIMESERIES_FAMILY_HISTORICAL_PREFIX = "processed/sarimax-prime/historical/family/"
TIMESERIES_STORE_HISTORICAL_PREFIX = "processed/sarimax-prime/historical/store/"
FAMILIES_MAPPING_KEY = "processed/sarimax-prime/historical/families_mapping.json"

SUBPRIME_BIWEEKLY_PREFIX = "processed/sarimax-subprime/biweekly/"
PRIME_BIWEEKLY_PREFIX = "processed/sarimax-prime/biweekly/"
SARIMAX_MODEL_BIWEEKLY_PREFIX = "sarimax/biweekly/"
SARIMAX_MODEL_HISTORICAL_PREFIX = "sarimax/historical/"
XGBOOST_PRIME_BIWEEKLY_PREFIX = "processed/xgboost-prime/biweekly/"
XGBOOST_MODEL_BIWEEKLY_PREFIX = "xgboost/biweekly/"


def biweek_data_prefix(base_prefix: str, year: int, biweek_num: int) -> str:
    return f"{base_prefix}{year}/BW-{biweek_num}/"
