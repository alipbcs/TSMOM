"""
Helper functions for financial computations.
"""
import pandas as pd
from main import database_manager
import datetime
from typing import Dict, Union, Tuple

LOOKBACK_PERIOD = 252


def compute_annual_returns_from_daily_return(dbm: database_manager.DatabaseManager, for_bloom: bool) -> Dict[str, pd.DataFrame]:
    """
    Computes annual returns for all assets.
    :param for_bloom: whether it is intended for bloom datasets or quandl.
    :param dbm: a DatabaseManager instance.
    :return: dictionary containing annaul returns for bloom datasets
    """
    d = {}

    table_names = dbm.bloom_dataset_names if for_bloom else dbm.quandl_dataset_names

    for tbl_name in table_names:
        df, info = dbm.get_table(tbl_name)

        if df is not None:
            df['annual_ret'] = df['PX_LAST'].pct_change(periods=LOOKBACK_PERIOD)
            df['annual_ret_sign'] = (df['annual_ret'] > 0)
            df['annual_ret_sign'] *= 2
            df['annual_ret_sign'] -= 1
            d[tbl_name] = df[['annual_ret', 'annual_ret_sign']]

    return d


def compute_monthly_returns(dbm: database_manager.DatabaseManager, tbl_name: str) -> \
        Union[Tuple[pd.DataFrame, Tuple[str, str, str, str, str], datetime.datetime], Tuple[None, None]]:
    """
    Computes compounded return for a month.
    :param dbm: A DatabaseManager instance.
    :param tbl_name: name of the table to compute monthly return for.
    :return: tuple consisting of (monthly return, info for table, first day the asset was traded).
    """
    tbl, info = dbm.get_table(tbl_name)

    if tbl is None:
        return None, None

    tbl.dropna(axis=0, inplace=True)

    first_date = tbl.index[0]
    last_date = tbl.index[-1]
    prev_month = first_date.month

    row_idx = 0
    curr_date, prev_date = None, None

    monthly_returns = []
    daily_ret = 0
    monthly_ret = 0

    while curr_date != last_date:
        row_idx += 1

        curr_date = tbl.index[row_idx]

        curr_month = curr_date.month

        curr_price = tbl.iloc[row_idx]['PX_LAST']
        prev_price = tbl.iloc[row_idx - 1]['PX_LAST']

        if curr_price == 0:
            daily_ret = 0
        elif prev_price == 0:
            daily_ret = tbl.iloc[row_idx - 2]['PX_LAST']
        else:
            daily_ret = (curr_price / prev_price) - 1.0

        monthly_ret = monthly_ret * (daily_ret + 1) if monthly_ret != 0 else daily_ret + 1

        if curr_month != prev_month:
            # remove compounding of last daily return
            monthly_ret /= (daily_ret + 1)

            monthly_returns.append((prev_date, monthly_ret - 1))

            # reset for next month
            monthly_ret = daily_ret + 1

        prev_month = curr_month
        prev_date = curr_date

    df = pd.DataFrame(monthly_returns, columns=['Dates', 'Monthly_Return'])
    df.set_index('Dates', inplace=True)

    return df, info, first_date
