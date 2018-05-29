"""
Helper functions for statistical computations.
"""
from main import database_manager, finance_metrics
import numpy as np
import pandas as pd
import datetime
from typing import Tuple, Union, Dict, Optional
from statsmodels.stats.weightstats import DescrStatsW


def compute_summary_statistics(dbm: database_manager.DatabaseManager, tbl_name: str) -> Optional[Dict[str, Tuple]]:
    """
    Computes summary statistics for given table.
    :param dbm: A DatabaseManager instance.
    :param tbl_name: name of the table to compute monthly return for.
    :return: dictionary containing various statistics.
    """
    df, info, start_date = finance_metrics.compute_monthly_returns(dbm, tbl_name)

    if df is not None and info is not None:
        stat = {}

        dsw = DescrStatsW(df['Monthly_Return'].values)

        stat['table_name'] = tbl_name
        stat['contract_name'] = info[1]
        stat['type'] = info[3] if info[3] is not None else None
        stat['subtype'] = info[4] if info[4] is not None else None
        stat['start-date'] = start_date
        stat['ar'] = df['Monthly_Return'].mean() * 12
        stat['vol'] = df['Monthly_Return'].std() * np.sqrt(12)
        stat['t-stat'] = dsw.ttest_mean(alternative='larger')[0]
        stat['p-value'] = dsw.ttest_mean(alternative='larger')[1]
        stat['kurt'] = df['Monthly_Return'].kurt()
        stat['skew'] = df['Monthly_Return'].skew()

        return stat

    return None


