"""
Helper functions for financial computations.
"""
import numpy as np
import pandas as pd
from main import database_manager
from datetime import datetime
from typing import Optional
import pyfolio as pf
from scipy import stats


def compute_portfolio_performance(daily_weight, daily_ret, start_date='1980-02-29', end_date='2018-5-8'):
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')

    except ValueError:
        print('Wrong date format, should be Y-m-d')
        return

    daily_ret = daily_ret[daily_ret.index > start_date]
    daily_ret = daily_ret[daily_ret.index < end_date]

    if daily_ret.shape[0] < 0:
        raise ValueError('no data in specified time period.')
        return

    daily_weight = daily_weight[daily_weight.index > start_date]
    daily_weight = daily_weight[daily_weight.index < end_date]

    volatility = pf.timeseries.annual_volatility(daily_ret.apply(np.array).tz_localize('UTC'))

    turnover = daily_weight.resample('BM').last()[:-1].diff().abs().sum(axis=1).mean()

    performance_statistics = {}

    # performance_statistics['average_return'] = ((daily_ret.mean() + 1) ** 252 - 1.0) * 100
    performance_statistics['annual_return'] = pf.timeseries.annual_return(daily_ret.apply(np.array).tz_localize('UTC')) * 100
    performance_statistics['volatility'] = volatility * 100
    performance_statistics['Sharpe'] = pf.timeseries.sharpe_ratio(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['turnover'] = turnover
    performance_statistics['skew'] = stats.skew(daily_ret)
    performance_statistics['kurtosis'] = stats.kurtosis(daily_ret)
    performance_statistics['sortino'] = pf.timeseries.sortino_ratio(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['calmar'] = pf.timeseries.calmar_ratio(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['avg_leverage'] = daily_weight.abs().sum(axis=1).mean()
    performance_statistics['max_drawdown'] = pf.timeseries.max_drawdown(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['common_sense_ratio'] = pf.timeseries.common_sense_ratio(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['omega_ratio'] = pf.timeseries.omega_ratio(daily_ret.apply(np.array).tz_localize('UTC'))
    performance_statistics['tail_ratio'] = pf.timeseries.tail_ratio(daily_ret.apply(np.array).tz_localize('UTC'))

    return performance_statistics


def compute_annual_return_from_daily_return(dbm: database_manager.DatabaseManager, tbl: str, lookback: int = 252) -> Optional[pd.DataFrame]:
    """
    Computes annual returns given asset.
    :param lookback: amount of lookback period.
    :param tbl: table to compute annual returns for.
    :param dbm: a DatabaseManager instance.
    :return: Dataframe containing annual returns.
    """
    df, _ = dbm.get_table(tbl)
    annual_ret = pd.DataFrame()

    if tbl is None:
        return None

    df['PX_LAST_forward_filled'] = df['PX_LAST'].fillna(method='ffill')
    annual_ret['annual_ret'] = df['PX_LAST_forward_filled'].pct_change(periods=lookback)
    annual_ret['Dates'] = df.index
    annual_ret.set_index('Dates', inplace=True)

    df.drop('PX_LAST_forward_filled', axis=1, inplace=True)

    return annual_ret


def compute_compounded_monthly_return_from_daily_return(dbm: database_manager.DatabaseManager, tbl_name: str) -> \
        Optional[pd.DataFrame]:
    """
    Computes compounded monthly return.
    :param dbm: A DatabaseManager instance.
    :param tbl_name: name of the table to compute monthly return for.
    :return: monthly return
    """
    df, _ = dbm.get_table(tbl_name)

    if df is None:
        return None

    df['PX_LAST_forward_filled'] = df['PX_LAST'].fillna(method='ffill')

    tbl_daily_ret = df['PX_LAST_forward_filled'].pct_change()
    monthly_ret = tbl_daily_ret.resample('BM').apply(lambda x: (x + 1).cumprod()[-1] - 1.0)

    df.drop('PX_LAST_forward_filled', axis=1, inplace=True)

    return monthly_ret
