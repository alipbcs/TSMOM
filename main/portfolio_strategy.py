"""
Helper functions for financial computations.
"""
import numpy as np
import pandas as pd
from main import database_manager
from abc import ABC, abstractmethod


LOOKBACK_PERIOD = 252


class TimeVaryingPortfolioStrategy(object):
    """
    Example Usage:
        st_1 = main.portfolio_strategy.ConstantVolatilityStrategy(dbm, df, 0.4)
        res_1 = st_1.compute_strategy()

        st_2 = main.portfolio_strategy.TSMOMStrategy(dbm, df, 0.4)
        res_2 = st_2.compute_strategy()
    """
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, verbose=False):
        self.dbm = dbm
        self.data = data
        self.n_t = None
        self.aggreagated_assets = None
        self.table_in_assets = []
        self.verbose = verbose
        self.sigma_target = sigma_target

    def aggregate_assets(self):
        """
        Joins PX_LAST of all assets.
        """
        agg_assets, _ = self.dbm.get_table(self.data.index[0])

        agg_assets['PX_LAST_' + self.data.index[0]] = agg_assets['PX_LAST']
        agg_assets['PX_LAST_' + self.data.index[0]].fillna(method='ffill', inplace=True)
        agg_assets = agg_assets[['PX_LAST_' + self.data.index[0]]]

        table_present = np.zeros(self.data.shape[0])
        table_present[0] = 1
        i = 1

        for t in range(1, self.data.shape[0]):
            if self.verbose:
                print('Progress: {0:.2f}%'.format(i / self.data.shape[0]))

            tbl_name = self.data.index[t]
            df_curr, _ = self.dbm.get_table(tbl_name)

            if df_curr is not None:
                if df_curr.shape[0] < LOOKBACK_PERIOD + 1:
                    i += 1
                    continue

                table_present[i] = 1

                df_curr['PX_LAST_' + tbl_name] = df_curr['PX_LAST']
                df_curr = df_curr[['PX_LAST_' + tbl_name]]

                agg_assets = agg_assets.join(df_curr, on='Dates', how='outer', sort=True)
                agg_assets.fillna(method='ffill', inplace=True)

            i += 1

        self.n_t = agg_assets.notna().sum(axis=1)
        self.aggreagated_assets = agg_assets
        self.table_in_assets = table_present

    def compute_annualized_returns(self):
        """
        Computes annualized retrurn and rollign standard deviation for lookback period.
        """
        i = 0

        for t in range(self.data.shape[0]):
            if self.verbose:
                print('Progress: {0:.2f}%'.format(i / self.data.shape[0]))

            tbl_name = self.data.index[t]

            if self.table_in_assets[i] == 1:
                curr_daily_ret = 'daily_ret_' + tbl_name
                curr_last = self.aggreagated_assets['PX_LAST_' + tbl_name]
                curr_annual = 'annual_ret_' + tbl_name
                curr_std = 'rolling_std_' + tbl_name

                self.aggreagated_assets[curr_daily_ret] = curr_last.pct_change()
                self.aggreagated_assets[curr_annual] = curr_last.pct_change(periods=LOOKBACK_PERIOD)
                self.aggreagated_assets[curr_std] = self.aggreagated_assets[curr_daily_ret].rolling(LOOKBACK_PERIOD).std() \
                                                    * np.sqrt(LOOKBACK_PERIOD)
                self.aggreagated_assets[curr_std][self.aggreagated_assets[curr_std] < (self.sigma_target / 10.0)] = self.sigma_target / 10.0

            i += 1

    @abstractmethod
    def compute_strategy(self):
        self.aggregate_assets()
        self.compute_annualized_returns()


class ConstantVolatilityStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self):
        super().compute_strategy()

        i = 0

        for t in range(self.data.shape[0]):
            if self.verbose:
                print('Progress: {0:.2f}%'.format(i / self.data.shape[0]))

            tbl_name = self.data.index[t]

            if self.table_in_assets[i] == 1:
                curr_daily_ret = 'daily_ret_' + tbl_name
                curr_last = 'PX_LAST_' + tbl_name
                curr_std = 'rolling_std_' + tbl_name
                curr_annual = 'annual_ret_' + tbl_name

                self.aggreagated_assets['ret_cvol_' + tbl_name] = self.sigma_target * self.aggreagated_assets[curr_daily_ret] /         \
                                                                  self.aggreagated_assets[curr_std]
                self.aggreagated_assets.drop(labels=[curr_daily_ret, curr_last, curr_std, curr_annual], axis=1, inplace=True)

            i += 1

        self.aggreagated_assets.shift(1)
        self.aggreagated_assets['sum'] = self.aggreagated_assets.sum(axis=1)
        self.aggreagated_assets['result'] = self.aggreagated_assets['sum'].div(self.n_t, axis=0)

        return self.aggreagated_assets[['result']]


class TSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self):
        super().compute_strategy()

        i = 0

        for t in range(self.data.shape[0]):
            if self.verbose:
                print('Progress: {0:.2f}%'.format(i / self.data.shape[0]))

            tbl_name = self.data.index[t]

            if self.table_in_assets[i] == 1:
                curr_daily_ret = 'daily_ret_' + tbl_name
                curr_last = 'PX_LAST_' + tbl_name
                curr_annual = 'annual_ret_' + tbl_name
                curr_std = 'rolling_std_' + tbl_name
                curr_cvol = 'ret_cvol_' + tbl_name

                self.aggreagated_assets[curr_annual] = (self.aggreagated_assets[curr_annual] > 0)
                self.aggreagated_assets[curr_annual] *= 2
                self.aggreagated_assets[curr_annual] -= 1
                self.aggreagated_assets[curr_cvol] = self.sigma_target * self.aggreagated_assets[curr_daily_ret] / self.aggreagated_assets[curr_std]
                self.aggreagated_assets['TSMOM_' + tbl_name] = self.aggreagated_assets[curr_cvol] * self.aggreagated_assets[curr_annual]
                self.aggreagated_assets.drop(labels=[curr_daily_ret, curr_last, curr_annual, curr_std, curr_cvol], axis=1, inplace=True)

            i += 1

        self.aggreagated_assets.shift(1)
        self.aggreagated_assets['sum'] = self.aggreagated_assets.sum(axis=1)
        self.aggreagated_assets['result'] = self.aggreagated_assets['sum'].div(self.n_t, axis=0)

        return self.aggreagated_assets[['result']]
