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
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float):
        self.dbm = dbm
        self.data = data
        self.n_t = None
        self.aggregated_assets = None
        self.table_in_assets = []
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
            # if self.verbose:
            #     print('Progress: {0:.2f}%'.format(i / self.data.shape[0]))

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
        self.aggregated_assets = agg_assets
        self.table_in_assets = table_present

    def compute_annualized_returns(self):
        """
        Computes annualized retrurn and rollign standard deviation for lookback period.
        """
        daily_ret = self.aggregated_assets.pct_change()
        annual_ret = self.aggregated_assets.pct_change(periods=LOOKBACK_PERIOD)
        rolling_std = daily_ret.rolling(LOOKBACK_PERIOD).std() * np.sqrt(LOOKBACK_PERIOD)
        rolling_std[rolling_std < self.sigma_target / 10.0] = self.sigma_target / 10.0

        return daily_ret, annual_ret, rolling_std

    def pre_strategy(self):
        self.aggregate_assets()
        daily_ret, annual_ret, rolling_std = self.compute_annualized_returns()

        return daily_ret, annual_ret, rolling_std

    @abstractmethod
    def compute_strategy(self):
        pass


class ConstantVolatilityStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self):
        daily_ret, annual_ret, rolling_std = super().pre_strategy()

        asset_weight = self.sigma_target / rolling_std
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * daily_ret).sum(axis=1)
        portfolio_return = portfolio_return.div(self.n_t, axis=0)

        return portfolio_return


class TSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self):
        daily_ret, annual_ret, rolling_std = super().pre_strategy()

        annual_ret = annual_ret > 0
        annual_ret = (annual_ret * 2) - 1

        asset_weight = self.sigma_target * annual_ret / rolling_std
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * daily_ret).sum(axis=1)
        portfolio_return = portfolio_return.div(self.n_t, axis=0)

        return portfolio_return
