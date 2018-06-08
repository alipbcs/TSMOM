"""
Helper functions for financial computations.
"""
import numpy as np
import pandas as pd
from main import database_manager
from typing import Tuple
from abc import ABC, abstractmethod


LOOKBACK_PERIOD_ANNUAL = 252


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

    def __aggregate_assets(self):
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
                if df_curr.shape[0] < LOOKBACK_PERIOD_ANNUAL + 1:
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

    def __compute_annualized_returns(self, sd_window) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Computes annualized return and rolling standard deviation for lookback period.
        :return: tuple of daily return, annual return and rolling standard deviation of all assets. 
        """
        daily_ret = self.aggregated_assets.pct_change()
        annual_ret = self.aggregated_assets.pct_change(periods=LOOKBACK_PERIOD_ANNUAL)
        rolling_std = daily_ret.rolling(sd_window).std() * np.sqrt(LOOKBACK_PERIOD_ANNUAL)
        rolling_std[rolling_std < self.sigma_target / 10.0] = self.sigma_target / 10.0

        return daily_ret, annual_ret, rolling_std

    def pre_strategy(self, sd_window) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepares & computes the necessary variables needed before computating the strategy.  
        :return: 
        """
        self.__aggregate_assets()
        daily_ret, annual_ret, rolling_std = self.__compute_annualized_returns(sd_window)

        return daily_ret, annual_ret, rolling_std

    @abstractmethod
    def compute_strategy(self, sd_window):
        pass


class ConstantVolatilityStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self, sd_window):
        daily_ret, annual_ret, rolling_std = super().pre_strategy(sd_window)

        asset_weight = self.sigma_target / rolling_std
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * daily_ret).sum(axis=1)
        # portfolio_return = portfolio_return.div(self.n_t, axis=0)

        return portfolio_return, asset_weight, daily_ret


class TSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self, sd_window):
        daily_ret, annual_ret, rolling_std = super().pre_strategy(sd_window)

        annual_ret = annual_ret > 0
        annual_ret = (annual_ret * 2) - 1

        asset_weight = self.sigma_target * annual_ret / rolling_std
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * daily_ret).sum(axis=1)
        # portfolio_return = portfolio_return.div(self.n_t, axis=0)

        return portfolio_return, asset_weight, daily_ret


class CorrAdjustedTSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target):
        super().__init__(dbm, data, sigma_target)

    def compute_strategy(self, sd_window):
        daily_ret, annual_ret, rolling_std = super().pre_strategy(sd_window)

        annual_ret_signed = (annual_ret > 0)
        annual_ret_signed = (annual_ret_signed * 2) - 1

        cf_list = []

        for t in range(annual_ret.shape[0]):
            curr_date = annual_ret.index[t]
            annual_ret_upto_curr = annual_ret[annual_ret.index <= curr_date]

            assets_present = annual_ret.columns[annual_ret.iloc[t].notnull()]

            if t % 100 == 0:
                print('Progress: {0:.2f}%'.format(int(t * 100 / self.n_t.shape[0])))

            annual_ret_upto_curr_assets = annual_ret_upto_curr[assets_present]
            annual_ret_upto_curr_assets = annual_ret_upto_curr_assets.dropna(how='all')

            if annual_ret_upto_curr_assets.shape[0] < 2 or annual_ret_upto_curr_assets.shape[1] < 2:
                cf_list.append(1)
                continue

            annual_ret_upto_curr_assets_signed = annual_ret_upto_curr_assets > 0
            annual_ret_upto_curr_assets_signed *= 2
            annual_ret_upto_curr_assets_signed -= 1

            asset_corr = annual_ret_upto_curr_assets.corr().values

            co_sign = np.eye(*asset_corr.shape)

            for i in range(co_sign.shape[0]):
                for j in range(i + 1, co_sign.shape[1]):
                    temp = annual_ret_upto_curr_assets_signed.iloc[-1].values
                    co_sign[i, j] = temp[i] * temp[j]
                    co_sign[j, i] = temp[i] * temp[j]

            # N = self.n_t[t]
            N = asset_corr.shape[0]
            rho_bar = ((asset_corr * co_sign).sum() - asset_corr.shape[0]) / (N * (N - 1))
            temp = N / (1 + ((N - 1) * rho_bar))

            if temp < 0:
                print('Warning: negative value encountered for taking square root.')
                cf_list.append(1)
                continue

            cf_t = np.sqrt(temp)
            cf_list.append(cf_t)

        asset_weight = self.sigma_target * annual_ret_signed / rolling_std
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.mul(np.array(cf_list), axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * daily_ret).sum(axis=1)
        # portfolio_return = portfolio_return.div(self.n_t * np.array(cf_list), axis=0)

        return portfolio_return, asset_weight, daily_ret
