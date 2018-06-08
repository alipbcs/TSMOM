"""
Interface & Implementation of Portfolio Strategies.
"""
import numpy as np
import pandas as pd
from main import database_manager
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from main import trade_rule

# LOOKBACK_PERIOD = 252


class TimeVaryingPortfolioStrategy(ABC):
    """
    Example Usage:
        st_1 = main.portfolio_strategy.ConstantVolatilityStrategy(dbm, df, 0.4)
        res_1 = st_1.compute_strategy()

        st_2 = main.portfolio_strategy.TSMOMStrategy(dbm, df, 0.4)
        res_2 = st_2.compute_strategy()
    """

    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252):
        self.dbm = dbm
        self.data = data
        self.n_t = None
        self.aggregated_assets = None
        self.table_in_assets = []
        self.sigma_target = sigma_target
        self.daily_ret = None
        self.volatility = None
        self.lookback_window = lookback

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
            tbl_name = self.data.index[t]
            df_curr, _ = self.dbm.get_table(tbl_name)

            if df_curr is not None:
                if df_curr.shape[0] < self.lookback_window + 1:
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

    def pre_strategy(self):
        """
        Prepares & computes the necessary variables needed before computing the strategy.
        """
        self.__aggregate_assets()

        self.daily_ret = self.aggregated_assets.pct_change()
        self.volatility = self.daily_ret.rolling(self.lookback_window).std() * np.sqrt(self.lookback_window)
        self.volatility[self.volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0

    @abstractmethod
    def compute_strategy(self):
        pass


class ConstantVolatilityStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252):
        super().__init__(dbm, data, sigma_target, lookback)

    def compute_strategy(self):
        super().pre_strategy()

        asset_weight = self.sigma_target / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        return portfolio_return, asset_weight


class TSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252):
        super().__init__(dbm, data, sigma_target, lookback)
        self.trade_rule = None

    def pre_strategy(self) -> pd.DataFrame:
        super().pre_strategy()

        # self.trade_rule = trade_rule.SIGN(self.aggregated_assets, self.daily_ret, 252)
        self.trade_rule = trade_rule.TREND(self.aggregated_assets, self.daily_ret, self.lookback_window)
        trade_rule_out = self.trade_rule.compute_rule()

        return trade_rule_out

    def compute_strategy(self):
        trade_rule_out = self.pre_strategy()

        asset_weight = self.sigma_target * trade_rule_out
        self.volatility.fillna(1.0, inplace=True)
        asset_weight = asset_weight / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        return portfolio_return, asset_weight


class CorrAdjustedTSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target, lookback: int = 252):
        super().__init__(dbm, data, sigma_target, lookback)
        self.trade_rule = None

    def pre_strategy(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        super().pre_strategy()

        # annual_ret = self.aggregated_assets.pct_change(periods=self.lookback_window)
        daily_ret = self.aggregated_assets.pct_change(periods=1)

        self.trade_rule = trade_rule.SIGN(self.aggregated_assets, self.daily_ret, 252)
        # self.trade_rule = trade_rule.TREND(self.aggregated_assets, self.daily_ret, self.lookback_window)
        trade_rule_out = self.trade_rule.compute_rule()

        return daily_ret, trade_rule_out

    def compute_strategy(self):
        daily_ret, trade_rule_out = self.pre_strategy()

        cf_list = []

        for t in range(daily_ret.shape[0]):
            if t < 34:
                cf_list.append(1)
                continue

            daily_ret_window = daily_ret.iloc[t - 34 + 1:t + 1]

            assets_present = daily_ret.columns[daily_ret.iloc[t].notnull()]

            if t % 100 == 0:
                print('Progress: {}%'.format(int(t * 100 / self.n_t.shape[0])))

            daily_ret_window_assets_present = daily_ret_window[assets_present]
            daily_ret_window_assets_present = daily_ret_window_assets_present.dropna(how='any')
            trade_rule_out_assets = trade_rule_out[assets_present]

            if daily_ret_window_assets_present.shape[0] < 2 or daily_ret_window_assets_present.shape[1] < 2:
                cf_list.append(1)
                continue

            asset_corr = daily_ret_window_assets_present.corr().values

            co_trade_rule = np.eye(*asset_corr.shape)

            for i in range(co_trade_rule.shape[0]):
                for j in range(i + 1, co_trade_rule.shape[1]):
                    # TODO
                    temp = trade_rule_out_assets.iloc[t].values
                    co_trade_rule[i, j] = temp[i] * temp[j]
                    co_trade_rule[j, i] = temp[i] * temp[j]

            N = asset_corr.shape[0]
            rho_bar = ((asset_corr * co_trade_rule).sum() - asset_corr.shape[0]) / (N * (N - 1))

            if 1 + (N - 1) * rho_bar == 0:
                print('Warning: Divide by zero.')
                cf_list.append(1)
                continue

            temp = N / (1 + ((N - 1) * rho_bar))

            if temp < 0:
                print('Warning: negative value encountered for taking square root.')
                cf_list.append(1)
                continue

            if np.isnan(temp) or np.isinf(temp):
                print('Infinity or NaN')
                cf_list.append(1)
                continue

            cf_t = np.sqrt(temp)
            cf_list.append(cf_t)

        asset_weight = self.sigma_target * trade_rule_out / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.mul(np.array(cf_list), axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)
        # portfolio_return = portfolio_return.div(self.n_t * np.array(cf_list), axis=0)

        return portfolio_return, asset_weight
