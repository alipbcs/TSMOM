"""
Interface & Implementation of Portfolio Strategies.
"""
import numpy as np
import pandas as pd
from main import database_manager
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from main import trade_rule
from main import volatility_estimator



class TimeVaryingPortfolioStrategy(ABC):
    """
    Example Usage:
        st_1 = main.portfolio_strategy.ConstantVolatilityStrategy(dbm, df, 0.4)
        res_1 = st_1.compute_strategy()

        st_2 = main.portfolio_strategy.TSMOMStrategy(dbm, df, 0.4)
        res_2 = st_2.compute_strategy()
    """

    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252,
                 volatilty_estimator: str = 'sd'):
        self.dbm = dbm
        self.data = data
        self.n_t = None
        self.aggregated_assets = None
        self.table_in_assets = []
        self.sigma_target = sigma_target
        self.daily_ret = None
        self.volatility = None
        self.volatility_estimator = volatilty_estimator
        self.lookback_window_vol = lookback

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
                if df_curr.shape[0] < self.lookback_window_vol + 1:
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

    def __compute_volatility(self):
        if str.lower(self.volatility_estimator) == 'sd':
            vol_est = volatility_estimator.VolatilitySD(self.daily_ret, self.lookback_window_vol, self.sigma_target)
            return vol_est.compute()

        elif str.lower(self.volatility_estimator) == 'yz':
            vol_df = pd.DataFrame(index=self.aggregated_assets.index)

            for asset in self.aggregated_assets.columns:
                tbl_name = str.replace(asset, 'PX_LAST_', '')
                df_curr, _ = self.dbm.get_table(tbl_name)

                df_curr = df_curr.reindex(self.aggregated_assets.index, method='ffill')

                vol_est = volatility_estimator.VolatilityYZ(self.lookback_window_vol, self.sigma_target, df_curr)
                vol_df[asset] = vol_est.compute()

            return vol_df
        else:
            raise ValueError('Unsupported Volatility Estimator.')

    def pre_strategy(self):
        """
        Prepares & computes the necessary variables needed before computing the strategy.
        """
        self.__aggregate_assets()

        self.daily_ret = self.aggregated_assets.pct_change()
        self.volatility = self.__compute_volatility()
        # self.volatility = self.daily_ret.rolling(self.lookback_window_vol).std() * np.sqrt(self.lookback_window_vol)
        # self.volatility[self.volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0

    @abstractmethod
    def compute_strategy(self):
        pass


class ConstantVolatilityStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252,
                 volatilty_estimator: str = 'sd'):

        super().__init__(dbm, data, sigma_target, lookback, volatilty_estimator)

    def compute_strategy(self):
        super().pre_strategy()

        asset_weight = self.sigma_target / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        return portfolio_return, asset_weight


class TSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252,
                 volatilty_estimator: str = 'sd', trade_rule_name: str = 'SIGN'):

        super().__init__(dbm, data, sigma_target, lookback, volatilty_estimator)
        self.trade_rule_name = trade_rule_name
        self.lookback = lookback

    def pre_strategy(self) -> pd.DataFrame:
        super().pre_strategy()

        rule = None

        if str.lower(self.trade_rule_name) == 'sign':
            rule = trade_rule.SIGN(self.aggregated_assets, self.daily_ret, 252)
        elif str.lower(self.trade_rule_name) == 'trend':
            rule = trade_rule.TREND(self.aggregated_assets, self.daily_ret, self.lookback)
        else:
            raise ValueError('Unsupported trade rule.')

        trade_rule_out = rule.compute_rule()

        return trade_rule_out

    def compute_strategy(self):
        trade_rule_out = self.pre_strategy()

        asset_weight = self.sigma_target * trade_rule_out
        asset_weight = asset_weight / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)

        asset_weight_bm = asset_weight.resample('BM').last()
        asset_weight_bm = asset_weight_bm.resample('B').last()
        asset_weight_bm.fillna(method='ffill', inplace=True)

        asset_weight_bm = asset_weight_bm.loc[asset_weight_bm.index.intersection(asset_weight.index)]
        # asset_weight = asset_weight.shift(1)

        # portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)
        portfolio_return = (asset_weight_bm * self.daily_ret).sum(axis=1)

        return portfolio_return, asset_weight_bm


class CorrAdjustedTSMOMStrategy(TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager, data: pd.DataFrame, sigma_target: float, lookback: int = 252,
                 volatilty_estimator: str = 'sd', lookback_corr:int = 34, trade_rule_name: str ='SIGN'):

        super().__init__(dbm, data, sigma_target, lookback, volatilty_estimator)
        self.lookback = lookback
        self.lookback_corr = lookback_corr
        self.trade_rule_name = trade_rule_name

    def pre_strategy(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        super().pre_strategy()

        rule = None

        if str.lower(self.trade_rule_name) == 'sign':
            rule = trade_rule.SIGN(self.aggregated_assets, self.daily_ret, 252)
        elif str.lower(self.trade_rule_name) == 'trend':
            rule = trade_rule.TREND(self.aggregated_assets, self.daily_ret, self.lookback)
        else:
            raise ValueError('Unsupported trade rule.')

        trade_rule_out = rule.compute_rule()

        return trade_rule_out

    def compute_strategy(self):
        trade_rule_out = self.pre_strategy()

        cf_list = []

        for t in range(self.daily_ret.shape[0]):
            if t < self.lookback_corr:
                cf_list.append(1)
                continue

            daily_ret_window = self.daily_ret.iloc[t - self.lookback_corr + 1:t + 1]

            assets_present = self.daily_ret.columns[self.daily_ret.iloc[t].notnull()]

            if t % 100 == 0:
                print('Progress: {}%'.format(int(t * 100 / self.n_t.shape[0])))

            daily_ret_window_assets_present = daily_ret_window[assets_present]
            daily_ret_window_assets_present = daily_ret_window_assets_present.dropna(how='any')
            non_zero_columns = daily_ret_window_assets_present.columns[(daily_ret_window_assets_present == 0).all()]
            daily_ret_window_assets_present = daily_ret_window_assets_present[non_zero_columns]

            trade_rule_out_assets = trade_rule_out[assets_present]

            if daily_ret_window_assets_present.shape[0] < 5 or daily_ret_window_assets_present.shape[1] < 5:
                cf_list.append(1)
                continue

            if (daily_ret_window_assets_present == 0).all().sum() > 0:
                print('Warning: ALL ZERO.')

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

            temp = N / (1 + ((N - 1) * rho_bar))

            if temp < 0:
                print('Warning: negative value encountered for taking square root.')
                cf_list.append(1)
                continue

            if np.isnan(temp) or np.isinf(temp):
                print('Infinity or NaN.')
                cf_list.append(1)
                continue

            cf_t = np.sqrt(temp)
            cf_list.append(cf_t)

        asset_weight = self.sigma_target * trade_rule_out / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.mul(np.array(cf_list), axis=0)
        #
        # asset_weight = asset_weight.shift(1)

        asset_weight_bm = asset_weight.resample('BM').last()
        asset_weight_bm = asset_weight_bm.resample('B').last()
        asset_weight_bm.fillna(method='ffill', inplace=True)

        # portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)
        portfolio_return = (asset_weight_bm * self.daily_ret).sum(axis=1)

        return portfolio_return, asset_weight_bm
