import numpy as np
import pandas as pd
from main import database_manager
from main.portfolio_strategy import time_varying
from main import trade_rule


class TSMOMStrategy(time_varying.TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager,
                 data: pd.DataFrame,
                 sigma_target: float,
                 lookback_vol: int = 252,
                 volatilty_estimator: str = 'sd',
                 trade_rule_name: str = 'SIGN',
                 lookback_trend: int = 34,
                 include_transaction: bool = False):

        super().__init__(dbm, data, sigma_target, lookback_vol, volatilty_estimator, include_transaction)
        self.trade_rule_name = trade_rule_name
        self.lookback_trend = lookback_trend

    def pre_strategy(self):
        super().pre_strategy()

        rule = None

        if str.lower(self.trade_rule_name) == 'sign':
            rule = trade_rule.SIGN(self.aggregated_assets, self.daily_ret, 252)
        elif str.lower(self.trade_rule_name) == 'trend':
            rule = trade_rule.TREND(self.aggregated_assets, self.daily_ret, self.lookback_trend)
        else:
            raise ValueError('Unsupported trade rule.')

        self.trade_rule_out = rule.compute_rule()

    def compute_strategy(self):
        self.pre_strategy()

        asset_weight = self.sigma_target * self.trade_rule_out / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)

        asset_weight_bm = asset_weight.resample('BM').mean()
        asset_weight_bm = asset_weight_bm.resample('B').last()
        asset_weight_bm = asset_weight_bm.fillna(method='ffill')

        asset_weight_bm = asset_weight_bm.loc[asset_weight_bm.index.intersection(asset_weight.index)]
        # asset_weight = asset_weight.shift(1)

        # portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        daily_ret_bm = self.daily_ret.loc[self.daily_ret.index.intersection(asset_weight_bm.index)]

        portfolio_return = (asset_weight_bm * daily_ret_bm).sum(axis=1)

        return super().post_strategy(portfolio_return, asset_weight)
