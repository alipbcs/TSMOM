"""
Interface for Portfolio Strategies.
"""

import pandas as pd
from main import database_manager
from main.portfolio_strategy import time_varying


class ConstantVolatilityStrategy(time_varying.TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager,
                 data: pd.DataFrame,
                 sigma_target: float,
                 include_transaction: bool = True,
                 lookback_vol: int = 252,
                 volatilty_estimator: str = 'sd'):

        super().__init__(dbm, data, sigma_target, lookback_vol, volatilty_estimator, include_transaction)

    def compute_strategy(self):
        super().pre_strategy()

        asset_weight = self.sigma_target / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.shift(1)

        portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        return self.post_strategy(portfolio_return, asset_weight)

