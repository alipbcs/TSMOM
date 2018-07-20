import numpy as np
import pandas as pd
from main import database_manager
from main.portfolio_strategy import time_varying
from main import trade_rule


class CorrAdjustedTSMOMStrategy(time_varying.TimeVaryingPortfolioStrategy):
    def __init__(self, dbm: database_manager.DatabaseManager,
                 data: pd.DataFrame,
                 sigma_target: float,
                 lookback_vol: int = 252,
                 volatilty_estimator: str = 'sd',
                 lookback_corr:int = 34,
                 trade_rule_name: str ='SIGN',
                 lookback_trend: int = 252,
                 include_transaction: bool = False):

        super().__init__(dbm, data, sigma_target, lookback_vol, volatilty_estimator, include_transaction)
        self.lookback_trend = lookback_trend
        self.lookback_corr = lookback_corr
        self.trade_rule_name = trade_rule_name

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

        cf_list = []

        for t in range(self.daily_ret.shape[0]):
            if t < self.lookback_corr:
                cf_list.append(1)
                continue

            daily_ret_window = self.daily_ret.iloc[t - self.lookback_corr:t]

            assets_present = self.daily_ret.columns[self.daily_ret.iloc[t].notnull()]

            if t % 100 == 0:
                print('Progress: {}%'.format(int(t * 100 / self.n_t.shape[0])))

            daily_ret_window_assets_present = daily_ret_window[assets_present]
            daily_ret_window_assets_present = daily_ret_window_assets_present.dropna(how='any')
            # remove columns which are all zeroes
            non_zero_columns = daily_ret_window_assets_present.columns[(daily_ret_window_assets_present > 0).any()]
            daily_ret_window_assets_present = daily_ret_window_assets_present[non_zero_columns]

            trade_rule_out_assets = self.trade_rule_out[non_zero_columns]

            if daily_ret_window_assets_present.shape[0] < 3 or daily_ret_window_assets_present.shape[1] < 3:
                cf_list.append(1)
                continue

            if (daily_ret_window_assets_present == 0).all().sum() > 0:
                print('Warning: ALL ZERO.')

            asset_corr = daily_ret_window_assets_present.corr().values

            co_trade_rule = np.zeros((asset_corr.shape[0], asset_corr.shape[1]))

            trade_rule_curr = trade_rule_out_assets.iloc[t].values
            for i in range(co_trade_rule.shape[0]):
                for j in range(i + 1, co_trade_rule.shape[1]):
                    co_trade_rule[i, j] = trade_rule_curr[i] * trade_rule_curr[j]

            co_trade_rule = co_trade_rule.T + co_trade_rule

            N = asset_corr.shape[0]
            rho_bar = (asset_corr * co_trade_rule).sum() / (N * (N - 1))
            temp = N / ((rho_bar * (N - 1)) + 1.0)

            if temp < 0:
                print('Warning: negative value encountered for taking square root.')
                cf_list.append(np.sqrt(N))
                continue

            if np.isnan(temp) or np.isinf(temp):
                print('Infinity or NaN.')
                cf_list.append(1)
                continue

            cf_t = np.sqrt(temp)
            cf_list.append(cf_t)

        asset_weight = self.sigma_target * self.trade_rule_out / self.volatility
        asset_weight = asset_weight.div(self.n_t, axis=0)
        asset_weight = asset_weight.mul(np.array(cf_list), axis=0)
        #
        # asset_weight = asset_weight.shift(1)

        asset_weight_bm = asset_weight.resample('BM').mean()
        asset_weight_bm = asset_weight_bm.resample('B').last()
        asset_weight_bm = asset_weight_bm.fillna(method='ffill')
        asset_weight_bm = asset_weight_bm.loc[asset_weight_bm.index.intersection(asset_weight.index)]

        daily_ret_bm = self.daily_ret.loc[self.daily_ret.index.intersection(asset_weight_bm.index)]
        # portfolio_return = (asset_weight * self.daily_ret).sum(axis=1)

        portfolio_return = (asset_weight_bm * daily_ret_bm).sum(axis=1)

        return self.post_strategy(portfolio_return, asset_weight)
