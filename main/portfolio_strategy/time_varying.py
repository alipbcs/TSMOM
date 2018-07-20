"""
Interface for Portfolio Strategies.
"""

import numpy as np
import pandas as pd
from main import database_manager
from abc import ABC, abstractmethod
from main import volatility_estimator


class TimeVaryingPortfolioStrategy(ABC):
    """
    Example Usage:
        st_1 = main.portfolio_strategy.ConstantVolatilityStrategy(dbm, df, 0.4)
        res_1 = st_1.compute_strategy()

        st_2 = main.portfolio_strategy.TSMOMStrategy(dbm, df, 0.4)
        res_2 = st_2.compute_strategy()
    """

    def __init__(self, dbm: database_manager.DatabaseManager,
                 data: pd.DataFrame,
                 sigma_target: float,
                 lookback_vol: int = 34,
                 vol_estimator: str = 'sd',
                 include_transaction: bool = False):
        self.dbm = dbm
        self.data = data
        self.n_t = None
        self.aggregated_assets = None
        self.table_in_assets = []
        self.sigma_target = sigma_target
        self.daily_ret = None
        self.volatility = None
        self.volatility_estimator = vol_estimator
        self.lookback_window_vol = lookback_vol
        self.trade_rule_out = None
        self.transaction_cost_rollover = {'currency': 8.0 / 10000.0,
                                          'equity': 10.0 / 10000.0,
                                          'interest rates': 8.0 / 10000.0,
                                          'commodity': 20.0 / 10000.0}

        self.transaction_cost_rebalance = {'currency': 3.0 / 10000.0,
                                           'equity': 5.0 / 10000.0,
                                           'interest rates': 4.0 / 10000.0,
                                           'commodity': 6.0 / 10000.0}

        self.include_transaction_cost = include_transaction

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
        """
        Computes volatility of returns based on chosen method.
        :return: pandas dataframe containing volatilities.
        """
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

    @abstractmethod
    def compute_strategy(self):
        pass

    def post_strategy(self, portfolio_return, asset_weight):
        """
        Performs any necessary computations after completion of computation of strategy.
        :param portfolio_return: dataframe containing returns from strategy.
        :param asset_weight: dataframe containing asset weights from strategy.
        :return: tuple of dataframes containing returns and asset weights from strategy.
        """
        if not self.include_transaction_cost:
            return portfolio_return, asset_weight

        self.rollover_cost = self.__compute_rollover_costs(asset_weight)
        self.rebalance_cost = self.__compute_rebalance_costs(asset_weight)

        q1 = self.rollover_cost.sum(axis=1)
        q2 = self.rebalance_cost.sum(axis=1)

        print('q1: {}, q2: {}'.format(q1.sum(), q2.sum()))

        return portfolio_return - q1 - q2, asset_weight

    def __compute_rollover_costs(self, asset_weight):
        """
        Computes rollover consts as part of transaction costs.
        :param asset_weight: dataframe containing asset weights from strategy.
        :return: dataframe containing rollover costs.
        """
        asset_weight_by = asset_weight.resample('BY').mean() / 252.0
        asset_weight_by = asset_weight_by.resample('B').last()
        asset_weight_by.fillna(method='ffill', inplace=True)
        asset_weight_by = asset_weight_by.loc[asset_weight_by.index.intersection(asset_weight.index)]
        asset_weight_by = asset_weight_by.reindex(asset_weight.index, method='ffill')
        asset_weight_by = asset_weight_by.fillna(0)

        rollover_cost_temp = pd.DataFrame(index=asset_weight_by.index)

        for asset in asset_weight_by.columns:
            tbl_name = str.replace(asset, 'PX_LAST_', '')
            tbl_type = str.lower(self.dbm.table_to_type_dict[tbl_name])

            rollover_cost_temp[asset] = np.ones(asset_weight_by.shape[0]) * self.transaction_cost_rollover[tbl_type]

        rollover_cost = pd.DataFrame(index=rollover_cost_temp.index,
                                     data=asset_weight_by.abs().values * rollover_cost_temp.values)

        return rollover_cost

    def __compute_rebalance_costs(self, asset_weight):
        """
        Computes rebalancing consts as part of transaction costs.
        :param asset_weight: dataframe containing asset weights from strategy.
        :return: dataframe containing rebalancing costs.
        """
        asset_weight_bm = asset_weight.resample('BM').last()
        asset_weight_bm = asset_weight_bm.diff().abs() / 20.0
        asset_weight_b = asset_weight_bm.resample('B').last()
        asset_weight_b = asset_weight_b.fillna(method='ffill')

        asset_weight_b = asset_weight_b.loc[asset_weight_b.index.intersection(asset_weight.index)]
        # asset_weight_bm = asset_weight_bm.reindex(asset_weight.index, method='ffill')
        asset_weight_b = asset_weight_b.fillna(0)

        rebalance_cost_temp = pd.DataFrame(index=asset_weight_b.index)

        for asset in asset_weight_b.columns:
            tbl_name = str.replace(asset, 'PX_LAST_', '')
            tbl_type = str.lower(self.dbm.table_to_type_dict[tbl_name])

            rebalance_cost_temp[asset] = np.ones(asset_weight_b.shape[0]) * self.transaction_cost_rebalance[tbl_type]

        rollover_cost = pd.DataFrame(index=rebalance_cost_temp.index,
                                     data=asset_weight_b * rebalance_cost_temp)

        return rollover_cost

