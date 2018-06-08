"""
Interface & Implementation of Portfolio Strategies.
"""
import numpy as np
import pandas as pd
# from typing import Tuple
from abc import ABC, abstractmethod
from statsmodels.stats.weightstats import DescrStatsW


class TradingRule(ABC):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    @abstractmethod
    def compute_rule(self) -> pd.DataFrame:
        pass


class SIGN(TradingRule):
    def __init__(self, data: pd.DataFrame, daily_ret: pd.DataFrame, lookback: int):
        super().__init__(data)
        self.daily_ret = daily_ret
        self.lookback = lookback

    def compute_rule(self) -> pd.DataFrame:
        """

        :return:
        """
        annual_ret = self.data.pct_change(periods=self.lookback)
        annual_ret = annual_ret > 0
        annual_ret = (annual_ret * 2) - 1

        return annual_ret


class TREND(TradingRule):
    def __init__(self, data: pd.DataFrame, daily_ret: pd.DataFrame, lookback: int):
        super().__init__(data)
        self.daily_ret = daily_ret
        self.lookback = lookback

    def compute_rule(self):
        # daily_ret_log = np.log(self.daily_ret)
        daily_ret_log = self.daily_ret
        df = pd.DataFrame()
        N = self.daily_ret.shape[0]

        for asset in daily_ret_log.columns:
            data = daily_ret_log[asset]
            first_not_null = 0
            t_scores = []

            for i in range(N):
                if np.isnan(data.iloc[i]):
                    first_not_null += 1
                    # TODO verify
                    t_scores.append(0)
                    continue

                if i <= first_not_null + self.lookback:
                    t_scores.append(0)
                    continue

                stats = DescrStatsW(data.iloc[i - self.lookback:i])
                t_scores.append(stats.ttest_mean(0, 'larger')[0])

            # df[asset.replace('PX_LAST_', '') + '_tstat'] = np.clip(t_scores, -1.0, 1.0)
            df[asset] = np.clip(t_scores, -1.0, 1.0)

        df['index'] = self.daily_ret.index
        df.set_index('index', inplace=True)

        return df
