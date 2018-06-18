import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class Volatility(ABC):
    def __init__(self, window: int, sigma_target: float):
        self.rolling_window = window
        self.sigma_target = sigma_target

    @abstractmethod
    def compute(self):
        pass


class VolatilitySD(Volatility):
    def __init__(self, daily_ret: pd.DataFrame, window: int, sigma_target: float):
        super().__init__(window, sigma_target)
        self.daily_ret = daily_ret

    def compute(self):
        volatility = self.daily_ret.rolling(self.rolling_window).std() * np.sqrt(252)
        volatility[volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0
        volatility.fillna(1.0, inplace=True)

        return volatility


class VolatilityYZ(Volatility):
    def __init__(self, window: int, sigma_target: float, asset_data: pd.DataFrame):
        super().__init__(window, sigma_target)
        self.data = asset_data

    def compute(self):
        volatility = self.__get_estimator(self.data, self.rolling_window)
        volatility[volatility < self.sigma_target / 10.0] = self.sigma_target / 10.0
        volatility.fillna(1.0, inplace=True)

        return volatility

    def __get_estimator(self, price_data: pd.DataFrame, window, trading_periods=252):
        log_ho = (price_data['PX_HIGH'] / price_data['PX_OPEN']).apply(np.log)
        log_lo = (price_data['PX_LOW'] / price_data['PX_OPEN']).apply(np.log)
        log_co = (price_data['PX_LAST'] / price_data['PX_OPEN']).apply(np.log)

        log_oc = (price_data['PX_OPEN'] / price_data['PX_LAST'].shift(1)).apply(np.log)
        log_oc_sq = log_oc ** 2

        log_cc = (price_data['PX_LAST'] / price_data['PX_LAST'].shift(1)).apply(np.log)
        log_cc_sq = log_cc ** 2

        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

        close_vol = log_cc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        open_vol = log_oc_sq.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))
        window_rs = rs.rolling(
            window=window,
            center=False
        ).sum() * (1.0 / (window - 1.0))

        k = 0.34 / (1 + (window + 1) / (window - 1))
        result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

        # if clean:
        #     return result.dropna()
        # else:
        return result
