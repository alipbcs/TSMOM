from main import finance_metrics, database_manager, portfolio_strategy
import unittest
import pandas as pd

class TestFinanceMetrics(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        self.df = pd.DataFrame(self.dbm.bloom_dataset_names)
        self.df.set_index(0, inplace=True)

    def test_annual_return_bloom(self):
        df = finance_metrics.compute_annual_return_from_daily_return(self.dbm, 'bloom_nk1')

        self.assertIsNotNone(df)

    def test_monthly_return_bloom(self):
        df = finance_metrics.compute_compounded_monthly_return_from_daily_return(self.dbm, 'bloom_ad1')

        self.assertIsNotNone(df)

    def test_portfolio_performance(self):
        qq2 = portfolio_strategy.TSMOMStrategy(self.dbm, self.df, 0.4, volatilty_estimator='sd', trade_rule_name='sign', lookback=30)
        res2, w2 = qq2.compute_strategy()

        d = finance_metrics.compute_portfolio_performance(w2, res2, '1990-01-01', '2005-01-01')

        self.assertIsNotNone(d)
