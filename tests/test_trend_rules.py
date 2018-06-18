from main import portfolio_strategy, trade_rule, database_manager
import unittest
import numpy as np


class TestTrendRules(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        self.df = self.dbm.get_all_bloom_assets()

    def test_TREND(self):
        np.seterr(all='raise')

        qq3 = portfolio_strategy.CorrAdjustedTSMOMStrategy(self.dbm, self.df, 0.4, volatilty_estimator='sd', trade_rule_name='trend',
                                           lookback=34)
        res3, w3 = qq3.compute_strategy()

        self.assertIsNotNone(res3)

