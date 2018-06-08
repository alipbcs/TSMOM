from main import portfolio_strategy, trade_rule, database_manager
import unittest


class TestPortfolioStrategires(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        self.df = self.dbm.get_assets_by_type('Equity')
        self.strategy = portfolio_strategy.ConstantVolatilityStrategy(self.dbm, self.df, 0.4)
        self.strategy.compute_strategy()

    def test_TREND(self):
        trend = trade_rule.TREND(self.strategy.aggregated_assets, self.strategy.daily_ret, 252)
        trend.compute_rule()
