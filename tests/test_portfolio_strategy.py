from main import database_manager, portfolio_strategy
import unittest


class TestPortfolioStrategires(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        self.df = self.dbm.get_assets_by_type('Equity')

    def test_constant_vol_strategy(self):
        st_1 = portfolio_strategy.ConstantVolatilityStrategy(self.dbm, self.df, 0.4)
        res_1 = st_1.compute_strategy()

        self.assertIsNotNone(res_1)

    def test_tsmom_strategy(self):
        st_2 = portfolio_strategy.TSMOMStrategy(self.dbm, self.df, 0.4)
        res_2 = st_2.compute_strategy()

        self.assertIsNotNone(res_2)

    def test_corr_adjusted_tsmom_strategy(self):
        st_2 = portfolio_strategy.CorrAdjustedTSMOMStrategy(self.dbm, self.df, 0.4)
        res_2 = st_2.compute_strategy()

        self.assertIsNotNone(res_2)
