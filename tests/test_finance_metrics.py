from main import finance_metrics, database_manager
import unittest

class TestFinanceMetrics(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()

    def test_annual_return_bloom(self):
        dic = finance_metrics.compute_annual_returns(self.dbm, True)

        self.assertIsNotNone(dic)
        self.assertGreater(len(dic.keys()), 0)

    def test_monthly_return_bloom(self):
        df = finance_metrics.compute_monthly_returns(self.dbm, 'bloom_ad1')

        self.assertIsNotNone(df)
