from main import stats_helper, database_manager
import unittest

class TestStats(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()

    def test_stats_summary_bloom(self):
        dic = stats_helper.compute_summary_statistics(self.dbm, 'bloom_ad1')

        self.assertIsNotNone(dic)

    def test_stats_summary_quandl(self):
        dic = stats_helper.compute_summary_statistics(self.dbm, 'quandl_CME_CL1_OR')

        self.assertIsNotNone(dic)