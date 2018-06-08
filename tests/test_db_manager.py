from main import database_manager
import unittest


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()

    def test_con(self):
        self.assertIsNotNone(self.dbm.con)

    def test_dataset_names(self):
        self.assertIsNotNone(self.dbm.bloom_dataset_names)
        self.assertGreater(len(self.dbm.bloom_dataset_names), 0)

        self.assertIsNotNone(self.dbm.quandl_dataset_names)
        self.assertGreater(len(self.dbm.quandl_dataset_names), 0)

    def test_dicts(self):
        self.assertIsNotNone(self.dbm.type_to_table_dict.keys())
        self.assertGreater(len(self.dbm.type_to_table_dict.keys()), 0)

        self.assertIsNotNone(self.dbm.subtype_to_table_dict.keys())
        self.assertGreater(len(self.dbm.subtype_to_table_dict.keys()), 0)

    def test_get_table_bloom(self):
        df, info = self.dbm.get_table('bloom_ad1')
        self.assertIsNotNone(df)
        self.assertIsNotNone(info)

    def test_get_table_quandl(self):
        df, info = self.dbm.get_table('quandl_CME_CL1_OR')
        self.assertIsNotNone(df)
        self.assertIsNotNone(info)

    def test_get_info_bloom(self):
        info = self.dbm.get_info('bloom_ad1')

        self.assertIsNotNone(info)

    def test_get_info_quandl(self):
        info = self.dbm.get_info('quandl_CME_CL1_OR')

        self.assertIsNotNone(info)

    def test_get_asset_by_type(self):
        df = self.dbm.get_assets_by_type('Equity')
        self.assertIsNotNone(df)

        df = self.dbm.get_assets_by_type('Equity', 'Developed')
        self.assertIsNotNone(df)