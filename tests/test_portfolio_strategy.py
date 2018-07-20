from main import database_manager, finance_metrics as fm, asset_manager
from main.portfolio_strategy import tsmom, corrTSMOM
import datetime
import unittest


class TestPortfolioStrategies(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        self.asset_manager = asset_manager.AssetManager()

        self.bloom_all = self.asset_manager.get_all_bloom_assets()
        self.bloom_equity = self.asset_manager.get_assets_by_type_bloom('Equity')
        self.bloom_bond = self.asset_manager.get_assets_by_type_bloom('Interest Rates')
        self.bloom_commodity = self.asset_manager.get_assets_by_type_bloom('Commodity')
        self.bloom_currency = self.asset_manager.get_assets_by_type_bloom('Currency')

        self.bloom_all_paper = self.asset_manager.get_all_bloom_assets(in_paper=True)
        self.bloom_equity_paper = self.asset_manager.get_assets_by_type_bloom('Equity', in_paper=True)
        self.bloom_bond_paper = self.asset_manager.get_assets_by_type_bloom('Interest Rates', in_paper=True)
        self.bloom_commodity_paper = self.asset_manager.get_assets_by_type_bloom('Commodity', in_paper=True)
        self.bloom_currency_paper = self.asset_manager.get_assets_by_type_bloom('Currency', in_paper=True)

        self.quandl_all = self.asset_manager.get_quandl_bloom_intersection_assets()

    def test_tsmom_strategy_without_transaction(self):
        st_2 = tsmom.TSMOMStrategy(self.dbm, self.bloom_all, 0.4,
                                   volatilty_estimator='yz',
                                   trade_rule_name='sign',
                                   lookback_trend=252,
                                   lookback_vol=20,
                                   include_transaction=False)

        res_2, w2 = st_2.compute_strategy()
        p1 = fm.compute_portfolio_performance(w2, res_2, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w2, res_2, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        self.assertIsNotNone(res_2)

    def test_tsmom_strategy_with_transaction(self):
        st_2 = tsmom.TSMOMStrategy(self.dbm, self.bloom_all, 0.4,
                                   volatilty_estimator='yz',
                                   trade_rule_name='sign',
                                   lookback_trend=252,
                                   lookback_vol=20,
                                   include_transaction=True)

        res_2, w2 = st_2.compute_strategy()
        p1 = fm.compute_portfolio_performance(w2, res_2, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w2, res_2, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        ro = st_2.rollover_cost[
            st_2.rollover_cost.index >= datetime.datetime(1984, 1, 1)]
        ro = ro[ro.index < datetime.datetime(2013, 2, 1)]

        re = st_2.rebalance_cost[
            st_2.rebalance_cost.index >= datetime.datetime(1984, 1, 1)]
        re = re[re.index < datetime.datetime(2013, 2, 1)]

        print(ro.sum(axis=1).sum() / (ro.sum(axis=1).sum() + re.sum(axis=1).sum()))

        self.assertIsNotNone(res_2)

    def test_corr_adjusted_tsmom_strategy_wtihout_transaction(self):
        st_3 = corrTSMOM.CorrAdjustedTSMOMStrategy(self.dbm, self.bloom_all, 0.12,
                                                   lookback_vol=20,
                                                   lookback_corr=60,
                                                   lookback_trend=252,
                                                   volatilty_estimator='sd',
                                                   trade_rule_name='sign',
                                                   include_transaction=False)
        res_3, w3 = st_3.compute_strategy()

        p1 = fm.compute_portfolio_performance(w3, res_3, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w3, res_3, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        self.assertIsNotNone(res_3)

    def test_corr_adjusted_tsmom_strategy_wtih_transaction(self):
        st_3 = corrTSMOM.CorrAdjustedTSMOMStrategy(self.dbm, self.bloom_all, 0.12,
                                                   lookback_vol=20,
                                                   lookback_corr=60,
                                                   lookback_trend=252,
                                                   volatilty_estimator='yz',
                                                   trade_rule_name='trend',
                                                   include_transaction=True)
        res_3, w3 = st_3.compute_strategy()

        p1 = fm.compute_portfolio_performance(w3, res_3, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w3, res_3, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        ro = st_3.rollover_cost[
            st_3.rollover_cost.index >= datetime.datetime(1984, 1, 1)]
        ro = ro[ro.index < datetime.datetime(2013, 2, 1)]

        re = st_3.rebalance_cost[
            st_3.rebalance_cost.index >= datetime.datetime(1984, 1, 1)]
        re = re[re.index < datetime.datetime(2013, 2, 1)]

        print(ro.sum(axis=1).sum() / (ro.sum(axis=1).sum() + re.sum(axis=1).sum()))

        self.assertIsNotNone(res_3)

