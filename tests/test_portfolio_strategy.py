from main import database_manager, finance_metrics as fm
from main.portfolio_strategy import tsmom, corrTSMOM
import unittest


class TestPortfolioStrategires(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
        # self.df = self.dbm.get_assets_by_type('Interest Rates')
        self.df = self.dbm.get_all_bloom_assets()

        paper_list = ['FTSE 100 IDX FUT',
                  'SWISS MKT IX FUTR',
                  'E-Mini Russ 2000',
                  'S&P 500 FUTURE',
                  'NIKKEI 225 (OSE)',
                  'CAC40 10 EURO FUT',
                  'DAX INDEX FUTURE',
                  'HANG SENG IDX FUT',
                  'NASDAQ 100 E-MINI',
                  'IBEX 35 INDX FUTR',
                  'S&P/TSX 60 IX FUT',
                  'AMSTERDAM IDX FUT',
                  'FTSE/MIB IDX FUT',
                  'EURO-BUND FUTURE',
                  'US 10YR NOTE',
                  'US LONG BOND(CBT)',
                  'CAN 10YR BOND FUT',
                  'JPN 10Y BOND(OSE)',
                  'EURO-SCHATZ FUT',
                  'EURO BUXL 30Y BND',
                  'US 5YR NOTE (CBT)',
                  'US 2YR NOTE (CBT)',
                  'LONG GILT FUTURE',
                  'EURO-BOBL FUTURE',
                  'CHF CURRENCY FUT',
                  'JPN YEN CURR FUT',
                  'BP CURRENCY FUT',
                  'AUDUSD Crncy Fut',
                  'C$ CURRENCY FUT',
                  'EURO FX CURR FUT',
                  'WHEAT FUTURE(CBT)',
                  'CORN FUTURE',
                  'SOYBEAN OIL FUTR',
                  'SOYBEAN FUTURE',
                  'SOYBEAN MEAL FUTR',
                  'OAT FUTURE',
                  'LUMBER FUTURE',
                  'COTTON NO.2 FUTR',
                  '''COFFEE 'C' FUTURE''',
                  'COCOA FUTURE',
                  'FCOJ-A FUTURE',
                  'SUGAR #11 (WORLD)',
                  'BRENT CRUDE FUTR',
                  'GASOLINE RBOB FUT',
                  'WTI CRUDE FUTURE',
                  'NATURAL GAS FUTR',
                  'NY Harb ULSD Fut',
                  'SILVER FUTURE',
                  'PALLADIUM FUTURE',
                  'PLATINUM FUTURE',
                  'GOLD 100 OZ FUTR',
                  'COPPER FUTURE',
                  'CATTLE FEEDER FUT',
                  'LIVE CATTLE FUTR',
                  'LEAN HOGS FUTURE']

        self.df_paper = self.df[self.df['contract_name'].isin(paper_list)]

    def test_tsmom_strategy(self):
        transact = True

        st_2 = tsmom.TSMOMStrategy(self.dbm, self.df_paper, 0.4,
                                   volatilty_estimator='yz',
                                   trade_rule_name='trend',
                                   lookback_trend=252,
                                   lookback_vol=20,
                                   include_transaction=False)
        res_2, w2 = st_2.compute_strategy()
        p1 = fm.compute_portfolio_performance(w2, res_2, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w2, res_2, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        self.assertIsNotNone(st_2)

    def test_corr_adjusted_tsmom_strategy(self):
        st_3 = corrTSMOM.CorrAdjustedTSMOMStrategy(self.dbm, self.df_paper, 0.12,
                                                   lookback_vol=20,
                                                   lookback_corr=60,
                                                   lookback_trend=252,
                                                   volatilty_estimator='yz',
                                                   trade_rule_name='trend',
                                                   include_transaction=False)
        res_3, w3 = st_3.compute_strategy()

        p1 = fm.compute_portfolio_performance(w3, res_3, '1984-01-01', '2013-02-01')
        p2 = fm.compute_portfolio_performance(w3, res_3, '2009-01-01', '2013-02-01')

        print(p1)
        print(p2)

        self.assertIsNotNone(st_3)
