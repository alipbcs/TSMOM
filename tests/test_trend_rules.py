from main import portfolio_strategy, trade_rule, database_manager
import unittest
import numpy as np

class TestTrendRules(unittest.TestCase):
    def setUp(self):
        self.dbm = database_manager.DatabaseManager()
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

    def test_TREND(self):
        np.seterr(all='raise')

        qq3 = portfolio_strategy.CorrAdjustedTSMOMStrategy(self.dbm, self.df_paper, 0.4, volatilty_estimator='sd', trade_rule_name='trend',
                                           lookback=34)
        res3, w3 = qq3.compute_strategy()

        self.assertIsNotNone(res3)

