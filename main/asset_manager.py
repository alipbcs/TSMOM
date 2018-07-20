from main import database_manager as db
import pandas as pd


class AssetManager(object):

    # Borg design pattern
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

        if not hasattr(self, 'dbm'):
            self.dbm = db.DatabaseManager()

        if not hasattr(self, 'paper_list'):
            self.paper_list = ['FTSE 100 IDX FUT',
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

    def get_all_bloom_assets(self, in_paper=False):
        assets = []

        for tbl in self.dbm.bloom_dataset_names:
            df, info = self.dbm.get_table(tbl)

            if df is not None:
                assets.append((tbl, info[1]))

        df = pd.DataFrame(assets, columns=['tbl_name', 'contract_name'])
        df.set_index('tbl_name', inplace=True)

        if in_paper:
            df = df[df['contract_name'].isin(self.paper_list)]

        return df

    def get_quandl_bloom_intersection_assets(self, in_paper=False):
        assets = []

        for tbl in self.dbm.bloom_dataset_names:
            df, info = self.dbm.get_table(tbl)

            if df is not None:
                analogous_bloom = self.dbm.bloom_to_qunadl_dict.get(tbl)

                if analogous_bloom is not None:
                    for quandl_tbl in analogous_bloom:
                        assets.append((quandl_tbl, info[1]))

        df = pd.DataFrame(assets, columns=['tbl_name', 'contract_name'])
        df.drop_duplicates(subset=['tbl_name'], inplace=True)
        df.set_index('tbl_name', inplace=True)

        if in_paper:
            df = df[df['contract_name'].isin(self.paper_list)]

        df.drop_duplicates()

        return df

    def get_assets_by_type_bloom(self, asset_type, asset_subtype=None, in_paper=False):
        assert(asset_type in ['Commodity', 'Equity', 'Interest Rates', 'Currency'])

        if asset_subtype is not None:
            assert(asset_subtype in ['Developed', 'Emerging', 'Grains', 'Energy', 'Softs', 'Precious Metal',
                                     'Meat', 'Metal'])

        df = self.dbm.get_assets_by_type(asset_type, asset_subtype)

        if df is not None:
            if in_paper:
                df = df[df['contract_name'].isin(self.paper_list)]

            return df

        return None

    def get_assets_by_type_quandl(self, asset_type, asset_subtype=None, in_paper=False):
        assert (asset_type in ['Commodity', 'Equity', 'Interest rates', 'Currency'])

        if asset_subtype is not None:
            assert (asset_subtype in ['Developed', 'Emerging', 'Grains', 'Energy', 'Softs', 'Precious Metal',
                                      'Meat', 'Metal'])

        df_type = self.dbm.get_assets_by_type(asset_type, asset_subtype)

        if df_type is None:
            return None

        df_type.dropna(how='any', inplace=True)
        df_type.set_index('quandl_table_name')

        if in_paper:
            df_type = df_type[df_type['contract_name'].isin(self.paper_list)]

        return df_type
