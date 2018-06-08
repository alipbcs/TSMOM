"""
Helper functions for plotting.
"""
from main import finance_metrics as fm, database_manager as db
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


class PlotHelper(object):
    """
    Helper class for various plotting functions.

    Example usage:
        ph = plot_helper.PlotHelper()

        ph.plot_annual_return('bloom_ad1')
        ph.plot_px_last('bloom_cl1')
        ph.plot_volatility('bloom_cl1')
        ph.plot_n_t()
        ph.plot_volatility_type('Interest Rates')
        ph.plot_volatility_subtype('Softs')
        ...
        ph.save_open_plots_to_pdf()
    """

    # Borg design pattern
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

        if not hasattr(self, 'dbm'):
            self.dbm = db.DatabaseManager()

        if not hasattr(self, '__annual_ret_dic_bloom'):
            self.__annual_ret_dic_bloom = None

        if not hasattr(self, '__annual_ret_dic_quandl'):
            self.__annual_ret_dic_quandl = None

        if not hasattr(self, '__cvol'):
            self.__cvol = None

        sns.set(color_codes=True)
        plt.rcParams['figure.figsize'] = (10.0, 10.0)  # set default size of plots

    def plot_annual_return(self, tbl_name: str) -> None:
        """
        Plots annual return of given table with sign superimposed.
        :param tbl_name: name of the table.
        """
        dataset_type = tbl_name.split('_')[0]

        for_bloom = False

        if dataset_type == 'bloom':
            for_bloom = True
        elif dataset_type == 'quandl':
            pass
        else:
            print('Table not found.')
            return

        fig, axe = plt.subplots(1, 1)
        info = self.dbm.get_info(tbl_name)

        if for_bloom:
            if self.__annual_ret_dic_bloom is None:
                self.__annual_ret_dic_bloom = fm.compute_annual_returns(self.dbm, True)

            axe.set_title('Annual Return: {}'.format(info[1]))

            self.__annual_ret_dic_bloom[tbl_name]['annual_ret'].plot()
            self.__annual_ret_dic_bloom[tbl_name]['annual_ret_sign'].plot(alpha=0.3)

        else:
            if self.__annual_ret_dic_quandl is None:
                self.__annual_ret_dic_quandl = fm.compute_annual_returns(self.dbm, False)

            axe.set_title('Annual Return: {}'.format(info[1]))

            self.__annual_ret_dic_quandl[tbl_name]['annual_ret'].plot()
            self.__annual_ret_dic_quandl[tbl_name]['annual_ret_sign'].plot(alpha=0.3)

    def plot_n_t(self):
        """
        Plots n_t needed in equation (5) of article.
        """
        if self.__cvol is None:
            self.__cvol = fm.compute_cvol(self.dbm, 0.4, True)

        fig, axe = plt.subplots(1, 1)
        axe.set_title('N_t')

        self.__cvol['n_t'].plot()

    def plot_volatility(self, tbl_name: str, window=252) -> None:
        """
        Plots volatility of given table as rolling std of last 252 daily returns.
        :param tbl_name: name of the table.
        """
        df, info = self.dbm.get_table(tbl_name)

        if df is None:
            print("Table doesn't exist.")
            return

        df['daily_ret'] = df['PX_LAST'].pct_change()
        df['rolling_std'] = df['daily_ret'].rolling(window).std() * np.sqrt(window)

        fig, axe = plt.subplots(1, 1)
        axe.set_title('Volatility: {}'.format(info[1]))

        df['rolling_std'].plot()

    def plot_px_last(self, tbl_name: str) -> None:
        """
        Plots PX_LAST for given table.
        :param tbl_name: name of the table.
        """
        df, info = self.dbm.get_table(tbl_name)

        if df is None:
            print("Table doesn't exist.")
            return

        empty_dates_idx = df.index[df['PX_LAST'].isnull()]
        empty_dates_count = df['PX_LAST'].isnull().sum()

        fig, axe = plt.subplots(1, 1)
        [axe.axvline(x_position, alpha=0.5) for x_position in empty_dates_idx]
        axe.set_title('{}__null_count:{}'.format(info[1], empty_dates_count))

        df['PX_LAST'].plot()

    def plot_volatility_type(self, asset_type: str) -> None:
        """
        Plots volatility (forward filled) for give asset type.
        :param asset_type: type of asset.
        """
        if self.__cvol is None:
            self.__cvol = fm.compute_cvol(self.dbm, 0.4, True)

        tbl_list = self.dbm.type_to_table_dict.get(asset_type)

        if tbl_list is None:
            print('Type not found.')
            return

        col_list = []

        for tbl_name in tbl_list:
            dataset_type = tbl_name.split('_')[0]

            if dataset_type == 'bloom':
                if 'rolling_std_' + tbl_name in self.__cvol.columns:
                    col_list.append('rolling_std_' + tbl_name)

            elif dataset_type == 'quandl':
                pass

        df_temp = self.__cvol[col_list]

        fig, axe = plt.subplots(1, 1)
        axe.set_title('Volatility: Type_{}'.format(asset_type))

        df_temp.plot()

    def plot_volatility_subtype(self, asset_type: str) -> None:
        """
        Plots volatility (forward filled) for give asset type.
        :param asset_type: type of asset.
        """
        if self.__cvol is None:
            self.__cvol = fm.compute_cvol(self.dbm, 0.4, True)

        tbl_list = self.dbm.subtype_to_table_dict.get(asset_type)

        if tbl_list is None:
            print('Subtype not found.')
            return

        col_list = []

        for tbl_name in tbl_list:
            dataset_type = tbl_name.split('_')[0]

            if dataset_type == 'bloom':
                if 'rolling_std_' + tbl_name in self.__cvol.columns:
                    col_list.append('rolling_std_' + tbl_name)

            elif dataset_type == 'quandl':
                pass

        df_temp = self.__cvol[col_list]

        fig, axe = plt.subplots(1, 1)
        axe.set_title('Volatility: Subtype_{}'.format(asset_type))

        df_temp.plot()

    @staticmethod
    def save_open_plots_to_pdf():
        pdf = PdfPages('foo17.pdf')

        for i in plt.get_fignums():
            plt.figure(i)
            pdf.savefig()

        pdf.close()
