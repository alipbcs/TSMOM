from main import finance_metrics, plot_helper, database_manager as db
from main import stats_helper

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

dbm = db.DatabaseManager()
dbm.get_info('quandl_CME_CL1_OR')
dbm.get_table('quandl_CME_CL1_OR')
info = dbm.get_info('bloom_ad1')

d = finance_metrics.compute_annual_returns(dbm, True)
df2 = finance_metrics.compute_cvol(dbm, 0.4, True)

# TODO convert string comparsino to lower

ph = plot_helper.PlotHelper()
ph.plot_annual_return('bloom_ad1')
ph.plot_px_last('bloom_cl1')
ph.plot_volatility('bloom_cl1')
ph.plot_n_t()

ph.plot_volatility_type('Interest Rates')
ph.plot_volatility_subtype('Softs')

ph.plot_annual_return('quandl_CME_CL1_OR')

ph.save_open_plots_to_pdf()