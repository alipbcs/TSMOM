# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
from main import database_manager, portfolio_strategy, utilities
import datetime
import importlib

importlib.reload(database_manager)
importlib.reload(portfolio_strategy)
importlib.reload(utilities)
dbm = database_manager.DatabaseManager()

assets = utilities.AssetMaster(dbm)

cvol_strat = portfolio_strategy.ConstantVolatilityStrategy(dbm, assets.paper_univ(), 0.4)
tsmom_strat = portfolio_strategy.TSMOMStrategy(dbm, assets.paper_univ(), 0.4)
cormom_strat = portfolio_strategy.CorrAdjustedTSMOMStrategy(dbm, assets.paper_univ(), 0.4)
sd_window = 34
daily_port_rets, daily_wts, daily_ret = tsmom_strat.compute_strategy(sd_window)

end_date = '2018-05-08'
# remove data after 5th May 2018 because not all series are updated
daily_ret = daily_ret.truncate(after=end_date)
daily_wts = daily_wts.truncate(after=end_date)
daily_ret = daily_ret.truncate(after=end_date)

daily_ret = daily_ret.dropna(how='all')
daily_wts = daily_wts.dropna(how='all')

daily_index = (1 + daily_ret).cumprod()
# careful this will create index for end of last month that is unknown
monthly_index = daily_index.resample('BM').last()
monthly_rets = monthly_index.pct_change()
monthly_wts = daily_wts.resample('BM').last()
# remove last month
monthly_rets = monthly_rets[:-1]
monthly_wts = monthly_wts[:-1]

monthly_port_ret = (monthly_rets * monthly_wts.shift(1)).sum(axis=1)
monthly_port_ret = monthly_port_ret[monthly_port_ret!=0]

(1 + monthly_port_ret).cumprod()**(12/len(monthly_port_ret)) - 1

daily_turnover = wts.diff().abs().sum(axis=1)
daily_turnover.resample('M').sum().mean()

import pyfolio as pf

ret_annual = pf.timeseries.annual_return(rets.tz_localize('UTC'))
vol_annual = pf.timeseries.annual_volatility(rets.tz_localize('UTC'))
sharpe = ret_annual / vol_annual

bmrk = rets.copy()
bmrk[:] = np.random.rand()/10000
wts_dlr = wts * 1000000
wts_dlr['cash'] = 0
leverage = wts_dlr.sum(axis=1)/1000000
pf.create_full_tear_sheet(rets.tz_localize('UTC'), benchmark_rets=bmrk.tz_localize('UTC'), positions=wts_dlr, )

pf.create_position_tear_sheet(rets.tz_localize('UTC'), wts_dlr.tz_localize('UTC'), gross_lev=leverage)

daily_turnover = wts.diff().abs().sum(axis=1)
daily_turnover.index.month

#dataset_names = dbm.bloom_dataset_namesm
#prev_table_name = dataset_names[0]
#
#cummulative_strategy, _ = dbm.get_table(prev_table_name)
#cummulative_strategy['PX_LAST_' + prev_table_name] = cummulative_strategy['PX_LAST']
#cummulative_strategy['PX_LAST_' + prev_table_name].fillna(method='ffill', inplace=True)
#cummulative_strategy = cummulative_strategy[['PX_LAST_' + prev_table_name]]
#
#table_present = np.zeros(len(dataset_names))
#table_present[0] = 1
#i = 1
#
#for tbl_name in dataset_names[1:]:
#    #if verbose:
#    #    print('Pass 1: {}%'.format(i / len(dataset_names)))
#
#    df_curr, info = dbm.get_table(tbl_name)
#
#    if df_curr is not None:
#        if df_curr.shape[0] < 260:
#            i += 1
#            continue
#
#        table_present[i] = 1
#
#        df_curr['PX_LAST_' + tbl_name] = df_curr['PX_LAST']
#        df_curr = df_curr[['PX_LAST_' + tbl_name]]
#
#        cummulative_strategy = cummulative_strategy.join(df_curr, on='Dates', how='outer', sort=True)
#        cummulative_strategy.fillna(method='ffill', inplace=True)
#
#    i += 1
#
#assets_present = cummulative_strategy.notna().sum(axis=1)
#sigma_target = 0.4
## daily returns
#daily_ret = cummulative_strategy.pct_change()
## annual returns
#annual_ret = cummulative_strategy.pct_change(periods=252)
## rolling standard deviations
#std_rolling = daily_ret.rolling(252).std() * np.sqrt(252)
#
## Max 10* weight on any asset
#std_rolling[std_rolling < sigma_target/10] = sigma_target/10
#
## asset weight 
#asset_wt = sigma_target / std_rolling
#annual_ret_signed = ((annual_ret > 0)* 2 )- 1
## asset_wt_signed = annual_ret * asset_wt
#asset_wt_signed = annual_ret_signed * asset_wt
#
#asset_wts_actual = asset_wt_signed.div(assets_present, axis=0)
## you need to shift weights by one day
#all_asset_returns = asset_wts_actual.shift(1) * daily_ret
#portfolio_return = all_asset_returns.sum(axis=1)
#
#import pyfolio as pf
## If import pyfolio breaks you might have to change the line 
##     from pandas.core.common import is_list_like
## to 
##   from pandas.api.types import is_list_like
#
#pf.timeseries.annual_return(portfolio_return.apply(np.array).tz_localize('UTC'))
#pf.timeseries.annual_volatility(portfolio_return.apply(np.array).tz_localize('UTC'))