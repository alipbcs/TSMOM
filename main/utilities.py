#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:43:40 2018

@author: macbook2
"""
from main import database_manager

class AssetMaster(object):
    """
    Helper class for asset groups
    """
    def __init__(self, dbm: database_manager.DatabaseManager):
        self.equity_dvp_universe = dbm.get_assets_by_type('Equity', 'Developed')
        self.equity_emg_universe = dbm.get_assets_by_type('Equity', 'Emerging')
        self.equity_universe = self.equity_dvp_universe.append(self.equity_emg_universe)
        self.bond_universe = dbm.get_assets_by_type('Interest Rates')
        self.currency_universe = dbm.get_assets_by_type('Currency')
        self.commodity_universe = dbm.get_assets_by_type('Commodity')
        self.grain_universe = dbm.get_assets_by_type('Commodity', 'Grains')
        self.metal_universe = dbm.get_assets_by_type('Commodity', 'Metal')
        self.precious_universe = dbm.get_assets_by_type('Commodity', 'Precious Metal')
        self.energy_universe = dbm.get_assets_by_type('Commodity', 'Energy')
        self.soft_universe = dbm.get_assets_by_type('Commodity', 'Softs')
        self.meat_universe = dbm.get_assets_by_type('Commodity', 'Meat')
        self.asset_universe = self.equity_universe.append([self.bond_universe, \
                                                           self.currency_universe, \
                                                           self.commodity_universe])
        
    def paper_univ(self):
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
         return self.asset_universe[self.asset_universe.contract_name.isin(paper_list)]


    
    
#    def all_asset(self):
#        return self.asset_universe
#    def bond_asset(self):
#        return self.bond_universe
#    def currency_asset(self):
#        return self.currency_universe
#    def commodity_asset(self):
#        return self.commodity_universe
#    def equity_dvp(self):
#        return self.equity_dvp_universe
#    def equity_emg(self):
#        return self.equity_emg_universe
#    def grain_asset(self):
#        return self.grain_universe
#    def metal_asset(self):
#        return self.metal_universe
#    def precious_asset(self):
#        return self.precious_universe
#    def energy_asset(self):
#        return self.energy_universe
#    def soft_asset(self):
#        return self.soft_universe
#    def meat_asset(self):
#        return self.meat_universe
    