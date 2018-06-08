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
    