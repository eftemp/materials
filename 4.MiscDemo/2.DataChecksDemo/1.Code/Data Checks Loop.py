# -*- coding: utf-8 -*-
"""
Data import/Checks

"""

import pandas as pd
import numpy as np
import os
import sys
import requests
import matplotlib.pyplot as plt

homedir = os.environ['HOMEPATH']
wdir = os.path.join(homedir , 'Google Drive/AI/Research findings/NON-LIVE STRATEGY RESEARCH/Long-term short strategies')
dropboxdir = os.path.join(homedir , 'Dropbox/Trading Automation/Price Data')
os.chdir(wdir+'/1. Data/Semi Annual')
fields =os.listdir()
fields = [k.replace('.csv','') for k in fields]
fields.remove('desktop.ini')

excludesecondary = True
fundamental_fill_limit = 260 #Number of days for which nans will be filled before the dataset becomes nan
for field in fields:
#    if field in ['ASSET_TO_EQY','BS_CUR_ASSET_REPORT','CF_CAP_EXPEND_PRPTY_ADD','SHAREHOLDER_YIELD','ASSET_TO_EQY','NET_DEBT_TO_FLOW','TOT_DEBT_TO_TOT_EQY']: continue
    # =============================================================================
    #     #Equity from wdir
    # =============================================================================
    Open = pd.read_csv(wdir+'/1. Data/Equity/PX_OPEN.csv', index_col = 0, parse_dates = True)
    High = pd.read_csv(wdir+'/1. Data/Equity/PX_HIGH.csv', index_col = 0, parse_dates = True)
    Low = pd.read_csv(wdir+'/1. Data/Equity/PX_LOW.csv', index_col = 0, parse_dates = True)
    Close = pd.read_csv(wdir+'/1. Data/Equity/PX_LAST.csv', index_col = 0, parse_dates = True)
    Volume = pd.read_csv(wdir+'/1. Data/Equity/PX_VOLUME.csv', index_col = 0, parse_dates = True)
    MarketCap = pd.read_csv(wdir+'/1. Data/Equity/CUR_MKT_CAP.csv', index_col = 0, parse_dates = True)*1000000
    Turnover = pd.read_csv(wdir+'/1. Data/Equity/TURNOVER.csv', index_col = 0, parse_dates = True)
    
    # =============================================================================
    #     #ALSI Constituent from Dropbox
    # =============================================================================
    ALSIconst = pd.read_csv(dropboxdir+'/JALSHWeight/Weight.csv', index_col = 0, parse_dates = True)
    ALSIconst.columns = [str(col) + ' Equity' for col in ALSIconst.columns]
    
    # =============================================================================
    #     #Fundamental Data prep: read in semi annual, plug holes with annual, exclude secondary tickers, rename to match Close names, reshape to match Close shape
    # =============================================================================
    # =============================================================================
    #   #Read in semi annual fundamentals and fill nans with annual data
    # =============================================================================
    fundamental1insa = pd.read_csv(wdir+'/1. Data/Semi Annual/'+field+'.csv', index_col = 0, parse_dates = True)
    fundamental1ina = pd.read_csv(wdir+'/1. Data/Annual/'+field+'.csv', index_col = 0, parse_dates = True)
    #Basically, the concern is that we can't reindex using Close.index, because in the event that we're looking at a 31/12/xxxx date, 31/12/xxxx won't be in Close index and may be lost
    dates = np.union1d(Close.index, fundamental1insa.index)
    dates = np.union1d(dates, fundamental1ina.index)
    fundamental1 = pd.DataFrame(index = dates, columns = fundamental1insa.columns)
    fundamental1[fundamental1insa.columns] = fundamental1insa
    temp = fundamental1ina.copy(deep = True)
    fundamental1ina = pd.DataFrame(index = dates, columns = temp.columns)
    fundamental1ina[temp.columns] = temp[temp.columns]
    fundamental1a = fundamental1.loc[fundamental1ina.index]
    fundamental1[fundamental1.isnull()] = fundamental1a[fundamental1.isnull()] #Basically, replace all nans in fundamental1 with values from fundamental1a, in the hope that at least some in fundamental1a are not nan, worst case- all nans replaced with all nans
    del fundamental1a
    del temp
    del fundamental1ina
    del fundamental1insa
    fundamental1 = fundamental1.fillna(method = 'pad', limit = fundamental_fill_limit)
    # =============================================================================
    #   #Identify secondary tickers if desired. Secondary = ticker where ALSI ticker maps to a ticker that is already also an ALSI ticker, e.g. FFB -> FFA and FFA is already ALSI
    # =============================================================================
    TickerMap = pd.read_csv(wdir+'/1. Data/TickerMap.csv')
    if excludesecondary:
        secondary =[False]*len(TickerMap)
        for k in range(len(TickerMap)):
            if pd.isnull(TickerMap['Primary Ticker'].iloc[k]): continue
            if TickerMap['ALSI Ticker'].iloc[k] != TickerMap['Primary Ticker'].iloc[k] and TickerMap['Primary Ticker'].iloc[k] in TickerMap['ALSI Ticker'].values:
                secondary[k] = True
        npsecondary = np.array(secondary)
    else:
        npsecondary = np.array([False]*len(TickerMap))
    
    # =============================================================================
    #   #Reindex all datasets to match
    # =============================================================================
    DataSets = [Open, High, Low, Close, Volume, MarketCap, Turnover, ALSIconst]
    
    StockNames = np.intersect1d(TickerMap['SJ Ticker'][~npsecondary].values, DataSets[0].columns)
    StockNames = np.intersect1d(StockNames,DataSets[1].columns)
    
    Dates = np.intersect1d(DataSets[0].index,DataSets[1].index)
    
    for dataset in DataSets[2:]:
        Dates = np.intersect1d(Dates, dataset.index)
        StockNames = np.intersect1d(StockNames, dataset.columns)
    
    for k in range(len(DataSets)):
        DataSets[k] = DataSets[k].loc[Dates, (StockNames)]
        DataSets[k] = DataSets[k].fillna(method = 'pad', limit = 5)
    Open = DataSets[0]
    High = DataSets[1]
    Low = DataSets[2]
    Close = DataSets[3]
    Volume = DataSets[4]
    Turnover = DataSets[6]
    ALSIconst = DataSets[7]
    dictmap = TickerMap[['SJ Ticker', 'Ticker List']].set_index('SJ Ticker').loc[StockNames]
    dictmap = dictmap.to_dict()['Ticker List']
    factor1 = pd.DataFrame(columns = Close.columns, index = Close.index)
    factor1[Close.columns] = fundamental1[Close.rename(columns = dictmap).columns]
    del fundamental1
    
    # =============================================================================
    #   #Create sub universe on which data reasonability checks are to be performed
    # =============================================================================
    universe = factor1[(~pd.isnull(factor1))]
    universe = universe*(~pd.isnull(ALSIconst))
    universe[universe == 0] = np.nan
    
    factor1vtime = universe.apply(pd.Series.describe, axis = 1)
    
    os.chdir(wdir+'/1. Data/DataVisualisation')
    plt.plot(factor1vtime['count'], label = 'Universe Size')
    plt.title('Coverage- '+field)
    plt.legend()
    plt.savefig('Coverage- '+field+'.png', bbox_inches='tight')
    plt.show()
    
    
    plt.plot(factor1vtime['25%'], label = '25%')
    plt.plot(factor1vtime['50%'], label = '50%')
    plt.plot(factor1vtime['75%'], label = '75%')
    plt.title('Q1, Q2, Q3 VS Time- '+field)
    plt.legend()
    plt.savefig('Q1, Q2, Q3 VS Time- '+field+'.png', bbox_inches='tight')   
    plt.show()
