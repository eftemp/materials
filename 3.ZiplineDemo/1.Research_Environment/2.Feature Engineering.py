#!/usr/bin/env python
# coding: utf-8

# %%
import sys
import numpy as np
import pandas as pd
# # Data Bundle
sys.path.insert(-1,"C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/Helper_Functions")
import helper
import helper_ZAEQ
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume

bundle_data = helper_ZAEQ.load_bundle()
engine = helper_ZAEQ.build_ZA_engine(bundle_data)
sector=helper_ZAEQ.Sector()

# %%
universe_end_date = pd.Timestamp('2018-01-05', tz='UTC')
universe = AverageDollarVolume(window_length=120).top(100) 
factor_start_date = universe_end_date - pd.DateOffset(years=12, days=2)

# %%
from trading_calendars import get_calendar
trading_calendar = get_calendar('XJSE')
data_portal = helper.create_data_portal(bundle_data, trading_calendar)

# %%
## Create alpha factor
### Momentum 1 year
from zipline.pipeline.factors import Returns

universe = AverageDollarVolume(window_length=120).top(100)

pipeline = Pipeline(screen=universe)

def momentum_1yr(window_length, universe, sector):
    return Returns(window_length=window_length,
                   mask=universe).demean(groupby=sector).rank().zscore()

pipeline.add(
    momentum_1yr(252, universe, sector),
    'Momentum_1YR')

from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import CustomFactor

class Sharpe(CustomFactor):  
    inputs = [EquityPricing.close] 
    window_length = 252
    
    def compute(self, today, assets, out, closes):
          
        dailyrets = np.diff(np.log((closes[-251:-21])), axis = 0)    
        mean =np.nanmean(dailyrets, axis = 0)
        stdev = np.nanstd(dailyrets, axis = 0)
        am = np.divide(mean,stdev)
        out[:] = am

def sharpe(window_length, universe, sector):
    return Sharpe(mask = universe,
                  window_length = window_length).demean(groupby=sector).rank().zscore()

pipeline.add(sharpe(252, universe, sector), 'Sharpe')

def countpos(data):
    out = np.sum(data >0)
    return out

def countneg(data):
    out = np.sum(data < 0)
    return out
    
def avepos(data):
    out = np.average(data[data>0])
    return out

def aveneg(data):
    out = np.average(data[data<0])
    return out

class EUbyEL(CustomFactor):  
    inputs = [EquityPricing.close] 
    window_length = 200

    def compute(self, today, assets, out, closes):
        ##Factor prep
        lookback = 100
        lag = 20
        npdailyrets = np.diff(np.log((closes[-(lookback+lag+1):-(lag)])), axis = 0)
        posretcount = np.array([countpos(npdailyrets[:,stock]) for stock in range(npdailyrets.shape[1])])
        negretcount = np.array([countneg(npdailyrets[:,stock]) for stock in range(npdailyrets.shape[1])])
        posretave = np.array([avepos(npdailyrets[:,stock]) for stock in range(npdailyrets.shape[1])])
        negretave = np.array([aveneg(npdailyrets[:,stock]) for stock in range(npdailyrets.shape[1])])
        factor = (posretcount*posretave)/np.abs((negretcount*negretave))
        out[:] = factor

def euel(window_length, universe, sector):
    return EUbyEL(mask = universe,
                  window_length = window_length).demean(groupby=sector).rank().zscore()

class AdvancedMomentum(CustomFactor):  
    inputs = [EquityPricing.close] 
    window_length = 252
    
    def compute(self, today, assets, out, closes):  
        am = np.divide(closes[-21] - closes[-252],closes[-252])    
        out[:] = am

pipeline.add(euel(252, universe, sector),'euel')

def advanced_momentum(window_length, universe, sector):
    return AdvancedMomentum(mask = universe, 
                            window_length=window_length
                            ).demean(groupby=sector).rank().zscore()

from zipline.pipeline.factors import SimpleMovingAverage

def advanced_momentum_smoothed(window_length, universe, sector):

    return SimpleMovingAverage(inputs=[
        advanced_momentum(window_length,
                            universe,
                            sector)],window_length = window_length).rank().zscore()

pipeline.add(
     advanced_momentum(252, universe, sector),
    'AM')
pipeline.add(
     advanced_momentum_smoothed(252, universe, sector),
    'AM_Smoothed')

from scipy import stats
def _slope(ts, x=None):
    if x is None:
        x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    return slope

class StocksOnTheMove(CustomFactor):
    """
    12 months Momentum
    Stocks on the move momentum factor
    """
    inputs = [EquityPricing.close]
    window_length = 252
           
    def compute(self, today, assets, out, close):
        x = np.arange(len(close))
        slope = np.apply_along_axis(_slope, 0, close, x.T)
        out[:] = slope

def stocks_on_the_move(window_length, universe, sector):
    return StocksOnTheMove(mask = universe, 
                            window_length=window_length
                            ).demean(groupby=sector).rank().zscore()

# pipeline.add(
#     stocks_on_the_move(252, universe, sector),
#     'StocksOnTheMove')

from zipline.pipeline.factors import AnnualizedVolatility

volatility_120d = -1*AnnualizedVolatility(window_length=120, mask=universe).rank().zscore()
volatility_20d = -1*AnnualizedVolatility(window_length=20, mask=universe).rank().zscore()

pipeline.add(volatility_20d, 'volatility_20d')
pipeline.add(volatility_120d, 'volatility_120d')

"""already imported earlier, but shown here for reference"""
#from zipline.pipeline.factors import AverageDollarVolume 

adv_20d = AverageDollarVolume(window_length=20, mask=universe).rank().zscore()
adv_120d = AverageDollarVolume(window_length=120, mask=universe).rank().zscore()

pipeline.add(adv_20d, 'adv_20d')
pipeline.add(adv_120d, 'adv_120d')

'''
Market Regime Features#
We are going to try to capture market-wide regimes: Market-wide means we'll look at the aggregate movement of the universe of stocks.

High and low dispersion: dispersion is looking at the dispersion (standard deviation) of the cross section of all stocks at each period of time (on each day). We'll inherit from CustomFactor. We'll feed in DailyReturns as the inputs
'''
from zipline.pipeline.factors import DailyReturns

class MarketDispersion(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1
    window_safe = True

    def compute(self, today, assets, out, returns):

        mean_returns = np.nanmean(returns)

        out[:] = np.sqrt(np.nanmean((returns - mean_returns)**2))


dispersion = MarketDispersion(mask=universe)

dispersion_20d = SimpleMovingAverage(inputs=[dispersion], window_length=20)
dispersion_120d = SimpleMovingAverage(inputs=[dispersion], window_length=120)

# Add to pipeline
pipeline.add(dispersion_20d, 'dispersion_20d')
pipeline.add(dispersion_120d, 'dispersion_120d')

def mean_reversion_5day_sector_neutral(window_length, universe, sector):
    """
    Generate the mean reversion 5 day sector neutral factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral factor
    """
    return momentum_1yr(window_length,universe,sector)*-1

def mean_reversion_5day_sector_neutral_smoothed(window_length, 
                                                universe, sector):
    """
    Generate the mean reversion 5 day sector neutral smoothed factor

    Parameters
    ----------
    window_length : int
        Returns window length
    universe : Zipline Filter
        Universe of stocks filter
    sector : Zipline Classifier
        Sector classifier

    Returns
    -------
    factor : Zipline Factor
        Mean reversion 5 day sector neutral smoothed factor
    """
    
    return SimpleMovingAverage(inputs=[
        mean_reversion_5day_sector_neutral(window_length,
                                           universe, 
                                           sector)],window_length=window_length).rank().zscore()




class MarketVolatility(CustomFactor):
    inputs = [DailyReturns()]
    window_length = 1  # We'll want to set this in the constructor when creating the object.
    window_safe = True
    params = {'annualization_factor': 252.0}

    def compute(self, today, assets, out, returns, annualization_factor):

        # TODO
        """ 
        For each row (each row represents one day of returns), 
        calculate the average of the cross-section of stock returns
        So that market_returns has one value for each day in the window_length
        So choose the appropriate axis (please see hints above)
        """
        mkt_returns = np.nanmean(returns, axis=1) 

        # TODO
        # Calculate the mean of market returns
        mkt_returns_mu = np.nanmean(mkt_returns)

        # TODO
        # Calculate the standard deviation of the market returns, then annualize them.
        out[:] = np.sqrt(annualization_factor * np.nanmean((mkt_returns-mkt_returns_mu)**2))

# TODO: create market volatility features using one month and six-month windows
market_vol_20d = MarketVolatility(window_length=20, mask=universe)
market_vol_120d = MarketVolatility(window_length=120, mask=universe)

# add market volatility features to pipeline
pipeline.add(market_vol_20d, 'market_vol_20d')
pipeline.add(market_vol_120d, 'market_vol_120d')

pipeline.add(sector, 'sector_code')
pipeline.add(
    mean_reversion_5day_sector_neutral_smoothed(5, universe, sector),
    'Mean_Reversion_5Day_Sector_Neutral_Smoothed')


class CTO(Returns):
    """
    Computes the overnight return, per hypothesis from
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2554010
    """
    inputs = [EquityPricing.open, EquityPricing.close]
    
    def compute(self, today, assets, out, opens, closes):
        """
        The opens and closes matrix is 2 rows x N assets, with the most recent at the bottom.
        As such, opens[-1] is the most recent open, and closes[0] is the earlier close
        """
        out[:] = (opens[-1] - closes[0]) / closes[0]
        
class TrailingOvernightReturns(Returns):
    """
    Sum of trailing 1m O/N returns
    """
    window_safe = True
    
    def compute(self, today, asset_ids, out, cto):
        out[:] = np.nansum(cto, axis=0)

def overnight_sentiment_smoothed(cto_window_length, trail_overnight_returns_window_length, universe):
    cto_out = CTO(mask=universe, window_length=cto_window_length)
    unsmoothed_factor = TrailingOvernightReturns(inputs=[cto_out], window_length=trail_overnight_returns_window_length).rank().zscore()
    return SimpleMovingAverage(inputs=[unsmoothed_factor], window_length=trail_overnight_returns_window_length).rank().zscore()

pipeline.add(overnight_sentiment_smoothed(2, 10, universe), 'Overnight_Sentiment_Smoothed')

all_factors = engine.run_pipeline(pipeline, factor_start_date, universe_end_date)

all_factors.head()


# %%

sector_code_l = set(all_factors['sector_code'])
sector_0 = all_factors['sector_code'] == 0
sector_0[0:5]

sector_0_numeric = sector_0.astype(int)
sector_0_numeric[0:5]

for s in sector_code_l:
    all_factors[f'sector_code_{s}'] = (all_factors['sector_code'] == s).astype(int)

all_factors.head()
all_factors.index.get_level_values(0)

print(all_factors.index.get_level_values(0).month)
print(all_factors.index.get_level_values(0).month == 1)
print( (all_factors.index.get_level_values(0).month == 1).astype(int) )


# TODO: create a feature that indicate whether it's January
all_factors['is_January'] = (all_factors.index.get_level_values(0).month == 1).astype(int)

# TODO: create a feature to indicate whether it's December
all_factors['is_December'] = (all_factors.index.get_level_values(0).month == 12).astype(int)

# %%
set(all_factors.index.get_level_values(0).weekday)

set(all_factors.index.get_level_values(0).quarter)

all_factors['weekday'] = all_factors.index.get_level_values(0).weekday
all_factors['quarter'] = all_factors.index.get_level_values(0).quarter
all_factors['year'] = all_factors.index.get_level_values(0).year

tmp = pd.date_range(start=factor_start_date, end=universe_end_date, freq='BM')
tmp

last_day_of_month = pd.date_range(start=factor_start_date, end=universe_end_date, freq='BM')
last_day_of_month

tmp_month_end = all_factors.index.get_level_values(0).isin(last_day_of_month)
tmp_month_end
tmp_month_end_int = tmp_month_end.astype(int)
tmp_month_end_int

all_factors['month_end'] = tmp_month_end_int
first_day_of_month = pd.date_range(start=factor_start_date, end=universe_end_date, freq='BMS')
all_factors['month_start'] = (all_factors.index.get_level_values(0).isin(first_day_of_month)).astype(int)

last_day_qtr = pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQ')
all_factors['qtr_end'] = (all_factors.index.get_level_values(0).isin(last_day_qtr)).astype(int)

first_day_qtr = pd.date_range(start=factor_start_date, end=universe_end_date, freq='BQS')
all_factors['qtr_start'] = (all_factors.index.get_level_values(0).isin(first_day_qtr)).astype(int)

list(all_factors.columns)

features = ['Momentum_1YR',
 'Sharpe',
 'euel',
 'AM',
 'AM_Smoothed',
#  'StocksOnTheMove',
 'volatility_20d',
 'volatility_120d',
 'adv_20d',
 'adv_120d',
 'dispersion_20d',
 'dispersion_120d',
 'market_vol_20d',
 'market_vol_120d',
 #'sector_code',
 'Mean_Reversion_5Day_Sector_Neutral_Smoothed',
 'sector_code_0',
 'sector_code_1',
 'sector_code_2',
 'sector_code_3',
 'sector_code_4',
 'sector_code_5',
 'sector_code_6',
 'sector_code_7',
 'sector_code_8',
 'sector_code_9',
 'is_January',
 'is_December',
 'weekday',
 'quarter',
 'year',
 'month_end',
 'month_start',
 'qtr_end',
 'qtr_start']
# %%
pipeline_target = Pipeline(screen=universe)

return_20d_2q = Returns(window_length=20, mask=universe).quantiles(2)
return_20d_2q

pipeline_target.add(return_20d_2q, 'return_20d_2q')

return_20d_5q = Returns(window_length=20, mask=universe).quantiles(5)

pipeline_target.add(return_20d_5q, 'return_20d_5q')

# Let's run the pipeline to get the dataframe
targets_df = engine.run_pipeline(pipeline_target, factor_start_date, universe_end_date)
targets_df.head()


# %%
targets_df.columns


target_label = 'return_20d_5q'

all_factors.index.get_level_values(1)
# %%
targets_df.index.get_level_values(1)

# %%
# Split into training, validation and test
def split_into_sets(data, set_sizes):
    assert np.sum(set_sizes) == 1
    
    last_i = 0
    sets = []
    for set_size in set_sizes:
        set_n = int(len(data) * set_size)
        sets.append(data[last_i:last_i + set_n])
        last_i = last_i + set_n
        
    return sets

def split_by_index(df, index_level, sets):
    set_indicies = split_into_sets(df.index.levels[index_level], sets)
    
    return [df.loc[indicies[0]:indicies[-1]] for indicies in set_indicies]
                                        
# %%
# put the features and target into one dataframe before 
# running dropna, so that the rows match.
tmp = all_factors.copy()
tmp [target_label] = targets_df[target_label]
tmp = tmp.dropna()
X = tmp[features]
y = tmp[target_label]

X_train, X_valid, X_test = split_by_index(X, 0, [0.6, 0.2, 0.2])
y_train, y_valid, y_test = split_by_index(y, 0, [0.6, 0.2, 0.2])                                        

# %%

X_train.shape
# %%
y_train.shape


# %%

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
        n_estimators=10,
        max_features='sqrt',
        min_samples_split=5000,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
        criterion='entropy',
        verbose=0,
        random_state=0
    )
clf.fit(X_train, y_train)

# %%
np.argsort([10,30,20])

# %%
def model_importances(m, features):
    # get the feature importances from the model
    importances = m.feature_importances_

    # sort the importances in descending order, and store the indices of that sort
    indices = np.argsort(importances)[::-1]
    """
    Iterate through the features, starting with the ones with the highest feature importances
    """
    features_ranked = []
    for f in range(X_train.shape[1]):
        print("%d. %s (%d) (%f)" % (f+1,features[indices[f]], indices[f], importances[indices[f]]))
        features_ranked.append(features[indices[f]])

    return features_ranked
# %%
features_skl = model_importances(clf, features)

# %%
import shap

shap.initjs() #initialize javascript to enable visualizations


# %%
explainer = shap.TreeExplainer(clf)
#TODO Check why this doesn't work with additivity=True
shap_values = explainer.shap_values(X_train, 
                                    tree_limit=5,
                                    check_additivity=False)


# %%
# mean of absolute values for each feature to get global feature importance
shap.summary_plot(shap_values, X_train, plot_type="bar")

# %%
tmp1 = np.concatenate(shap_values)
tmp1.shape


# %%
tmp2 = np.abs(tmp1)
tmp2
# %%
tmp3 = np.nanmean(tmp2,axis=0)
tmp3
# %%


def model_shap_importances(model, features,X):
    """
    Note that the observations should be numeric (integer or float).
    So booleans should be converted to 1 (True) and 0 (False)
    """
    # calculate shap values
    shap_values = shap.TreeExplainer(model).shap_values(X, tree_limit=5, check_additivity=False)

    # concatenate the shap values into one matrix
    shap_values_matrix = np.concatenate(shap_values)

    # take the absolute values
    shap_abs = np.abs(shap_values_matrix)

    # Take the average for each feature (each column)
    global_importances = np.nanmean(shap_abs, axis=0)

    # get the indices sorted in descending order of global feature importance
    indices = np.argsort(global_importances)[::-1]
    features_ranked = []
    for f in range(X.shape[1]):
        print("%d. %s (%d) (%f)" % (f+1,features[indices[f]], indices[f], global_importances[indices[f]]))
        features_ranked.append(features[indices[f]])

    return features_ranked


# this will take a few seconds to run
features_ranked = model_shap_importances(clf,features,X_train)




# %%


def model_shap_importances(model, features,X):
    """
    Note that the observations should be numeric (integer or float).
    So booleans should be converted to 1 (True) and 0 (False)
    """
    # calculate shap values
    shap_values = shap.TreeExplainer(model).shap_values(X, tree_limit=5)

    # concatenate the shap values into one matrix
    shap_values_matrix = np.concatenate(shap_values)

    # take the absolute values
    shap_abs = np.abs(shap_values_matrix)

    # Take the average for each feature (each column)
    global_importances = np.nanmean(shap_abs, axis=0)

    # get the indices sorted in descending order of global feature importance
    indices = np.argsort(global_importances)[::-1]
    features_ranked = []
    for f in range(X.shape[1]):
        print("%d. %s (%d) (%f)" % (f+1,features[indices[f]], indices[f], global_importances[indices[f]]))
        features_ranked.append(features[indices[f]])

    return features_ranked


# this will take a few seconds to run
features_ranked = model_shap_importances(clf,features,X_train)


# %%
features_ranked
# %%
