# %%
import sys
import pandas as pd
# import cvxpy as cvx

# # Data Bundle
sys.path.insert(-1,"C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/Helper_Functions")
# import os
import helper
import helper_ZAEQ
import helper_CVXPY

import zipline
from zipline.api import attach_pipeline, record, order_target_percent, symbol, schedule_function, date_rules, time_rules
from zipline.pipeline.domain import ZA_EQUITIES
from six import viewkeys

from zipline.pipeline import Pipeline
from zipline.pipeline.factors import RSI
from zipline.finance import commission, slippage
from zipline.pipeline.factors import AverageDollarVolume
domain_za = ZA_EQUITIES

bundle_data = helper_ZAEQ.load_bundle()
pricing_loader = helper.PricingLoader(bundle_data)
  
def make_pipeline():
    tradable_universe = helper_ZAEQ.create_tradable_universe()
    universe = AverageDollarVolume(window_length=120, mask = tradable_universe).top(100) 

    rsi = RSI(mask = universe)
    return Pipeline(
        columns={'universe':universe,
                 'RSI':rsi
                 },
        domain = ZA_EQUITIES
    )
#%%
def initialize(context):
    # Set up a benchmark to measure against
    # context.set_benchmark(symbol('AGL'))
    attach_pipeline(make_pipeline(), 'my_pipeline')
    # Rebalance each day.  In daily mode, this is equivalent to putting
    # `rebalance` in our handle_data, but in minute mode, it's equivalent to
    # running at the start of the day each day.
    #TODO: Change to daily
    schedule_function(rebalance,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close())

    # Explicitly set the commission/slippage to the "old" value until we can
    # rebuild example data.
    # github.com/quantopian/zipline/blob/master/tests/resources/
    # rebuild_example_data#L105
    context.set_commission(commission.PerShare(cost=.0075, min_trade_cost=1.0))
    context.set_slippage(slippage.VolumeShareSlippage())

from zipline.api import pipeline_output

def before_trading_start(context, data):
    """
    Called every day before market open.
    """

def calculate_optimal(Close, pipeline):
    five_year_returns = Close.pct_change()[1:].fillna(0)
    num_factor_exposures = 20
    pca = helper_CVXPY.fit_pca(five_year_returns, 
                               num_factor_exposures, 
                               'full')
    # pca.components_
    risk_model = {}
    import numpy as np
    risk_model['factor_betas'] = helper_CVXPY.factor_betas(pca,
                                                           five_year_returns.columns.values,
                                                           np.arange(num_factor_exposures))

    risk_model['factor_returns'] = helper_CVXPY.factor_returns(
                                        pca,
                                        five_year_returns,
                                        five_year_returns.index,
                                        np.arange(num_factor_exposures))

    ann_factor = 252
    risk_model['factor_cov_matrix'] = pd.DataFrame(helper_CVXPY.factor_cov_matrix(risk_model['factor_returns'], ann_factor))
    risk_model['idiosyncratic_var_matrix'] = helper_CVXPY.idiosyncratic_var_matrix(five_year_returns, risk_model['factor_returns'], risk_model['factor_betas'], ann_factor)
    risk_model['idiosyncratic_var_vector'] = helper_CVXPY.idiosyncratic_var_vector(five_year_returns, risk_model['idiosyncratic_var_matrix'])
    tradeable = pipeline.index[pipeline['universe']==True]
    alpha_vector = pipeline[['RSI']].loc[tradeable]

    # alpha_vector = alphas.loc[pipeline.index]
    optimal_weights = helper_CVXPY.OptimalHoldings().find(alpha_vector,
                                                            risk_model['factor_betas'].loc[tradeable], 
                                                            risk_model['factor_cov_matrix'], 
                                                            risk_model['idiosyncratic_var_vector'].loc[tradeable])
    #  optimal_weights_1 = helper_CVXPY.OptimalHoldingsRegualization(lambda_reg=5.0).find(alpha_vector, 
    #                                                                        risk_model['factor_betas'].loc[tradeable],
    #                                                                        risk_model['factor_cov_matrix'], 
    #                                                                        risk_model['idiosyncratic_var_vector'])
    # optimal_weights_2 = helper_CVXPY.OptimalHoldingsStrictFactor(
    #     weights_max=0.02,
    #     weights_min=-0.02,
    #     risk_cap=0.0015,
    #     factor_max=0.015,
    #     factor_min=-0.015).find(alpha_vector, risk_model['factor_betas'], risk_model['factor_cov_matrix'], risk_model['idiosyncratic_var_vector'])

    return optimal_weights

def rebalance(context, data):

    # Pipeline data will be a dataframe with boolean columns named 'longs' and
    # 'shorts'.
    context.output = pipeline_output('my_pipeline')
    pipeline_data = context.output
    all_assets = pipeline_data.index
    hist = data.history(all_assets, "close", 252, "1d").dropna(how = 'all')
    optimal_weights = calculate_optimal(hist, pipeline_data)

    record(universe_size=len(all_assets))

    for asset in optimal_weights.index:
        order_target_percent(asset, optimal_weights.loc[asset].item())

    # Remove any assets that should no longer be in our portfolio.
    positions = context.portfolio.positions
    for asset in viewkeys(positions) - set(optimal_weights.index):
        # This will fail if the asset was removed from our portfolio because it
        # was delisted.
        print(asset)
        print("---")
        print("no longer in universe")
        if data.can_trade(asset):
            order_target_percent(asset, 0)



    # These are the securities that we are interested in trading each day.
    #context.security_list = context.output.index


#def rebalance(context, data):
#    """
#    Execute orders according to our schedule_function() timing.
#    """
#    for sec in context.secs:  
#        order_target_percent(sec, context.weights, limit_price=None, stop_price=None)    

#%%
start = pd.Timestamp('2014-1-3', tz='utc')
end = pd.Timestamp('2016-01-05', tz='utc')

from trading_calendars import get_calendar


za_calendar = get_calendar('XJSE')
from zipline.pipeline.loaders import USEquityPricingLoader, EquityPricingLoader

# Fire off backtest
result = zipline.run_algorithm(
    start=start, # Set start
    end=end,  # Set end
    initialize=initialize, # Define startup function
    capital_base=100000, # Set initial capital
    data_frequency = 'daily',  # Set data frequency
    bundle='ZAEQ',
    trading_calendar=za_calendar,
    custom_loader = pricing_loader) # Select bundle
result.to_csv('result '+pd.Timestamp.now().strftime('%Y-%m-%d %H%M')+'.csv')
print("Ready to analyze result.")

#%%
# Create a benchmark dataframe
def create_benchmark(fname):
    # benchmark_rets (pd.Series, optional) -- Daily noncumulative returns of the benchmark. This is in the same style as returns.
    bench = pd.read_csv('{}.csv'.format(fname), index_col='Date', parse_dates=True, date_parser=lambda col: pd.to_datetime(col, utc=True))
    # Create a series
    bench_series = pd.Series(bench['return'].values, index=bench.index)
    bench_series.rename(fname, inplace=True)
    return bench_series


bench_series = create_benchmark('STX40')


# %%
result.index = result.index.normalize() # to set the time to 00:00:00
bench_series = bench_series[bench_series.index.isin(result.index)]
# bench_series

# %%
import pyfolio as pf
def analyse(perfdata, bench_returns):
    returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(perfdata)
    pf.create_returns_tear_sheet(returns, benchmark_rets=bench_returns)
    
analyse(result, bench_series)

# %%

returns, positions, transactions = pf.utils.extract_rets_pos_txn_from_zipline(result)
pf.create_full_tear_sheet(returns, positions=positions, transactions=transactions,
                          live_start_date='2015-1-3', round_trips=True)

# %%
