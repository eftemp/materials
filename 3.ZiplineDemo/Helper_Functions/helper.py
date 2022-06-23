
# %%
# Import packages 
## Generic packages
import numpy as np
import pandas as pd
## PricingLoader packages
from zipline.pipeline.loaders import EquityPricingLoader
from zipline.assets._assets import Equity  # Required for EquityPricing
from zipline.pipeline.data import EquityPricing
from zipline.data.fx import ExplodingFXRateReader
# DataPortal packages
from zipline.data.data_portal import DataPortal
# Research environment packages
from zipline.pipeline.engine import SimplePipelineEngine

# %%
# Constants
ZIPLINE_LOCATION = 'C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline'

# %%
# Generic PricingLoader
class PricingLoader(object):
    def __init__(self, bundle_data):
        self.loader = EquityPricingLoader(
            bundle_data.equity_daily_bar_reader,
            bundle_data.adjustment_reader,
            ExplodingFXRateReader())

    def get_loader(self, column):
        #TODO: Fix exception handling below
        # if column not in EquityPricing.columns:
        # raise Exception('Column not in EquityPricing')
        return self.loader

    def get(self, column):
        # if column not in EquityPricing.columns:
        # raise Exception('Column not in EquityPricing')
        return self.loader

# %%
def create_data_portal(bundle_data, trading_calendar):
    return DataPortal(
        bundle_data.asset_finder,
        trading_calendar=trading_calendar,
        first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
        equity_minute_reader=None,
        equity_daily_reader=bundle_data.equity_daily_bar_reader,
        adjustment_reader=bundle_data.adjustment_reader)

# %%
# Pricing functions
def get_pricing(data_portal, trading_calendar, assets,
                start_date, end_date, field='close'):
    end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC')
    start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC')

    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)

    return data_portal.get_history_window(
        assets=assets,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily')

# %%
# Pipeline engine for research environment
def build_pipeline_engine(bundle_data, domain):
    pricing_loader = PricingLoader(bundle_data)

    engine = SimplePipelineEngine(
        get_loader=pricing_loader.get_loader,
        asset_finder=bundle_data.asset_finder,
        default_domain=domain)

    return engine

