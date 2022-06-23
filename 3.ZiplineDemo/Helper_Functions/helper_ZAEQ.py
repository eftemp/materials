import sys
import numpy as np
import pandas as pd
# %%
## Sector packages
from zipline.pipeline.classifiers import Classifier
from zipline.utils.numpy_utils import int64_dtype


sys.path.insert(1,"C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/Helper_Functions")
from helper import ZIPLINE_LOCATION, build_pipeline_engine, create_data_portal


EOD_BUNDLE_NAME = 'ZAEQ'

# %%
# Sector data
def Sector(bundle_name="ZAEQ"):
    if bundle_name == "ZAEQ":
        sector_link = ZIPLINE_LOCATION+'/0.Data/JSE/2018.08.31_ZAEQ/1.SECTOR/GICS_SECTOR_NAME.npy'
    elif bundle_name == "ZAEQ_INVESTING":
        sector_link = ZIPLINE_LOCATION+'/0.Data/JSE/2021.11.23_ZAEQ_INVESTING/1.SECTOR/GICS_SECTOR_NAME.npy'
    elif bundle_name == "ZAEQ2021":
        sector_link = ZIPLINE_LOCATION+'/0.Data/JSE/2021.11.25_ZAEQ/1.SECTOR/GICS_SECTOR_NAME.npy'
        
    class Sector_zip(Classifier):
        dtype=int64_dtype
        window_length=0
        window_safe=True
        inputs=()
        missing_value=-1

        def __init__(self):
            self.data=np.load(sector_link)

        def _compute(self, arrays, dates, assets, mask):
            return np.where(
                mask,
                self.data[assets],
                self.missing_value,
            )
    return(Sector_zip())
sector = Sector("ZAEQ")
sector_map = pd.read_csv('C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/0.Data/JSE/2021.11.23_ZAEQ_INVESTING/1.SECTOR/GICS_SECTOR_NAME_sector_integer_map.csv', index_col = 1)

class Sector_ZAEQ_Investing(Classifier):
    dtype=int64_dtype
    window_length=0
    window_safe=True    
    inputs=()
    missing_value=-1

    def __init__(self):
        self.link = ZIPLINE_LOCATION+'/0.Data/JSE/2021.11.23_ZAEQ_INVESTING/1.Sector/GICS_SECTOR_NAME.npy'
        self.data=np.load(self.link)

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )

class Sector_ZAEQ_new(Classifier):
    dtype=int64_dtype
    window_length=0
    window_safe=True    
    inputs=()
    missing_value=-1

    def __init__(self):
        self.data=np.load("C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/0.Data/JSE/2021.11.25_ZAEQ/1.SECTOR/GICS_SECTOR_NAME.npy")

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )
        
# %%

class Primary(Classifier):
    dtype=int64_dtype
    window_length=0
    window_safe=True    
    inputs=()
    missing_value=-1

    def __init__(self):
        self.data=np.load(ZIPLINE_LOCATION+'/0.Data/JSE/2018.08.31_ZAEQ/1.SECTOR/PRIMARY.npy')

    def _compute(self, arrays, dates, assets, mask):
        return np.where(
            mask,
            self.data[assets],
            self.missing_value,
        )

# %%
#load bundle
from zipline.data import bundles

def load_bundle(bundle_name=EOD_BUNDLE_NAME):
    ingest_func = bundles.csvdir.csvdir_equities(['daily'], bundle_name)
    bundles.register(bundle_name, ingest_func)
    bundle_data = bundles.load(bundle_name)
    return bundle_data
# %%
#build engine
from zipline.pipeline.domain import ZA_EQUITIES
domain_za = ZA_EQUITIES

def build_ZA_engine(bundle_data):
    return build_pipeline_engine(bundle_data, domain_za)

# %%
#Create tradable universe
from zipline.pipeline.data import EquityPricing
from zipline.pipeline.factors import AverageDollarVolume, Latest
from zipline.pipeline.filters import AllPresent, All

def create_tradable_universe():
    # Equities listed as common stock (not preferred stock, ETF, ADR, LP, etc)
#    common_stock = master.SecuritiesMaster.usstock_SecurityType2.latest.eq('Common Stock')

    # Filter for primary share equities; primary shares can be identified by a
    # null usstock_PrimaryShareSid field (i.e. no pointer to a primary share)
#    is_primary_share = master.SecuritiesMaster.usstock_PrimaryShareSid.latest.isnull()

    # combine the security type filters to begin forming our universe
 #   tradable_stocks = common_stock & is_primary_share

    # # dollar volume over ZARc2.5M over trailing 200 days
    tradable_stocks = AverageDollarVolume(window_length=120) >= 2.5e6
    tradable_stocks = AverageDollarVolume(window_length=120, mask = tradable_stocks).top(120)

    # also require price > $5. Note that we use Latest(...) instead of EquityPricing.close.latest
    # so that we can pass a mask
    tradable_stocks = Latest([EquityPricing.close], mask=tradable_stocks) > 500

    # also require no missing data for 200 days
    tradable_stocks = AllPresent(inputs=[EquityPricing.close], window_length=200, mask=tradable_stocks)
    tradable_stocks = All([EquityPricing.volume.latest > 2000], window_length=200, mask=tradable_stocks)
    
    return tradable_stocks