#!/usr/bin/env python
# coding: utf-8

# # Trade Optimization

# cf: https://github.com/jamesdellinger/ai_for_trading_nanodegree_alpha_research_multi_factor_modeling_project
# To do: 
# - use this to identify factors
# - implement unit tests
# - backtest factors which show promise

# In[1]:


import sys
import cvxpy as cvx
import numpy as np
import pandas as pd
import time

sys.path.insert(1,"C:/Users/User/OneDrive/Documents/Finance/Quant/Zipline/Helper_Functions")

import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)


# # Data Bundle

# In[2]:


import os
import helper
from zipline.data import bundles
from trading_calendars import get_calendar
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline import Pipeline
from zipline.pipeline.factors import AverageDollarVolume

# ingest_func = bundles.csvdir.csvdir_equities(['daily'], helper.EOD_BUNDLE_NAME)
# bundles.register(helper.EOD_BUNDLE_NAME, ingest_func)

from zipline.pipeline.domain import ZA_EQUITIES
# bundle_data = bundles.load('ZAEQ')
# domain_za = ZA_EQUITIES
# engine = helper.build_pipeline_engine(bundle_data, domain_za)


# ## View Data

# In[3]:


# universe_end_date = pd.Timestamp('2016-01-05', tz='UTC')
# universe = AverageDollarVolume(window_length=120).top(100) 

# universe_tickers = engine    .run_pipeline(
#         Pipeline(screen=universe),
#         universe_end_date,
#         universe_end_date)\
#     .index.get_level_values(1)\
#     .values.tolist()


# ## Returns

# In[4]:


# from zipline.data.data_portal import DataPortal
# from trading_calendars import get_calendar
# trading_calendar = get_calendar('XJSE')

# data_portal = DataPortal(
#     bundle_data.asset_finder,
#     trading_calendar=trading_calendar,
#     first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
#     equity_minute_reader=None,
#     equity_daily_reader=bundle_data.equity_daily_bar_reader,
#     adjustment_reader=bundle_data.adjustment_reader)

# def get_pricing(data_portal, trading_calendar, assets, start_date, end_date, field='close'):
#     end_dt = pd.Timestamp(end_date.strftime('%Y-%m-%d'), tz='UTC')
#     start_dt = pd.Timestamp(start_date.strftime('%Y-%m-%d'), tz='UTC')

#     end_loc = trading_calendar.closes.index.get_loc(end_dt)
#     start_loc = trading_calendar.closes.index.get_loc(start_dt)

#     return data_portal.get_history_window(
#         assets=assets,
#         end_dt=end_dt,
#         bar_count=end_loc - start_loc,
#         frequency='1d',
#         field=field,
#         data_frequency='daily')


# In[5]:


# five_year_returns =     get_pricing(
#         data_portal,
#         trading_calendar,
#         universe_tickers,
#         universe_end_date - pd.DateOffset(years=5),
#         universe_end_date)\
#     .pct_change()[1:].fillna(0)


# ## Fit PCA

# In[6]:


from sklearn.decomposition import PCA

def fit_pca(returns, num_factor_exposures, svd_solver):
    """
    Fit PCA model with returns.

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    num_factor_exposures : int
        Number of factors for PCA
    svd_solver: str
        The solver to use for the PCA model

    Returns
    -------
    pca : PCA
        Model fit to returns
    """

    pca = PCA(n_components = num_factor_exposures, 
              svd_solver = svd_solver)
    pca.fit(returns)
    
    return pca


# In[7]:




# In[8]:


# plt.bar(np.arange(num_factor_exposures), pca.explained_variance_ratio_)


# ## Factor Betas

# In[9]:


def factor_betas(pca, factor_beta_indices, factor_beta_columns):
    """
    Get the factor betas from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    factor_beta_indices : 1 dimensional Ndarray
        Factor beta indices
    factor_beta_columns : 1 dimensional Ndarray
        Factor beta columns

    Returns
    -------
    factor_betas : DataFrame
        Factor betas
    """
    assert len(factor_beta_indices.shape) == 1
    assert len(factor_beta_columns.shape) == 1

    return pd.DataFrame(pca.components_.T,
                        factor_beta_indices,
                        factor_beta_columns)


# In[10]:




# ## Factor Returns

# In[62]:


def factor_returns(pca, returns, factor_return_indices, factor_return_columns):
    """
    Get the factor returns from the PCA model.

    Parameters
    ----------
    pca : PCA
        Model fit to returns
    returns : DataFrame
        Returns for each ticker and date
    factor_return_indices : 1 dimensional Ndarray
        Factor return indices
    factor_return_columns : 1 dimensional Ndarray
        Factor return columns

    Returns
    -------
    factor_returns : DataFrame
        Factor returns
    """
    assert len(factor_return_indices.shape) == 1
    assert len(factor_return_columns.shape) == 1

    return pd.DataFrame(pca.transform(returns.values),
                        factor_return_indices,
                        factor_return_columns)


# In[63]:




# ## Factor Covariance Matrix

# In[13]:


def factor_cov_matrix(factor_returns, ann_factor):
    """
    Get the factor covariance matrix

    Parameters
    ----------
    factor_returns : DataFrame
        Factor returns
    ann_factor : int
        Annualization factor

    Returns
    -------
    factor_cov_matrix : DataFrame
        Factor covariance matrix
    """
    return np.diag(factor_returns.var(axis=0,ddof=1)*ann_factor)


# In[14]:




# ## Idiosyncratic Variance Matrix

# In[15]:


def idiosyncratic_var_matrix(returns, factor_returns, factor_betas, ann_factor):
    """
    Get the idiosyncratic variance matrix

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    factor_returns : DataFrame
        Factor returns
    factor_betas : DataFrame
        Factor betas
    ann_factor : int
        Annualization factor

    Returns
    -------
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    """
    common_returns_=pd.DataFrame(np.dot(factor_returns,factor_betas.T),
                                 returns.index,
                                 returns.columns)
    residuals_=returns-common_returns_
    return pd.DataFrame(np.diag(np.var(residuals_))*ann_factor,
                        returns.columns,
                        returns.columns)


# In[16]:




# ## Idiosyncratic Variance Vector

# In[17]:


def idiosyncratic_var_vector(returns, idiosyncratic_var_matrix):
    """
    Get the idiosyncratic variance vector

    Parameters
    ----------
    returns : DataFrame
        Returns for each ticker and date
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix

    Returns
    -------
    idiosyncratic_var_vector : DataFrame
        Idiosyncratic variance Vector
    """
    return pd.DataFrame(np.diagonal(idiosyncratic_var_matrix), returns.columns)


# In[18]:




# ## Predict using the model risk

# In[19]:


def predict_portfolio_risk(factor_betas, factor_cov_matrix, idiosyncratic_var_matrix, weights):
    """
    Get the predicted portfolio risk
    
    Formula for predicted portfolio risk is sqrt(X.T(BFB.T + S)X) where:
      X is the portfolio weights
      B is the factor betas
      F is the factor covariance matrix
      S is the idiosyncratic variance matrix

    Parameters
    ----------
    factor_betas : DataFrame
        Factor betas
    factor_cov_matrix : 2 dimensional Ndarray
        Factor covariance matrix
    idiosyncratic_var_matrix : DataFrame
        Idiosyncratic variance matrix
    weights : DataFrame
        Portfolio weights

    Returns
    -------
    predicted_portfolio_risk : float
        Predicted portfolio risk
    """
    assert len(factor_cov_matrix.shape) == 2
    return np.sqrt(weights.T.dot(factor_betas.dot(factor_cov_matrix.dot(factor_betas.T)) 
                                 + idiosyncratic_var_matrix).dot(weights)).iloc[0][0]


# %%
# # Optimal Portfolio Constrained by Risk Model

from abc import ABC, abstractmethod

class AbstractOptimalHoldings(ABC):    
    @abstractmethod
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        
        raise NotImplementedError()
    
    @abstractmethod
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        
        raise NotImplementedError()
        
    def _get_risk(self, weights, factor_betas, alpha_vector_index, factor_cov_matrix, idiosyncratic_var_vector):
        f = factor_betas.loc[alpha_vector_index].values.T @ weights
        X = factor_cov_matrix
        S = np.diag(idiosyncratic_var_vector.loc[alpha_vector_index].values.flatten())

        return cvx.quad_form(f, X) + cvx.quad_form(weights, S)
    
    def find(self, alpha_vector, factor_betas, factor_cov_matrix, idiosyncratic_var_vector):
        weights = cvx.Variable(len(alpha_vector))
        risk = self._get_risk(weights, factor_betas, alpha_vector.index, factor_cov_matrix, idiosyncratic_var_vector)
        
        obj = self._get_obj(weights, alpha_vector)
        constraints = self._get_constraints(weights, factor_betas.loc[alpha_vector.index].values, risk)
        
        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iters=500)

        optimal_weights = np.asarray(weights.value).flatten()
        
        return pd.DataFrame(data=optimal_weights, index=alpha_vector.index)


# In[68]:


class OptimalHoldings(AbstractOptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        return cvx.Minimize(-alpha_vector.T.values[0]@weights)
    
    def _get_constraints(self, weights, factor_betas, risk):
        """
        Get the constraints

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        factor_betas : 2 dimensional Ndarray
            Factor betas
        risk: CVXPY Atom
            Predicted variance of the portfolio returns

        Returns
        -------
        constraints : List of CVXPY Constraint
            Constraints
        """
        assert(len(factor_betas.shape) == 2)
        
        return [risk<=self.risk_cap**2,factor_betas.T@weights<=self.factor_max,factor_betas.T@weights>=self.factor_min,
                sum(weights)==0,sum(cvx.abs(weights))<=1,weights>=self.weights_min,weights<=self.weights_max]

    def __init__(self, risk_cap=0.175, factor_max=10.0, factor_min=-10.0, weights_max=0.512, weights_min=-0.12):
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


# ## Optimize with a Regularization Parameter

# In[43]:


class OptimalHoldingsRegualization(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)                
        return cvx.Minimize(-alpha_vector.T.values[0]@weights + self.lambda_reg*cvx.pnorm(weights,2))

    def __init__(self, lambda_reg=0.5, risk_cap=0.15, factor_max=10.0, factor_min=-10.0, weights_max=0.15, weights_min=-0.15):
        self.lambda_reg = lambda_reg
        self.risk_cap=risk_cap
        self.factor_max=factor_max
        self.factor_min=factor_min
        self.weights_max=weights_max
        self.weights_min=weights_min


# In[44]:

# ## Optimize with a Strict Factor Constraints and Target Weighting

# In[46]:


class OptimalHoldingsStrictFactor(OptimalHoldings):
    def _get_obj(self, weights, alpha_vector):
        """
        Get the objective function

        Parameters
        ----------
        weights : CVXPY Variable
            Portfolio weights
        alpha_vector : DataFrame
            Alpha vector

        Returns
        -------
        objective : CVXPY Objective
            Objective function
        """
        assert(len(alpha_vector.columns) == 1)
        
        alpha_vec_vals=alpha_vector.values[:,0]
        x_star=(alpha_vec_vals-np.mean(alpha_vec_vals))/sum(abs(alpha_vec_vals))
        return cvx.Minimize(cvx.pnorm(weights-x_star,2))

