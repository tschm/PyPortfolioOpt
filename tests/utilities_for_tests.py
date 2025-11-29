"""
Utility functions and setup helpers for testing.

This module provides helper functions and setup utilities used across
the PyPortfolioOpt test suite.
"""

import os

import numpy as np
import pandas as pd

from pypfopt import expected_returns, risk_models
from pypfopt.cla import CLA
from pypfopt.efficient_frontier import (
    EfficientCDaR,
    EfficientCVaR,
    EfficientFrontier,
    EfficientSemivariance,
)


def resource(name):
    """
    Get the path to a test resource file.

    Parameters
    ----------
    name : str
        Name of the resource file.

    Returns
    -------
    str
        Full path to the resource file.
    """
    return os.path.join(os.path.dirname(__file__), "resources", name)


def get_data():
    """
    Load stock price test data.

    Returns
    -------
    pd.DataFrame
        Stock prices with date index and ticker columns.
    """
    return pd.read_csv(resource("stock_prices.csv"), parse_dates=True, index_col="date")


def get_benchmark_data():
    """
    Load benchmark (SPY) price test data.

    Returns
    -------
    pd.DataFrame
        SPY prices with date index.
    """
    return pd.read_csv(resource("spy_prices.csv"), parse_dates=True, index_col="date")


def get_market_caps():
    """
    Get sample market capitalization data.

    Returns
    -------
    dict
        Dictionary mapping tickers to market caps in USD.
    """
    mcaps = {
        "GOOG": 927e9,
        "AAPL": 1.19e12,
        "FB": 574e9,
        "BABA": 533e9,
        "AMZN": 867e9,
        "GE": 96e9,
        "AMD": 43e9,
        "WMT": 339e9,
        "BAC": 301e9,
        "GM": 51e9,
        "T": 61e9,
        "UAA": 78e9,
        "SHLD": 0,
        "XOM": 295e9,
        "RRC": 1e9,
        "BBY": 22e9,
        "MA": 288e9,
        "PFE": 212e9,
        "JPM": 422e9,
        "SBUX": 102e9,
    }
    return mcaps


def get_cov_matrix():
    """
    Load pre-computed covariance matrix test data.

    Returns
    -------
    pd.DataFrame
        Covariance matrix with ticker index and columns.
    """
    return pd.read_csv(resource("cov_matrix.csv"), index_col=0)


def setup_efficient_frontier(data_only=False, *args, **kwargs):
    """
    Set up an EfficientFrontier object for testing.

    Parameters
    ----------
    data_only : bool, optional
        If True, return only the data (mu, S) without creating EfficientFrontier.
        Defaults to False.
    *args : tuple
        Additional positional arguments for EfficientFrontier.
    **kwargs : dict
        Additional keyword arguments for EfficientFrontier.

    Returns
    -------
    EfficientFrontier or tuple
        If data_only is False, returns an EfficientFrontier object.
        If data_only is True, returns a tuple of (mean_return, sample_cov_matrix).
    """
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return mean_return, sample_cov_matrix
    return EfficientFrontier(
        mean_return, sample_cov_matrix, verbose=True, *args, **kwargs
    )


def setup_efficient_semivariance(data_only=False, *args, **kwargs):
    """
    Set up an EfficientSemivariance object for testing.

    Parameters
    ----------
    data_only : bool, optional
        If True, return only the data (mu, returns) without creating object.
        Defaults to False.
    *args : tuple
        Additional positional arguments for EfficientSemivariance.
    **kwargs : dict
        Additional keyword arguments for EfficientSemivariance.

    Returns
    -------
    EfficientSemivariance or tuple
        If data_only is False, returns an EfficientSemivariance object.
        If data_only is True, returns a tuple of (mean_return, historic_returns).
    """
    df = get_data().dropna(axis=0, how="any")
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return mean_return, historic_returns
    return EfficientSemivariance(
        mean_return, historic_returns, verbose=True, *args, **kwargs
    )


def setup_efficient_cvar(data_only=False, *args, **kwargs):
    """
    Set up an EfficientCVaR object for testing.

    Parameters
    ----------
    data_only : bool, optional
        If True, return only the data (mu, returns) without creating object.
        Defaults to False.
    *args : tuple
        Additional positional arguments for EfficientCVaR.
    **kwargs : dict
        Additional keyword arguments for EfficientCVaR.

    Returns
    -------
    EfficientCVaR or tuple
        If data_only is False, returns an EfficientCVaR object.
        If data_only is True, returns a tuple of (mean_return, historic_returns).
    """
    df = get_data().dropna(axis=0, how="any")
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return mean_return, historic_returns
    return EfficientCVaR(mean_return, historic_returns, verbose=True, *args, **kwargs)


def setup_efficient_cdar(data_only=False, *args, **kwargs):
    """
    Set up an EfficientCDaR object for testing.

    Parameters
    ----------
    data_only : bool, optional
        If True, return only the data (mu, returns) without creating object.
        Defaults to False.
    *args : tuple
        Additional positional arguments for EfficientCDaR.
    **kwargs : dict
        Additional keyword arguments for EfficientCDaR.

    Returns
    -------
    EfficientCDaR or tuple
        If data_only is False, returns an EfficientCDaR object.
        If data_only is True, returns a tuple of (mean_return, historic_returns).
    """
    df = get_data().dropna(axis=0, how="any")
    mean_return = expected_returns.mean_historical_return(df)
    historic_returns = expected_returns.returns_from_prices(df)
    if data_only:
        return mean_return, historic_returns
    return EfficientCDaR(mean_return, historic_returns, verbose=True, *args, **kwargs)


def setup_cla(data_only=False, *args, **kwargs):
    """
    Set up a CLA object for testing.

    Parameters
    ----------
    data_only : bool, optional
        If True, return only the data (mu, S) without creating CLA.
        Defaults to False.
    *args : tuple
        Additional positional arguments for CLA.
    **kwargs : dict
        Additional keyword arguments for CLA.

    Returns
    -------
    CLA or tuple
        If data_only is False, returns a CLA object.
        If data_only is True, returns a tuple of (mean_return, sample_cov_matrix).
    """
    df = get_data()
    mean_return = expected_returns.mean_historical_return(df)
    sample_cov_matrix = risk_models.sample_cov(df)
    if data_only:
        return mean_return, sample_cov_matrix
    return CLA(mean_return, sample_cov_matrix, *args, **kwargs)


def simple_ef_weights(expected_returns, cov_matrix, target_return, weights_sum):
    """
    Calculate efficient frontier weights using Lagrangian method.

    This is a simple test utility for calculating weights to achieve a
    target return on the efficient frontier. The only constraint is the
    sum of the weights.

    Note: This is just a simple test utility, it does not support the
    generalised constraints that EfficientFrontier does and is used to
    check the results of EfficientFrontier in simple cases. In particular,
    it is not capable of preventing negative weights (shorting).

    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns for each asset.
    cov_matrix : np.ndarray
        Covariance of returns for each asset.
    target_return : float
        The target return for the portfolio to achieve.
    weights_sum : float
        The sum of the returned weights, optimization constraint.

    Returns
    -------
    np.ndarray
        Weight for each asset, which sum to weights_sum.
    """
    # Solve using Lagrangian and matrix inversion.
    r = expected_returns.reshape((-1, 1))
    m = np.block(
        [
            [cov_matrix, r, np.ones(r.shape)],
            [r.transpose(), 0, 0],
            [np.ones(r.shape).transpose(), 0, 0],
        ]
    )
    y = np.block([[np.zeros(r.shape)], [target_return], [weights_sum]])
    x = np.linalg.inv(m) @ y
    # Weights are all but the last 2 elements, which are the lambdas.
    w = x.flatten()[:-2]
    return w
