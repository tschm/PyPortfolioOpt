"""
The ``hierarchical_portfolio`` module implements hierarchical clustering portfolio methods.

This module seeks to implement one of the recent advances in portfolio optimization -
the application of hierarchical clustering models in allocation.

All of the hierarchical classes have a similar API to ``EfficientFrontier``, though since
many hierarchical models currently don't support different objectives, the actual allocation
happens with a call to ``optimize()``.

Currently implemented:

- ``HRPOpt`` implements the Hierarchical Risk Parity (HRP) portfolio. Code reproduced with
  permission from Marcos Lopez de Prado (2016).
"""

import collections

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from . import base_optimizer, risk_models


class HRPOpt(base_optimizer.BaseOptimizer):
    """
    Construct a hierarchical risk parity portfolio.

    A HRPOpt object (inheriting from BaseOptimizer) uses hierarchical clustering
    to construct a portfolio that accounts for the hierarchical structure of
    correlations between assets.

    Attributes
    ----------
    n_assets : int
        Number of assets.
    tickers : list
        List of asset tickers.
    returns : pd.DataFrame
        Historical returns data.
    cov_matrix : pd.DataFrame
        Covariance matrix.
    weights : np.ndarray
        Optimized portfolio weights.
    clusters : np.ndarray
        Linkage matrix corresponding to clustered assets.

    Examples
    --------
    >>> import pandas as pd
    >>> from pypfopt import HRPOpt, expected_returns
    >>> # prices = pd.read_csv("stock_prices.csv", index_col="date", parse_dates=True)
    >>> # returns = expected_returns.returns_from_prices(prices)
    >>> # hrp = HRPOpt(returns)
    >>> # weights = hrp.optimize()
    """

    def __init__(self, returns=None, cov_matrix=None):
        """
        Initialize the HRPOpt object.

        Parameters
        ----------
        returns : pd.DataFrame, optional
            Asset historical returns. Either returns or cov_matrix must be provided.
        cov_matrix : pd.DataFrame, optional
            Covariance of asset returns. Either returns or cov_matrix must be provided.

        Raises
        ------
        ValueError
            If neither returns nor cov_matrix is provided.
        TypeError
            If returns is not a DataFrame.
        """
        if returns is None and cov_matrix is None:
            raise ValueError("Either returns or cov_matrix must be provided")

        if returns is not None and not isinstance(returns, pd.DataFrame):
            raise TypeError("returns are not a dataframe")

        self.returns = returns
        self.cov_matrix = cov_matrix
        self.clusters = None

        if returns is None:
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(returns.columns)
        super().__init__(len(tickers), tickers)

    @staticmethod
    def _get_cluster_var(cov, cluster_items):
        """
        Compute the variance per cluster.

        Parameters
        ----------
        cov : pd.DataFrame
            Covariance matrix.
        cluster_items : list
            Tickers in the cluster.

        Returns
        -------
        float
            The variance per cluster.
        """
        # Compute variance per cluster
        cov_slice = cov.loc[cluster_items, cluster_items]
        weights = 1 / np.diag(cov_slice)  # Inverse variance weights
        weights /= weights.sum()
        return np.linalg.multi_dot((weights, cov_slice, weights))

    @staticmethod
    def _get_quasi_diag(link):
        """
        Sort clustered items by distance.

        Parameters
        ----------
        link : np.ndarray
            Linkage matrix after clustering.

        Returns
        -------
        list
            Sorted list of indices.
        """
        return sch.to_tree(link, rd=False).pre_order()

    @staticmethod
    def _raw_hrp_allocation(cov, ordered_tickers):
        """
        Compute raw HRP portfolio allocation.

        Given the clusters, compute the portfolio that minimises risk by
        recursively traversing the hierarchical tree from the top.

        Parameters
        ----------
        cov : pd.DataFrame
            Covariance matrix.
        ordered_tickers : list
            List of tickers ordered by distance.

        Returns
        -------
        pd.Series
            Raw portfolio weights.
        """
        w = pd.Series(1.0, index=ordered_tickers)
        cluster_items = [ordered_tickers]  # initialize all items in one cluster

        while len(cluster_items) > 0:
            cluster_items = [
                i[j:k]
                for i in cluster_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]  # bi-section
            # For each pair, optimize locally.
            for i in range(0, len(cluster_items), 2):
                first_cluster = cluster_items[i]
                second_cluster = cluster_items[i + 1]
                # Form the inverse variance portfolio for this pair
                first_variance = HRPOpt._get_cluster_var(cov, first_cluster)
                second_variance = HRPOpt._get_cluster_var(cov, second_cluster)
                alpha = 1 - first_variance / (first_variance + second_variance)
                w[first_cluster] *= alpha  # weight 1
                w[second_cluster] *= 1 - alpha  # weight 2
        return w

    def optimize(self, linkage_method="single"):
        """
        Construct a hierarchical risk parity portfolio.

        Uses Scipy hierarchical clustering. See `scipy.cluster.hierarchy.linkage
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_
        for more details.

        Parameters
        ----------
        linkage_method : str, optional
            Which scipy linkage method to use. Defaults to "single".
            Options include "single", "complete", "average", "weighted", "centroid",
            "median", "ward".

        Returns
        -------
        OrderedDict
            Weights for the HRP portfolio.

        Examples
        --------
        >>> from pypfopt import HRPOpt
        >>> # hrp = HRPOpt(returns)
        >>> # weights = hrp.optimize(linkage_method="ward")
        """
        if linkage_method not in sch._LINKAGE_METHODS:
            raise ValueError("linkage_method must be one recognised by scipy")

        if self.returns is None:
            cov = self.cov_matrix
            corr = risk_models.cov_to_corr(self.cov_matrix).round(6)
        else:
            corr, cov = self.returns.corr(), self.returns.cov()

        # Compute distance matrix, with ClusterWarning fix as
        # per https://stackoverflow.com/questions/18952587/

        # this can avoid some nasty floating point issues
        matrix = np.sqrt(np.clip((1.0 - corr) / 2.0, a_min=0.0, a_max=1.0))
        dist = ssd.squareform(matrix, checks=False)

        self.clusters = sch.linkage(dist, linkage_method)
        sort_ix = HRPOpt._get_quasi_diag(self.clusters)
        ordered_tickers = corr.index[sort_ix].tolist()
        hrp = HRPOpt._raw_hrp_allocation(cov, ordered_tickers)
        weights = collections.OrderedDict(hrp.sort_index())
        self.set_weights(weights)
        return weights

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0, frequency=252):
        """
        Calculate the performance of the optimal HRP portfolio.

        After optimising, calculate (and optionally print) the performance of the
        optimal portfolio. Currently calculates expected return, volatility, and the
        Sharpe ratio assuming returns are daily.

        Parameters
        ----------
        verbose : bool, optional
            Whether performance should be printed. Defaults to False.
        risk_free_rate : float, optional
            Risk-free rate of borrowing/lending. Defaults to 0.0.
            The period of the risk-free rate should correspond to the
            frequency of expected returns.
        frequency : int, optional
            Number of time periods in a year, defaults to 252 (trading days).

        Returns
        -------
        tuple
            A tuple of (expected return, volatility, Sharpe ratio).

        Raises
        ------
        ValueError
            If weights have not been calculated yet.

        Examples
        --------
        >>> from pypfopt import HRPOpt
        >>> # hrp = HRPOpt(returns)
        >>> # hrp.optimize()
        >>> # mu, sigma, sharpe = hrp.portfolio_performance(verbose=True)
        """
        if self.returns is None:
            cov = self.cov_matrix
            mu = None
        else:
            cov = self.returns.cov() * frequency
            mu = self.returns.mean() * frequency

        return base_optimizer.portfolio_performance(
            self.weights, mu, cov, verbose, risk_free_rate
        )
