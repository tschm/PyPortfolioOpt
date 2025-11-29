"""
The ``efficient_cvar`` submodule houses the EfficientCVaR class.

This module generates portfolios along the mean-CVaR frontier.
"""

import warnings

import cvxpy as cp
import numpy as np

from .. import objective_functions
from .efficient_frontier import EfficientFrontier


class EfficientCVaR(EfficientFrontier):
    """
    Optimization along the mean-CVaR (Conditional Value at Risk) frontier.

    The EfficientCVaR class allows for optimization along the mean-CVaR
    frontier, using the formulation of Rockafellar and Ursayev (2001).

    Attributes
    ----------
    n_assets : int
        Number of assets.
    tickers : list
        List of asset tickers.
    bounds : tuple or list
        Weight bounds for each asset.
    returns : pd.DataFrame
        Historical returns data.
    expected_returns : np.ndarray
        Expected returns for each asset.
    solver : str
        CVXPY solver name.
    solver_options : dict
        Solver parameters.
    weights : np.ndarray
        Optimized portfolio weights.

    Examples
    --------
    >>> from pypfopt import EfficientCVaR, expected_returns
    >>> # mu = expected_returns.mean_historical_return(prices)
    >>> # returns = expected_returns.returns_from_prices(prices)
    >>> # ef = EfficientCVaR(mu, returns)
    >>> # weights = ef.min_cvar()
    """

    def __init__(
        self,
        expected_returns,
        returns,
        beta=0.95,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        Initialize the EfficientCVaR object.

        Parameters
        ----------
        expected_returns : pd.Series, list, or np.ndarray
            Expected returns for each asset. Can be None if
            optimising for conditional value at risk only.
        returns : pd.DataFrame or np.ndarray
            (Historic) returns for all your assets (no NaNs).
            See ``expected_returns.returns_from_prices``.
        beta : float, optional
            Confidence level, defaults to 0.95 (i.e expected loss
            on the worst (1-beta) days).
        weight_bounds : tuple or list, optional
            Minimum and maximum weight of each asset OR single min/max pair
            if all identical, defaults to (0, 1). Must be changed to (-1, 1)
            for portfolios with shorting.
        solver : str, optional
            Name of solver. List available solvers with ``cvxpy.installed_solvers()``.
        verbose : bool, optional
            Whether performance and debugging info should be printed,
            defaults to False.
        solver_options : dict, optional
            Parameters for the given solver.

        Raises
        ------
        TypeError
            If ``expected_returns`` is not a series, list or array.
        """
        super().__init__(
            expected_returns=expected_returns,
            cov_matrix=np.zeros((returns.shape[1],) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        self.returns = self._validate_returns(returns)
        self._beta = self._validate_beta(beta)
        self._alpha = cp.Variable()
        self._u = cp.Variable(len(self.returns))

    def set_weights(self, input_weights):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as set_weights is not available in EfficientCVaR.
        """
        raise NotImplementedError("Method not available in EfficientCVaR.")

    @staticmethod
    def _validate_beta(beta):
        """
        Validate the beta (confidence level) parameter.

        Parameters
        ----------
        beta : float
            Confidence level.

        Returns
        -------
        float
            Validated beta value.

        Raises
        ------
        ValueError
            If beta is not between 0 and 1.
        """
        if not (0 <= beta < 1):
            raise ValueError("beta must be between 0 and 1")
        if beta <= 0.2:
            warnings.warn(
                "Warning: beta is the confidence-level, not the quantile. Typical values are 80%, 90%, 95%.",
                UserWarning,
            )
        return beta

    def min_volatility(self):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as min_volatility is not available in EfficientCVaR.
            Use min_cvar instead.
        """
        raise NotImplementedError("Please use min_cvar instead.")

    def max_sharpe(self, risk_free_rate=0.0):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as max_sharpe is not available in EfficientCVaR.
        """
        raise NotImplementedError("Method not available in EfficientCVaR.")

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as max_quadratic_utility is not available in EfficientCVaR.
        """
        raise NotImplementedError("Method not available in EfficientCVaR.")

    def min_cvar(self, market_neutral=False):
        """
        Minimise portfolio CVaR (Conditional Value at Risk).

        Parameters
        ----------
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the CVaR-minimising portfolio.

        Examples
        --------
        >>> from pypfopt import EfficientCVaR, expected_returns
        >>> # mu = expected_returns.mean_historical_return(prices)
        >>> # returns = expected_returns.returns_from_prices(prices)
        >>> # ef = EfficientCVaR(mu, returns)
        >>> # weights = ef.min_cvar()
        """
        self._objective = self._alpha + 1.0 / (
            len(self.returns) * (1 - self._beta)
        ) * cp.sum(self._u)

        for obj in self._additional_objectives:
            self._objective += obj

        self.add_constraint(lambda _: self._u >= 0.0)
        self.add_constraint(
            lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
        )

        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """
        Minimise CVaR for a given target return.

        Parameters
        ----------
        target_return : float
            The desired return of the resulting portfolio.
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the optimal portfolio.

        Raises
        ------
        ValueError
            If ``target_return`` is not a positive float.
            If no portfolio can be found with return equal to ``target_return``.

        Examples
        --------
        >>> from pypfopt import EfficientCVaR
        >>> # ef = EfficientCVaR(mu, returns)
        >>> # weights = ef.efficient_return(target_return=0.15)
        """
        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            self._objective = self._alpha + 1.0 / (
                len(self.returns) * (1 - self._beta)
            ) * cp.sum(self._u)

            for obj in self._additional_objectives:
                self._objective += obj

            self.add_constraint(lambda _: self._u >= 0.0)
            self.add_constraint(
                lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
            )

            ret = self.expected_returns.T @ self._w
            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(lambda _: ret >= target_return_par)

            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_cvar, market_neutral=False):
        """
        Maximise return for a target CVaR.

        The resulting portfolio will have a CVaR less than the target
        (but not guaranteed to be equal).

        Parameters
        ----------
        target_cvar : float
            The desired conditional value at risk of the resulting portfolio.
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the efficient risk portfolio.

        Examples
        --------
        >>> from pypfopt import EfficientCVaR
        >>> # ef = EfficientCVaR(mu, returns)
        >>> # weights = ef.efficient_risk(target_cvar=0.10)
        """
        update_existing_parameter = self.is_parameter_defined("target_cvar")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_cvar", target_cvar)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            for obj in self._additional_objectives:
                self._objective += obj

            cvar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(
                self._u
            )
            target_cvar_par = cp.Parameter(
                value=target_cvar, name="target_cvar", nonneg=True
            )

            self.add_constraint(lambda _: cvar <= target_cvar_par)
            self.add_constraint(lambda _: self._u >= 0.0)
            self.add_constraint(
                lambda w: self.returns.values @ w + self._alpha + self._u >= 0.0
            )

            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False):
        """
        Calculate the performance of the optimal portfolio.

        After optimising, calculate (and optionally print) the performance
        of the optimal portfolio, specifically: expected return, CVaR.

        Parameters
        ----------
        verbose : bool, optional
            Whether performance should be printed, defaults to False.

        Returns
        -------
        tuple
            A tuple of (expected return, CVaR).

        Raises
        ------
        ValueError
            If weights have not been calculated yet.

        Examples
        --------
        >>> from pypfopt import EfficientCVaR
        >>> # ef = EfficientCVaR(mu, returns)
        >>> # ef.min_cvar()
        >>> # mu, cvar = ef.portfolio_performance(verbose=True)
        """
        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )

        cvar = self._alpha + 1.0 / (len(self.returns) * (1 - self._beta)) * cp.sum(
            self._u
        )
        cvar_val = cvar.value

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Conditional Value at Risk: {:.2f}%".format(100 * cvar_val))

        return mu, cvar_val
