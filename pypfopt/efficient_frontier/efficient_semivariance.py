"""
The ``efficient_semivariance`` submodule houses the EfficientSemivariance class.

This module generates portfolios along the mean-semivariance frontier.
"""

import cvxpy as cp
import numpy as np

from .. import objective_functions
from .efficient_frontier import EfficientFrontier


class EfficientSemivariance(EfficientFrontier):
    """
    Optimization along the mean-semivariance frontier.

    EfficientSemivariance objects allow for optimization along the
    mean-semivariance frontier. This may be relevant for users who are
    more concerned about downside deviation.

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
    >>> from pypfopt import EfficientSemivariance, expected_returns
    >>> # mu = expected_returns.mean_historical_return(prices)
    >>> # returns = expected_returns.returns_from_prices(prices)
    >>> # es = EfficientSemivariance(mu, returns)
    >>> # weights = es.min_semivariance()
    """

    def __init__(
        self,
        expected_returns,
        returns,
        frequency=252,
        benchmark=0,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        Initialize the EfficientSemivariance object.

        Parameters
        ----------
        expected_returns : pd.Series, list, or np.ndarray
            Expected returns for each asset. Can be None if
            optimising for semideviation only.
        returns : pd.DataFrame or np.ndarray
            (Historic) returns for all your assets (no NaNs).
            See ``expected_returns.returns_from_prices``.
        frequency : int, optional
            Number of time periods in a year, defaults to 252.
            This must agree with the frequency parameter used
            in your ``expected_returns``.
        benchmark : float, optional
            The return threshold to distinguish "downside" and "upside".
            This should match the frequency of your ``returns``,
            i.e., this should be a benchmark daily return if your
            ``returns`` are also daily. Defaults to 0.
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
        # Instantiate parent
        super().__init__(
            expected_returns=expected_returns,
            cov_matrix=np.zeros((returns.shape[1],) * 2),  # dummy
            weight_bounds=weight_bounds,
            solver=solver,
            verbose=verbose,
            solver_options=solver_options,
        )

        self.returns = self._validate_returns(returns)
        self.benchmark = benchmark
        self.frequency = frequency
        self._T = self.returns.shape[0]

    def min_volatility(self):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as min_volatility is not available in EfficientSemivariance.
            Use min_semivariance instead.
        """
        raise NotImplementedError("Please use min_semivariance instead.")

    def max_sharpe(self, risk_free_rate=0.0):
        """
        Override parent method.

        Raises
        ------
        NotImplementedError
            Always, as max_sharpe is not available in EfficientSemivariance.
        """
        raise NotImplementedError("Method not available in EfficientSemivariance")

    def min_semivariance(self, market_neutral=False):
        """
        Minimise portfolio semivariance (downside deviation).

        Parameters
        ----------
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the semivariance-minimising portfolio.

        Examples
        --------
        >>> from pypfopt import EfficientSemivariance
        >>> # es = EfficientSemivariance(mu, returns)
        >>> # weights = es.min_semivariance()
        """
        p = cp.Variable(self._T, nonneg=True)
        n = cp.Variable(self._T, nonneg=True)
        self._objective = cp.sum(cp.square(n))

        for obj in self._additional_objectives:
            self._objective += obj

        B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
        self.add_constraint(lambda w: B @ w - p + n == 0)
        self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def max_quadratic_utility(self, risk_aversion=1, market_neutral=False):
        """
        Maximise the quadratic utility using portfolio semivariance.

        Parameters
        ----------
        risk_aversion : float, optional
            Risk aversion parameter (must be greater than 0), defaults to 1.
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the maximum-utility portfolio.

        Raises
        ------
        ValueError
            If risk_aversion is not greater than zero.

        Examples
        --------
        >>> from pypfopt import EfficientSemivariance
        >>> # es = EfficientSemivariance(mu, returns)
        >>> # weights = es.max_quadratic_utility(risk_aversion=2)
        """
        if risk_aversion <= 0:
            raise ValueError("risk aversion coefficient must be greater than zero")

        update_existing_parameter = self.is_parameter_defined("risk_aversion")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("risk_aversion", risk_aversion)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            mu = objective_functions.portfolio_return(self._w, self.expected_returns)
            mu /= self.frequency
            risk_aversion_par = cp.Parameter(
                value=risk_aversion, name="risk_aversion", nonneg=True
            )
            self._objective = mu + 0.5 * risk_aversion_par * cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj

            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_risk(self, target_semideviation, market_neutral=False):
        """
        Maximise return for a target semideviation (downside standard deviation).

        The resulting portfolio will have a semideviation less than the target
        (but not guaranteed to be equal).

        Parameters
        ----------
        target_semideviation : float
            The desired maximum semideviation of the resulting portfolio.
        market_neutral : bool, optional
            Whether the portfolio should be market neutral (weights sum to zero),
            defaults to False. Requires negative lower weight bound.

        Returns
        -------
        OrderedDict
            Asset weights for the efficient risk portfolio.

        Examples
        --------
        >>> from pypfopt import EfficientSemivariance
        >>> # es = EfficientSemivariance(mu, returns)
        >>> # weights = es.efficient_risk(target_semideviation=0.10)
        """
        update_existing_parameter = self.is_parameter_defined("target_semivariance")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_semivariance", target_semideviation**2)
        else:
            self._objective = objective_functions.portfolio_return(
                self._w, self.expected_returns
            )
            for obj in self._additional_objectives:
                self._objective += obj

            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)

            target_semivariance = cp.Parameter(
                value=target_semideviation**2, name="target_semivariance", nonneg=True
            )
            self.add_constraint(
                lambda _: self.frequency * cp.sum(cp.square(n)) <= target_semivariance
            )
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def efficient_return(self, target_return, market_neutral=False):
        """
        Minimise semideviation for a given target return.

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
        >>> from pypfopt import EfficientSemivariance
        >>> # es = EfficientSemivariance(mu, returns)
        >>> # weights = es.efficient_return(target_return=0.15)
        """
        if not isinstance(target_return, float) or target_return < 0:
            raise ValueError("target_return should be a positive float")
        if target_return > np.abs(self.expected_returns).max():
            raise ValueError(
                "target_return must be lower than the largest expected return"
            )

        update_existing_parameter = self.is_parameter_defined("target_return")
        if update_existing_parameter:
            self._validate_market_neutral(market_neutral)
            self.update_parameter_value("target_return", target_return)
        else:
            p = cp.Variable(self._T, nonneg=True)
            n = cp.Variable(self._T, nonneg=True)
            self._objective = cp.sum(cp.square(n))
            for obj in self._additional_objectives:
                self._objective += obj

            target_return_par = cp.Parameter(name="target_return", value=target_return)
            self.add_constraint(
                lambda w: cp.sum(w @ self.expected_returns) >= target_return_par
            )
            B = (self.returns.values - self.benchmark) / np.sqrt(self._T)
            self.add_constraint(lambda w: B @ w - p + n == 0)
            self._make_weight_sum_constraint(market_neutral)
        return self._solve_cvxpy_opt_problem()

    def portfolio_performance(self, verbose=False, risk_free_rate=0.0):
        """
        Calculate the performance of the optimal portfolio.

        After optimising, calculate (and optionally print) the performance
        of the optimal portfolio, specifically: expected return, semideviation,
        Sortino ratio.

        Parameters
        ----------
        verbose : bool, optional
            Whether performance should be printed, defaults to False.
        risk_free_rate : float, optional
            Risk-free rate of borrowing/lending, defaults to 0.0.
            The period of the risk-free rate should correspond to the
            frequency of expected returns.

        Returns
        -------
        tuple
            A tuple of (expected return, semideviation, Sortino ratio).

        Raises
        ------
        ValueError
            If weights have not been calculated yet.

        Examples
        --------
        >>> from pypfopt import EfficientSemivariance
        >>> # es = EfficientSemivariance(mu, returns)
        >>> # es.min_semivariance()
        >>> # mu, semi, sortino = es.portfolio_performance(verbose=True)
        """
        mu = objective_functions.portfolio_return(
            self.weights, self.expected_returns, negative=False
        )

        portfolio_returns = self.returns @ self.weights
        drops = np.fmin(portfolio_returns - self.benchmark, 0)
        semivariance = np.sum(np.square(drops)) / self._T * self.frequency
        semi_deviation = np.sqrt(semivariance)
        sortino_ratio = (mu - risk_free_rate) / semi_deviation

        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual semi-deviation: {:.1f}%".format(100 * semi_deviation))
            print("Sortino Ratio: {:.2f}".format(sortino_ratio))

        return mu, semi_deviation, sortino_ratio
