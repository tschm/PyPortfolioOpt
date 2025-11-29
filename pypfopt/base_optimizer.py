"""
The ``base_optimizer`` module houses the parent classes for optimization.

This module contains the ``BaseOptimizer`` class from which all optimizers
inherit, and ``BaseConvexOptimizer`` which is the base class for all ``cvxpy``
(and ``scipy``) optimization.

Additionally, we define a general utility function ``portfolio_performance``
to evaluate return and risk for a given set of portfolio weights.
"""

import collections
from collections.abc import Iterable
import copy
import json
from typing import List
import warnings

import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.optimize as sco

from . import exceptions, objective_functions


class BaseOptimizer:
    """
    Base class for all portfolio optimizers.

    Attributes
    ----------
    n_assets : int
        Number of assets in the portfolio.
    tickers : list
        List of asset tickers/identifiers.
    weights : np.ndarray or None
        Optimized portfolio weights, None before optimization.

    Examples
    --------
    >>> from pypfopt.base_optimizer import BaseOptimizer
    >>> opt = BaseOptimizer(n_assets=5, tickers=['A', 'B', 'C', 'D', 'E'])
    """

    def __init__(self, n_assets, tickers=None):
        """
        Initialize the BaseOptimizer.

        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        tickers : list, optional
            List of asset names/tickers. If None, uses integer indices.
        """
        self.n_assets = n_assets
        if tickers is None:
            self.tickers = list(range(n_assets))
        else:
            self.tickers = tickers
        self._risk_free_rate = None
        # Outputs
        self.weights = None

    def _make_output_weights(self, weights=None):
        """
        Create output weight dictionary from weight array.

        Utility function to make output weight dict from weight attribute
        (np.array). If no arguments passed, use self.tickers and self.weights.
        If one argument is passed, assume it is an alternative weight array
        so use self.tickers and the argument.

        Parameters
        ----------
        weights : np.ndarray, optional
            Alternative weight array to use instead of self.weights.

        Returns
        -------
        OrderedDict
            Dictionary mapping tickers to weights.
        """
        if weights is None:
            weights = self.weights

        # Convert numpy float64 to plain Python float
        weights = [float(w) for w in weights]

        return collections.OrderedDict(zip(self.tickers, weights))

    def set_weights(self, input_weights):
        """
        Set weights attribute from a dictionary.

        Parameters
        ----------
        input_weights : dict
            Dictionary mapping tickers to weights, e.g., {'AAPL': 0.3, 'GOOG': 0.7}.
        """
        self.weights = np.array([input_weights[ticker] for ticker in self.tickers])

    def clean_weights(self, cutoff=1e-4, rounding=5):
        """
        Clean the raw weights by rounding and clipping near-zeros.

        Helper method to clean the raw weights, setting any weights whose
        absolute values are below the cutoff to zero, and rounding the rest.

        Parameters
        ----------
        cutoff : float, optional
            The lower bound for weights. Weights with absolute values below
            this threshold are set to zero. Defaults to 1e-4.
        rounding : int or None, optional
            Number of decimal places to round the weights. Set to None if
            rounding is not desired. Defaults to 5.

        Returns
        -------
        OrderedDict
            Cleaned asset weights.

        Raises
        ------
        AttributeError
            If weights have not been computed yet.
        ValueError
            If rounding is not a positive integer.

        Examples
        --------
        >>> from pypfopt import EfficientFrontier
        >>> # ef = EfficientFrontier(mu, S)
        >>> # ef.min_volatility()
        >>> # clean_weights = ef.clean_weights()
        """
        if self.weights is None:
            raise AttributeError("Weights not yet computed")
        clean_weights = self.weights.copy()
        clean_weights[np.abs(clean_weights) < cutoff] = 0
        if rounding is not None:
            if not isinstance(rounding, int) or rounding < 1:
                raise ValueError("rounding must be a positive integer")
            clean_weights = np.round(clean_weights, rounding)

        return self._make_output_weights(clean_weights)

    def save_weights_to_file(self, filename="weights.csv"):
        """
        Save optimized weights to a file.

        Utility method to save weights to a text file.

        Parameters
        ----------
        filename : str, optional
            Name of file. Should be csv, json, or txt. Defaults to "weights.csv".

        Raises
        ------
        NotImplementedError
            If the file extension is not csv, json, or txt.

        Examples
        --------
        >>> from pypfopt import EfficientFrontier
        >>> # ef = EfficientFrontier(mu, S)
        >>> # ef.min_volatility()
        >>> # ef.save_weights_to_file("my_weights.json")
        """
        clean_weights = self.clean_weights()

        ext = filename.split(".")[-1].lower()
        if ext == "csv":
            pd.Series(clean_weights).to_csv(filename, header=False)
        elif ext == "json":
            with open(filename, "w") as fp:
                json.dump(clean_weights, fp)
        elif ext == "txt":
            with open(filename, "w") as f:
                f.write(str(dict(clean_weights)))
        else:
            raise NotImplementedError("Only supports .txt .json .csv")


class BaseConvexOptimizer(BaseOptimizer):
    """
    Base class for convex portfolio optimization using cvxpy.

    The BaseConvexOptimizer contains many private variables for use by
    ``cvxpy``. For example, the immutable optimization variable for weights
    is stored as self._w. Interacting directly with these variables directly
    is discouraged.

    Attributes
    ----------
    n_assets : int
        Number of assets in the portfolio.
    tickers : list
        List of asset tickers/identifiers.
    weights : np.ndarray or None
        Optimized portfolio weights.

    Notes
    -----
    This class provides the foundation for convex optimization methods
    including min_volatility, max_sharpe, and efficient_frontier methods.
    """

    def __init__(
        self,
        n_assets,
        tickers=None,
        weight_bounds=(0, 1),
        solver=None,
        verbose=False,
        solver_options=None,
    ):
        """
        Initialize the BaseConvexOptimizer.

        Parameters
        ----------
        n_assets : int
            Number of assets in the portfolio.
        tickers : list, optional
            List of asset names/tickers.
        weight_bounds : tuple or list, optional
            Minimum and maximum weight of each asset OR single min/max pair
            if all identical. Defaults to (0, 1). Must be changed to (-1, 1)
            for portfolios with shorting.
        solver : str, optional
            Name of solver. List available solvers with ``cvxpy.installed_solvers()``.
        verbose : bool, optional
            Whether performance and debugging info should be printed. Defaults to False.
        solver_options : dict, optional
            Additional parameters for the given solver.
        """
        super().__init__(n_assets, tickers)

        # Optimization variables
        self._w = cp.Variable(n_assets)
        self._objective = None
        self._additional_objectives = []
        self._constraints = []
        self._lower_bounds = None
        self._upper_bounds = None
        self._opt = None
        self._solver = solver
        self._verbose = verbose
        self._solver_options = solver_options if solver_options else {}
        self._map_bounds_to_constraints(weight_bounds)

    def deepcopy(self):
        """
        Return a custom deep copy of the optimizer.

        This is necessary because ``cvxpy`` expressions do not support deepcopy,
        but the mutable arguments need to be copied to avoid unintended side effects.
        Instead, we create a shallow copy of the optimizer and then manually copy
        the mutable arguments.

        Returns
        -------
        BaseConvexOptimizer
            A copy of the optimizer with copied mutable arguments.
        """
        self_copy = copy.copy(self)
        self_copy._additional_objectives = [
            copy.copy(obj) for obj in self_copy._additional_objectives
        ]
        self_copy._constraints = [copy.copy(con) for con in self_copy._constraints]
        return self_copy

    def _map_bounds_to_constraints(self, test_bounds):
        """
        Convert input bounds into cvxpy constraints.

        Parameters
        ----------
        test_bounds : tuple or list
            Minimum and maximum weight of each asset OR single min/max pair
            if all identical OR pair of arrays corresponding to lower/upper bounds.

        Raises
        ------
        TypeError
            If test_bounds is not of the right type.
        """
        # If it is a collection with the right length, assume they are all bounds.
        if len(test_bounds) == self.n_assets and not isinstance(
            test_bounds[0], (float, int)
        ):
            bounds = np.array(test_bounds, dtype=float)
            self._lower_bounds = np.nan_to_num(bounds[:, 0], nan=-np.inf)
            self._upper_bounds = np.nan_to_num(bounds[:, 1], nan=np.inf)
        else:
            # Otherwise this must be a pair.
            if len(test_bounds) != 2 or not isinstance(test_bounds, (tuple, list)):
                raise TypeError(
                    "test_bounds must be a pair (lower bound, upper bound) OR a collection of bounds for each asset"
                )
            lower, upper = test_bounds

            # Replace None values with the appropriate +/- 1
            if np.isscalar(lower) or lower is None:
                lower = -1 if lower is None else lower
                self._lower_bounds = np.array([lower] * self.n_assets)
                upper = 1 if upper is None else upper
                self._upper_bounds = np.array([upper] * self.n_assets)
            else:
                self._lower_bounds = np.nan_to_num(lower, nan=-1)
                self._upper_bounds = np.nan_to_num(upper, nan=1)

        self.add_constraint(lambda w: w >= self._lower_bounds)
        self.add_constraint(lambda w: w <= self._upper_bounds)

    def is_parameter_defined(self, parameter_name: str) -> bool:
        """
        Check if a named parameter is defined in the optimization problem.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to check.

        Returns
        -------
        bool
            True if the parameter is defined, False otherwise.

        Raises
        ------
        InstantiationError
            If the parameter name is defined multiple times.
        """
        is_defined = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name and not is_defined:
                    is_defined = True
                elif param.name() == parameter_name and is_defined:
                    raise exceptions.InstantiationError(
                        "Parameter name defined multiple times"
                    )
        return is_defined

    def update_parameter_value(self, parameter_name: str, new_value: float) -> None:
        """
        Update the value of a named parameter in the optimization problem.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter to update.
        new_value : float
            The new value for the parameter.

        Raises
        ------
        InstantiationError
            If the parameter has not been defined or was not updated.
        """
        if not self.is_parameter_defined(parameter_name):
            raise exceptions.InstantiationError("Parameter has not been defined")
        was_updated = False
        objective_and_constraints = (
            self._constraints + [self._objective]
            if self._objective is not None
            else self._constraints
        )
        for expr in objective_and_constraints:
            params = [
                arg for arg in _get_all_args(expr) if isinstance(arg, cp.Parameter)
            ]
            for param in params:
                if param.name() == parameter_name:
                    param.value = new_value
                    was_updated = True
        if not was_updated:
            raise exceptions.InstantiationError("Parameter was not updated")

    def _solve_cvxpy_opt_problem(self):
        """
        Solve the cvxpy optimization problem.

        Helper method to solve the cvxpy problem and check output,
        once objectives and constraints have been defined.

        Returns
        -------
        OrderedDict
            The optimized portfolio weights.

        Raises
        ------
        OptimizationError
            If the problem is not solvable by cvxpy.
        InstantiationError
            If objectives or constraints were changed after initial optimization.
        """
        try:
            if self._opt is None:
                self._opt = cp.Problem(cp.Minimize(self._objective), self._constraints)
                self._initial_objective = self._objective.id
                self._initial_constraint_ids = {const.id for const in self._constraints}
            else:
                if not self._objective.id == self._initial_objective:
                    raise exceptions.InstantiationError(
                        "The objective function was changed after the initial optimization. "
                        "Please create a new instance instead."
                    )

                constr_ids = {const.id for const in self._constraints}
                if not constr_ids == self._initial_constraint_ids:
                    raise exceptions.InstantiationError(
                        "The constraints were changed after the initial optimization. "
                        "Please create a new instance instead."
                    )
            self._opt.solve(
                solver=self._solver, verbose=self._verbose, **self._solver_options
            )

        except (TypeError, cp.DCPError) as e:
            raise exceptions.OptimizationError from e

        if self._opt.status not in {"optimal", "optimal_inaccurate"}:
            raise exceptions.OptimizationError(
                "Solver status: {}".format(self._opt.status)
            )
        self.weights = self._w.value.round(16) + 0.0  # +0.0 removes signed zero
        return self._make_output_weights()

    def add_objective(self, new_objective, **kwargs):
        """
        Add a new term into the objective function.

        This term must be convex and built from cvxpy atomic functions.

        Parameters
        ----------
        new_objective : callable
            A function that takes the weight variable and returns a cvxpy expression.
            Signature should be (w, **kwargs) -> cp.Expression.
        **kwargs : dict
            Additional arguments to pass to the objective function.

        Raises
        ------
        InstantiationError
            If called after the problem has already been solved.

        Examples
        --------
        >>> import cvxpy as cp
        >>> from pypfopt import EfficientFrontier
        >>> def L1_norm(w, k=1):
        ...     return k * cp.norm(w, 1)
        >>> # ef = EfficientFrontier(mu, S)
        >>> # ef.add_objective(L1_norm, k=2)
        >>> # ef.min_volatility()
        """
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding objectives to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of objectives."
            )
        self._additional_objectives.append(new_objective(self._w, **kwargs))

    def add_constraint(self, new_constraint):
        """
        Add a new constraint to the optimization problem.

        This constraint must satisfy DCP rules, i.e., be either a linear
        equality constraint or convex inequality constraint.

        Parameters
        ----------
        new_constraint : callable
            A callable (e.g., lambda function) that takes the weight variable
            and returns a cvxpy constraint expression.

        Raises
        ------
        TypeError
            If new_constraint is not callable.
        InstantiationError
            If called after the problem has already been solved.

        Examples
        --------
        >>> from pypfopt import EfficientFrontier
        >>> import numpy as np
        >>> # ef = EfficientFrontier(mu, S)
        >>> # ef.add_constraint(lambda x: x[0] == 0.02)
        >>> # ef.add_constraint(lambda x: x >= 0.01)
        >>> # ef.add_constraint(lambda x: x <= np.array([0.01, 0.08, ..., 0.5]))
        """
        if not callable(new_constraint):
            raise TypeError(
                "New constraint must be provided as a callable (e.g lambda function)"
            )
        if self._opt is not None:
            raise exceptions.InstantiationError(
                "Adding constraints to an already solved problem might have unintended consequences. "
                "A new instance should be created for the new set of constraints."
            )
        self._constraints.append(new_constraint(self._w))

    def add_sector_constraints(self, sector_mapper, sector_lower, sector_upper):
        """
        Adds constraints on the sum of weights of different groups of assets.
        Most commonly, these will be sector constraints e.g portfolio's exposure to
        tech must be less than x%::

            sector_mapper = {
                "GOOG": "tech",
                "FB": "tech",,
                "XOM": "Oil/Gas",
                "RRC": "Oil/Gas",
                "MA": "Financials",
                "JPM": "Financials",
            }

            sector_lower = {"tech": 0.1}  # at least 10% to tech
            sector_upper = {
                "tech": 0.4, # less than 40% tech
                "Oil/Gas": 0.1 # less than 10% oil and gas
            }

        :param sector_mapper: dict that maps tickers to sectors
        :type sector_mapper: {str: str} dict
        :param sector_lower: lower bounds for each sector
        :type sector_lower: {str: float} dict
        :param sector_upper: upper bounds for each sector
        :type sector_upper: {str:float} dict
        """
        if np.any(self._lower_bounds < 0):
            warnings.warn(
                "Sector constraints may not produce reasonable results if shorts are allowed."
            )
        for sector in sector_upper:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) <= sector_upper[sector])
        for sector in sector_lower:
            is_sector = [sector_mapper.get(t) == sector for t in self.tickers]
            self.add_constraint(lambda w: cp.sum(w[is_sector]) >= sector_lower[sector])

    def convex_objective(self, custom_objective, weights_sum_to_one=True, **kwargs):
        """
        Optimize a custom convex objective function.

        Constraints should be added with ``add_constraint()``. Optimizer arguments
        must be passed as keyword-args.

        Parameters
        ----------
        custom_objective : callable
            An objective function to be MINIMISED. This should be written using
            cvxpy atoms. Signature: (w, **kwargs) -> cp.Expression.
        weights_sum_to_one : bool, optional
            Whether to add the constraint that weights sum to one. Defaults to True.
        **kwargs : dict
            Arguments to pass to the objective function.

        Returns
        -------
        OrderedDict
            Asset weights for the optimal portfolio.

        Raises
        ------
        OptimizationError
            If the objective is nonconvex or constraints are nonlinear.

        Examples
        --------
        >>> import cvxpy as cp
        >>> from pypfopt import EfficientFrontier
        >>> def logarithmic_barrier(w, cov_matrix, k=0.1):
        ...     # 60 Years of Portfolio Optimization, Kolm et al (2014)
        ...     return cp.quad_form(w, cov_matrix) - k * cp.sum(cp.log(w))
        >>> # ef = EfficientFrontier(mu, S)
        >>> # w = ef.convex_objective(logarithmic_barrier, cov_matrix=ef.cov_matrix)
        """
        # custom_objective must have the right signature (w, **kwargs)
        self._objective = custom_objective(self._w, **kwargs)

        for obj in self._additional_objectives:
            self._objective += obj

        if weights_sum_to_one:
            self.add_constraint(lambda w: cp.sum(w) == 1)

        return self._solve_cvxpy_opt_problem()

    def nonconvex_objective(
        self,
        custom_objective,
        objective_args=None,
        weights_sum_to_one=True,
        constraints=None,
        solver="SLSQP",
        initial_guess=None,
    ):
        """
        Optimize a nonconvex objective using the scipy backend.

        This method can support nonconvex objectives and nonlinear constraints,
        but may get stuck at local minima. Use with caution.

        Parameters
        ----------
        custom_objective : callable
            An objective function to be MINIMISED. This function should map
            (weight, args) -> cost.
        objective_args : tuple, optional
            Arguments for the objective function (excluding weight).
        weights_sum_to_one : bool, optional
            Whether to add the constraint that weights sum to one. Defaults to True.
        constraints : list of dict, optional
            List of constraints in the scipy format.
        solver : str, optional
            Which scipy solver to use, e.g., "SLSQP", "COBYLA", "BFGS".
            User beware: different optimizers require different inputs. Defaults to "SLSQP".
        initial_guess : np.ndarray, optional
            The initial guess for the weights, shape (n,) or (n, 1).
            Defaults to equal weights.

        Returns
        -------
        OrderedDict
            Asset weights that optimize the custom objective.

        Examples
        --------
        >>> import numpy as np
        >>> from pypfopt import EfficientFrontier
        >>> # Market-neutral efficient risk
        >>> # target_risk = 0.15
        >>> # constraints = [
        >>> #     {"type": "eq", "fun": lambda w: np.sum(w)},  # weights sum to zero
        >>> #     {"type": "eq", "fun": lambda w: target_risk ** 2 - np.dot(w.T, np.dot(ef.cov_matrix, w))},
        >>> # ]
        >>> # ef = EfficientFrontier(mu, S)
        >>> # ef.nonconvex_objective(
        >>> #     lambda w, mu: -w.T.dot(mu),
        >>> #     objective_args=(ef.expected_returns,),
        >>> #     weights_sum_to_one=False,
        >>> #     constraints=constraints,
        >>> # )
        """
        # Sanitise inputs
        if not isinstance(objective_args, tuple):
            objective_args = (objective_args,)

        # Make scipy bounds
        bound_array = np.vstack((self._lower_bounds, self._upper_bounds)).T
        bounds = list(map(tuple, bound_array))

        if initial_guess is None:
            initial_guess = np.array([1 / self.n_assets] * self.n_assets)

        # Construct constraints
        final_constraints = []
        if weights_sum_to_one:
            final_constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})
        if constraints is not None:
            final_constraints += constraints

        result = sco.minimize(
            custom_objective,
            x0=initial_guess,
            args=objective_args,
            method=solver,
            bounds=bounds,
            constraints=final_constraints,
        )
        self.weights = result["x"]
        return self._make_output_weights()


def portfolio_performance(
    weights, expected_returns, cov_matrix, verbose=False, risk_free_rate=0.0
):
    """
    Calculate the performance of a portfolio.

    Calculate (and optionally print) the performance metrics of a portfolio
    given weights, expected returns, and a covariance matrix. Currently
    calculates expected return, volatility, and the Sharpe ratio.

    Parameters
    ----------
    weights : list, np.ndarray, or dict
        Portfolio weights for each asset.
    expected_returns : np.ndarray or pd.Series
        Expected returns for each asset. Can be None if optimising for
        volatility only (but not recommended).
    cov_matrix : np.ndarray or pd.DataFrame
        Covariance of returns for each asset.
    verbose : bool, optional
        Whether performance should be printed. Defaults to False.
    risk_free_rate : float, optional
        Risk-free rate of borrowing/lending. Defaults to 0.0.
        The period of the risk-free rate should correspond to the
        frequency of expected returns.

    Returns
    -------
    tuple
        A tuple of (expected return, volatility, Sharpe ratio).
        If expected_returns is None, returns (None, volatility, None).

    Raises
    ------
    ValueError
        If weights have not been calculated yet.
        If weights add to zero or ticker names don't match.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt.base_optimizer import portfolio_performance
    >>> weights = np.array([0.3, 0.4, 0.3])
    >>> expected_returns = np.array([0.10, 0.12, 0.08])
    >>> cov_matrix = np.array([[0.04, 0.01, 0.02],
    ...                        [0.01, 0.09, 0.01],
    ...                        [0.02, 0.01, 0.16]])
    >>> mu, sigma, sharpe = portfolio_performance(weights, expected_returns, cov_matrix)
    """
    if isinstance(weights, dict):
        if isinstance(expected_returns, pd.Series):
            tickers = list(expected_returns.index)
        elif isinstance(cov_matrix, pd.DataFrame):
            tickers = list(cov_matrix.columns)
        else:
            tickers = list(range(len(expected_returns)))
        new_weights = np.zeros(len(tickers))

        for i, k in enumerate(tickers):
            if k in weights:
                new_weights[i] = weights[k]
        if new_weights.sum() == 0:
            raise ValueError("Weights add to zero, or ticker names don't match")
    elif weights is not None:
        new_weights = np.asarray(weights)
    else:
        raise ValueError("Weights is None")

    sigma = np.sqrt(objective_functions.portfolio_variance(new_weights, cov_matrix))

    if expected_returns is not None:
        mu = objective_functions.portfolio_return(
            new_weights, expected_returns, negative=False
        )

        sharpe = objective_functions.sharpe_ratio(
            new_weights,
            expected_returns,
            cov_matrix,
            risk_free_rate=risk_free_rate,
            negative=False,
        )
        if verbose:
            print("Expected annual return: {:.1f}%".format(100 * mu))
            print("Annual volatility: {:.1f}%".format(100 * sigma))
            print("Sharpe Ratio: {:.2f}".format(sharpe))
        return mu, sigma, sharpe
    else:
        if verbose:
            print("Annual volatility: {:.1f}%".format(100 * sigma))
        return None, sigma, None


def _get_all_args(expression: cp.Expression) -> List[cp.Expression]:
    """
    Recursively get all arguments from a cvxpy expression.

    Parameters
    ----------
    expression : cp.Expression
        Input cvxpy expression.

    Returns
    -------
    list of cp.Expression
        A list of all cvxpy arguments in the expression.
    """
    if expression.args == []:
        return [expression]
    else:
        return list(_flatten([_get_all_args(arg) for arg in expression.args]))


def _flatten(alist: Iterable) -> Iterable:
    """
    Flatten a nested iterable.

    Parameters
    ----------
    alist : Iterable
        A potentially nested iterable.

    Yields
    ------
    object
        Individual elements from the flattened iterable.
    """
    for v in alist:
        if isinstance(v, Iterable) and not isinstance(v, (str, bytes)):
            yield from _flatten(v)
        else:
            yield v
