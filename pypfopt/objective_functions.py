"""
The ``objective_functions`` module provides optimization objectives.

This module contains the actual objective functions called by the
``EfficientFrontier`` object's optimization methods. These methods are primarily
designed for internal use during optimization and each requires a different
signature (which is why they have not been factored into a class).

For obvious reasons, any objective function must accept ``weights`` as an argument,
and must also have at least one of ``expected_returns`` or ``cov_matrix``.

The objective functions either compute the objective given a numpy array of weights,
or they return a cvxpy *expression* when weights are a ``cp.Variable``. In this way,
the same objective function can be used both internally for optimization and
externally for computing the objective given weights. ``_objective_value()``
automatically chooses between the two behaviours.

``objective_functions`` defaults to objectives for minimisation. In the cases of
objectives that clearly should be maximised (e.g Sharpe Ratio, portfolio return),
the objective function actually returns the negative quantity, since minimising
the negative is equivalent to maximising the positive. This behaviour is controlled
by the ``negative=True`` optional argument.

Currently implemented:

- Portfolio variance (i.e square of volatility)
- Portfolio return
- Sharpe ratio
- L2 regularisation (minimising this reduces nonzero weights)
- Quadratic utility
- Transaction cost model (a simple one)
- Ex-ante (squared) tracking error
- Ex-post (squared) tracking error
"""

import cvxpy as cp
import numpy as np


def _objective_value(w, obj):
    """
    Return the value of an objective function or the expression itself.

    Helper method to return either the value of the objective function
    or the objective function as a cvxpy object depending on whether
    w is a cvxpy variable or np array.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Portfolio weights.
    obj : cp.Expression
        Objective function expression.

    Returns
    -------
    float or cp.Expression
        The numerical value of the objective if w is a numpy array,
        or the cvxpy expression if w is a cvxpy Variable.
    """
    if isinstance(w, np.ndarray):
        if np.isscalar(obj):
            return obj
        elif np.isscalar(obj.value):
            return obj.value
        else:
            return obj.value.item()
    else:
        return obj


def portfolio_variance(w, cov_matrix):
    """
    Calculate the total portfolio variance (i.e square of volatility).

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.

    Returns
    -------
    float or cp.Expression
        The portfolio variance value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.02, 0.01, 0.16]])
    >>> variance = objective_functions.portfolio_variance(weights, cov)
    """
    variance = cp.quad_form(w, cov_matrix, assume_PSD=True)
    return _objective_value(w, variance)


def portfolio_return(w, expected_returns, negative=True):
    """
    Calculate the (negative) mean return of a portfolio.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    expected_returns : np.ndarray
        Expected return of each asset.
    negative : bool, optional
        Whether the quantity should be made negative (so we can minimise).
        Defaults to True.

    Returns
    -------
    float or cp.Expression
        The (negative) portfolio return value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> expected_returns = np.array([0.10, 0.15, 0.12])
    >>> ret = objective_functions.portfolio_return(weights, expected_returns, negative=False)
    """
    sign = -1 if negative else 1
    mu = w @ expected_returns
    return _objective_value(w, sign * mu)


def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.0, negative=True):
    """
    Calculate the (negative) Sharpe ratio of a portfolio.

    The Sharpe ratio is defined as (portfolio_return - risk_free_rate) / portfolio_volatility.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    expected_returns : np.ndarray
        Expected return of each asset.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.
    risk_free_rate : float, optional
        Risk-free rate of borrowing/lending, defaults to 0.0.
        The period of the risk-free rate should correspond to the
        frequency of expected returns.
    negative : bool, optional
        Whether the quantity should be made negative (so we can minimise).
        Defaults to True.

    Returns
    -------
    float or cp.Expression
        The (negative) Sharpe ratio value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> expected_returns = np.array([0.10, 0.15, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.02, 0.01, 0.16]])
    >>> sharpe = objective_functions.sharpe_ratio(weights, expected_returns, cov, negative=False)
    """
    mu = w @ expected_returns
    sigma = cp.sqrt(cp.quad_form(w, cov_matrix, assume_PSD=True))
    sign = -1 if negative else 1
    sharpe = (mu - risk_free_rate) / sigma
    return _objective_value(w, sign * sharpe)


def L2_reg(w, gamma=1):
    r"""
    L2 regularisation to increase the number of nonzero weights.

    This adds a penalty term :math:`\gamma ||w||^2` to the objective function,
    which encourages the optimizer to spread weights more evenly across assets.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    gamma : float, optional
        L2 regularisation parameter, defaults to 1. Increase if you want more
        non-negligible weights.

    Returns
    -------
    float or cp.Expression
        The L2 regularisation term value or cvxpy expression.

    Examples
    --------
    >>> from pypfopt import EfficientFrontier, objective_functions
    >>> from pypfopt import expected_returns, risk_models
    >>> # Assuming you have price data in a DataFrame called 'prices'
    >>> # mu = expected_returns.mean_historical_return(prices)
    >>> # S = risk_models.sample_cov(prices)
    >>> # ef = EfficientFrontier(mu, S)
    >>> # ef.add_objective(objective_functions.L2_reg, gamma=2)
    >>> # ef.min_volatility()
    """
    L2_reg = gamma * cp.sum_squares(w)
    return _objective_value(w, L2_reg)


def quadratic_utility(w, expected_returns, cov_matrix, risk_aversion, negative=True):
    r"""
    Quadratic utility function.

    The quadratic utility is defined as :math:`\mu - \frac{1}{2} \delta w^T \Sigma w`,
    where :math:`\mu` is the portfolio return, :math:`\delta` is the risk aversion
    coefficient, and :math:`\Sigma` is the covariance matrix.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    expected_returns : np.ndarray
        Expected return of each asset.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.
    risk_aversion : float
        Risk aversion coefficient. Increase to reduce risk.
    negative : bool, optional
        Whether the quantity should be made negative (so we can minimise).
        Defaults to True.

    Returns
    -------
    float or cp.Expression
        The (negative) quadratic utility value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> expected_returns = np.array([0.10, 0.15, 0.12])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.02, 0.01, 0.16]])
    >>> utility = objective_functions.quadratic_utility(
    ...     weights, expected_returns, cov, risk_aversion=1, negative=False
    ... )
    """
    sign = -1 if negative else 1
    mu = w @ expected_returns
    variance = cp.quad_form(w, cov_matrix, assume_PSD=True)

    risk_aversion_par = cp.Parameter(
        value=risk_aversion, name="risk_aversion", nonneg=True
    )
    utility = mu - 0.5 * risk_aversion_par * variance
    return _objective_value(w, sign * utility)


def transaction_cost(w, w_prev, k=0.001):
    """
    A simple transaction cost model.

    This model sums all the weight changes and multiplies by a given fraction
    (default 10bps). This simulates a fixed percentage commission from your broker.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        New asset weights in the portfolio.
    w_prev : np.ndarray
        Previous asset weights.
    k : float, optional
        Fractional cost per unit weight exchanged, defaults to 0.001 (10 basis points).

    Returns
    -------
    float or cp.Expression
        The transaction cost value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> old_weights = np.array([0.25, 0.25, 0.5])
    >>> new_weights = np.array([0.3, 0.3, 0.4])
    >>> cost = objective_functions.transaction_cost(new_weights, old_weights, k=0.001)
    """
    return _objective_value(w, k * cp.norm(w - w_prev, 1))


def ex_ante_tracking_error(w, cov_matrix, benchmark_weights):
    r"""
    Calculate the (square of) the ex-ante Tracking Error.

    The ex-ante tracking error is defined as
    :math:`(w - w_b)^T \Sigma (w-w_b)`,
    where :math:`w` is the portfolio weights, :math:`w_b` is the benchmark weights,
    and :math:`\Sigma` is the covariance matrix.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    cov_matrix : np.ndarray
        Covariance matrix of asset returns.
    benchmark_weights : np.ndarray
        Asset weights in the benchmark portfolio.

    Returns
    -------
    float or cp.Expression
        The ex-ante tracking error squared value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> benchmark = np.array([0.33, 0.33, 0.34])
    >>> cov = np.array([[0.04, 0.01, 0.02],
    ...                 [0.01, 0.09, 0.01],
    ...                 [0.02, 0.01, 0.16]])
    >>> te = objective_functions.ex_ante_tracking_error(weights, cov, benchmark)
    """
    relative_weights = w - benchmark_weights
    tracking_error = cp.quad_form(relative_weights, cov_matrix)
    return _objective_value(w, tracking_error)


def ex_post_tracking_error(w, historic_returns, benchmark_returns):
    r"""
    Calculate the (square of) the ex-post Tracking Error.

    The ex-post tracking error is defined as :math:`Var(r - r_b)`,
    where :math:`r` is the portfolio returns and :math:`r_b` is the benchmark returns.

    Parameters
    ----------
    w : np.ndarray or cp.Variable
        Asset weights in the portfolio.
    historic_returns : np.ndarray
        Historic asset returns, with shape (n_periods, n_assets).
    benchmark_returns : pd.Series or np.ndarray
        Historic benchmark returns with shape (n_periods,).

    Returns
    -------
    float or cp.Expression
        The ex-post tracking error squared value or cvxpy expression.

    Examples
    --------
    >>> import numpy as np
    >>> from pypfopt import objective_functions
    >>> weights = np.array([0.3, 0.3, 0.4])
    >>> returns = np.array([[0.01, 0.02, 0.015],
    ...                     [0.02, 0.01, 0.02],
    ...                     [-0.01, 0.03, 0.01]])
    >>> benchmark = np.array([0.015, 0.015, 0.01])
    >>> te = objective_functions.ex_post_tracking_error(weights, returns, benchmark)
    """
    if not isinstance(historic_returns, np.ndarray):
        historic_returns = np.array(historic_returns)
    if not isinstance(benchmark_returns, np.ndarray):
        benchmark_returns = np.array(benchmark_returns)

    x_i = w @ historic_returns.T - benchmark_returns
    mean = cp.sum(x_i) / len(benchmark_returns)
    tracking_error = cp.sum_squares(x_i - mean)
    return _objective_value(w, tracking_error)
