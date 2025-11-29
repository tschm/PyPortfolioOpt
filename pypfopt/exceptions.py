"""
The ``exceptions`` module houses custom exceptions.

Currently implemented:

- OptimizationError
- InstantiationError
"""


class OptimizationError(Exception):
    """
    Exception raised when an optimization routine fails.

    Usually, this means that cvxpy has not returned the "optimal" flag.
    This can happen when the problem is infeasible, unbounded, or when
    the solver encounters numerical difficulties.

    Parameters
    ----------
    *args : tuple
        Variable length argument list passed to the base Exception class.
    **kwargs : dict
        Arbitrary keyword arguments passed to the base Exception class.

    Examples
    --------
    >>> from pypfopt.exceptions import OptimizationError
    >>> try:
    ...     raise OptimizationError("Custom message")
    ... except OptimizationError as e:
    ...     print("Optimization failed")
    Optimization failed
    """

    def __init__(self, *args, **kwargs):
        default_message = (
            "Please check your objectives/constraints or use a different solver."
        )
        super().__init__(default_message, *args, **kwargs)


class InstantiationError(Exception):
    """
    Exception raised for errors related to the instantiation of pypfopt objects.

    This exception is raised when attempting to perform invalid operations
    on already-solved optimization problems, such as adding constraints
    or objectives after the problem has been solved.

    Examples
    --------
    >>> from pypfopt.exceptions import InstantiationError
    >>> try:
    ...     raise InstantiationError("Cannot modify solved problem")
    ... except InstantiationError as e:
    ...     print("Instantiation error occurred")
    Instantiation error occurred
    """

    pass
