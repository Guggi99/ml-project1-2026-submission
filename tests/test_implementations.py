"""Our own checks on implementations.py, on top of the official grading tests."""

import numpy as np
import pytest

import implementations

MODULES = [implementations]

Y = np.array([0.1, 0.3, 0.5])
TX = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
W0 = np.array([0.5, 1.0])

CALLS = {
    "mean_squared_error_gd": lambda m: m.mean_squared_error_gd(Y, TX, W0, 2, 0.1),
    "mean_squared_error_sgd": lambda m: m.mean_squared_error_sgd(Y, TX, W0, 2, 0.1),
    "least_squares": lambda m: m.least_squares(Y, TX),
    "ridge_regression": lambda m: m.ridge_regression(Y, TX, 0.1),
    "logistic_regression": lambda m: m.logistic_regression(
        np.array([0, 1, 1]), TX, W0, 2, 0.1
    ),
    "reg_logistic_regression": lambda m: m.reg_logistic_regression(
        np.array([0, 1, 1]), TX, 0.1, W0, 2, 0.1
    ),
}


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
@pytest.mark.parametrize("name", CALLS)
def test_returns_1d_w_and_scalar_loss(module, name):
    try:
        w, loss = CALLS[name](module)
    except NotImplementedError:
        pytest.skip(f"{module.__name__}.{name} not implemented yet")
    assert w.shape == (2,)
    assert np.ndim(loss) == 0


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_initial_w_not_modified(module):
    w0 = W0.copy()
    module.mean_squared_error_gd(Y, TX, w0, 2, 0.1)
    np.testing.assert_array_equal(w0, W0)
