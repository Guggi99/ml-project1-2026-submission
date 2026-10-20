"""The six methods required by CS-433 Project 1 (auto-graded).

Contract, from the project description:
- every function returns ``(w, loss)``: the LAST weight vector and its loss;
- the loss of regularized methods does NOT include the penalty term;
- MSE carries a factor 0.5; SGD uses mini-batch size 1;
- logistic methods expect ``y`` in {0, 1};
- all vectors are 1-D arrays of shape (D,), never (D, 1);
- ``np.linalg.lstsq`` is not allowed.

Only NumPy and the standard library may be imported here.
"""

import numpy as np


def compute_mse(y, tx, w):
    """Mean squared error of ``w``, with the factor 0.5 of the lecture notes.

    Shared by every least-squares method here, so the loss of a model is
    always measured the same way.

    Residuals of [1, 2] give 0.5 * mean([1, 4]) = 1.25:

    >>> import numpy as np
    >>> float(compute_mse(np.array([1.0, 2.0]), np.eye(2), np.zeros(2)))
    1.25
    """
    # Residuals: e = y - Xw
    # MSE loss: L(w) = (1 / (2N)) * sum_i(e_i^2)
    err = y - tx @ w
    return 0.5 * np.mean(err**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of :func:`compute_mse` with respect to ``w``.

    >>> import numpy as np
    >>> compute_mse_gradient(np.array([1.0, 2.0]), np.eye(2), np.zeros(2))
    array([-0.5, -1. ])
    """
    # Residuals: e = y - Xw
    # Gradient: grad L(w) = -(1 / N) * X.T @ e
    err = y - tx @ w
    return -(tx.T @ err) / y.shape[0]


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.


    >>> import numpy as np
    >>> y = np.array([0.1, 0.3, 0.5])
    >>> tx = np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])
    >>> w, loss = mean_squared_error_gd(y, tx, np.array([0.5, 1.0]), 2, 0.1)
    >>> np.round(w, 6)
    array([-0.050586,  0.203718])
    >>> round(float(loss), 6)
    0.051534
    """
    # MSE loss: L(w) = (1 / (2N)) * ||y - Xw||^2
    # Gradient: grad L(w) = -(1 / N) * X.T @ (y - Xw)
    # GD update: w_(t+1) = w_t - gamma * grad L(w_t)
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)

    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    One training row is sampled for every update. A local fixed random seed
    makes repeated calls reproducible without changing NumPy's global random
    state. The returned loss is evaluated on the complete dataset.

    >>> import numpy as np
    >>> y = np.array([0.1])
    >>> tx = np.array([[2.3, 3.2]])
    >>> w, loss = mean_squared_error_sgd(y, tx, np.array([0.5, 1.0]), 2, 0.1)
    >>> np.round(w, 6)
    array([0.063058, 0.39208 ])
    >>> round(float(loss), 6)
    0.844595
    """
    y = np.asarray(y)
    tx = np.asarray(tx)
    w = np.asarray(initial_w, dtype=float).copy()

    rng = np.random.default_rng(42)
    for _ in range(max_iters):
        index = rng.integers(y.shape[0])  # mini-batch size must be 1 for the project
        y_batch = y[index : index + 1]
        tx_batch = tx[index : index + 1]
        w = w - gamma * compute_mse_gradient(y_batch, tx_batch, w)

    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using normal equations.

    Three points fitted by a line w0 + w1 * x, checkable by hand:
    tx.T @ tx = [[3, 3], [3, 5]] and tx.T @ y = [7, 10] give w = [5/6, 3/2];
    the residuals [1/6, -1/3, 1/6] give 0.5 * mean(residual**2) = 1/36.

    >>> import numpy as np
    >>> tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> w, loss = least_squares(y, tx)
    >>> np.round(w, 6)
    array([0.833333, 1.5     ])
    >>> round(float(loss), 6)
    0.027778
    """
    gram = tx.T @ tx
    rhs = tx.T @ y
    w = np.linalg.solve(gram, rhs)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.

    The penalty enters as 2 * N * lambda_ * I (the course convention). With
    the data of least_squares, N = 3 and lambda_ = 1/6 make it exactly I:
    [[4, 3], [3, 6]] @ w = [7, 10] gives w = [4/5, 19/15]. The loss is the
    plain MSE at that w, without the penalty term.

    >>> import numpy as np
    >>> tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    >>> y = np.array([1.0, 2.0, 4.0])
    >>> w, loss = ridge_regression(y, tx, 1 / 6)
    >>> np.round(w, 6)
    array([0.8     , 1.266667])
    >>> round(float(loss), 6)
    0.081481

    With lambda_ = 0 ridge is plain least squares:

    >>> w, loss = ridge_regression(y, tx, 0.0)
    >>> np.round(w, 6)
    array([0.833333, 1.5     ])
    """
    n_samples, n_features = tx.shape
    regularizer = 2.0 * n_samples * lambda_ * np.eye(n_features)
    w = np.linalg.solve(tx.T @ tx + regularizer, tx.T @ y)
    return w, compute_mse(y, tx, w)


def sigmoid(t):
    """Return the logistic sigmoid, evaluated without numerical overflow.

    >>> np.round(sigmoid(np.array([-1000.0, 0.0, 1000.0])), 6)
    array([0. , 0.5, 1. ])
    """
    t = np.asarray(t, dtype=float)
    result = np.empty_like(t)
    nonnegative = t >= 0

    result[nonnegative] = 1.0 / (1.0 + np.exp(-t[nonnegative]))
    exp_t = np.exp(t[~nonnegative])
    result[~nonnegative] = exp_t / (1.0 + exp_t)
    return result


def compute_logistic_loss(y, tx, w):
    """Return the mean binary cross-entropy for labels in ``{0, 1}``.

    ``logaddexp(0, score) - y * score`` is equivalent to the usual
    cross-entropy expression but stays finite for extreme scores.
    """
    scores = tx @ w
    return np.mean(np.logaddexp(0.0, scores) - y * scores)


def compute_logistic_gradient(y, tx, w):
    """Return the gradient of the mean binary cross-entropy loss."""
    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Zero steps return initial_w unchanged; at w = 0 every probability is 1/2,
    so the loss is log(2):

    >>> import numpy as np
    >>> tx = np.array([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]])
    >>> y = np.array([0, 1, 1, 1])
    >>> w, loss = logistic_regression(y, tx, np.zeros(2), 0, 1.0)
    >>> w
    array([0., 0.])
    >>> round(float(loss), 6)
    0.693147

    One step with gamma = 1, by hand: sigmoid(tx @ 0) - y = [1/2, -1/2, -1/2, -1/2],
    the gradient tx.T @ that / 4 = [-1/4, -1/2], so w = [1/4, 1/2]:

    >>> w, loss = logistic_regression(y, tx, np.zeros(2), 1, 1.0)
    >>> np.round(w, 6)
    array([0.25, 0.5 ])
    >>> round(float(loss), 6)
    0.462903
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)

    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Penalty term is lambda_ * ||w||^2; the returned loss excludes it.
    Its gradient is 2 * lambda_ * w, added to the logistic gradient.

    Zero steps: the loss at w = [1, 1] is the plain logistic loss, NOT
    increased by the penalty lambda_ * ||w||^2 = 1:

    >>> import numpy as np
    >>> tx = np.array([[1.0, -2.0], [1.0, -1.0], [1.0, 1.0], [1.0, 2.0]])
    >>> y = np.array([0, 1, 1, 1])
    >>> w0 = np.array([1.0, 1.0])
    >>> w, loss = reg_logistic_regression(y, tx, 0.5, w0, 0, 0.5)
    >>> round(float(loss), 6)
    0.295481

    One step with gamma = 0.5: the penalty gradient 2 * 0.5 * w0 = [1, 1]
    pulls w towards 0 by gamma * [1, 1] = [0.5, 0.5] compared with plain
    logistic regression (which would give [1.049711, 1.031492]):

    >>> w, loss = reg_logistic_regression(y, tx, 0.5, w0, 1, 0.5)
    >>> np.round(w, 6)
    array([0.549711, 0.531492])
    >>> round(float(loss), 6)
    0.406752
    """
    w = np.asarray(initial_w, dtype=float).copy()

    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * gradient

    # The project requires the returned loss to exclude the penalty term.
    return w, compute_logistic_loss(y, tx, w)
