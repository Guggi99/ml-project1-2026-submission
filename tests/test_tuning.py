"""Small-data checks for issue #12's tuning helpers."""

import numpy as np

import implementations
from src.tuning import (
    fit_logistic_checkpoints,
    model_config,
    newton_logistic_reference,
    regularized_logistic_objective,
    select_near_best,
)

RNG = np.random.default_rng(12)
TX = np.column_stack([np.ones(80), RNG.normal(size=(80, 3))])
Y = np.where(TX[:, 1] - 0.5 * TX[:, 2] > 0.2, 1, -1)


def test_checkpoint_fit_matches_graded_plain_logistic():
    snapshots = fit_logistic_checkpoints(Y, TX, 0.2, [5, 20])
    expected, _ = implementations.logistic_regression(
        (Y == 1).astype(float), TX, np.zeros(TX.shape[1]), 20, 0.2
    )
    np.testing.assert_allclose(snapshots[20]["w"], expected)


def test_checkpoint_fit_matches_graded_regularized_logistic():
    snapshots = fit_logistic_checkpoints(Y, TX, 0.2, [20], lambda_=1e-3)
    expected, _ = implementations.reg_logistic_regression(
        (Y == 1).astype(float), TX, 1e-3, np.zeros(TX.shape[1]), 20, 0.2
    )
    np.testing.assert_allclose(snapshots[20]["w"], expected)


def test_near_best_prefers_less_work_within_tolerance():
    rows = [
        {"f1_mean": 0.4400, "max_iters": 2000, "gamma": 0.05},
        {"f1_mean": 0.4397, "max_iters": 1000, "gamma": 0.10},
        {"f1_mean": 0.4380, "max_iters": 500, "gamma": 0.12},
    ]
    selected = select_near_best(
        rows, tolerance=0.0005, preference=("max_iters", "gamma")
    )
    assert selected["max_iters"] == 1000


def test_model_config_uses_plain_model_when_lambda_is_zero():
    assert model_config(0.1, 1000, 0)["name"] == "logistic_regression"
    regularized = model_config(0.1, 1000, 1e-4)
    assert regularized["name"] == "reg_logistic_regression"
    assert regularized["lambda_"] == 1e-4


def test_newton_reference_improves_the_regularized_objective():
    y01 = (Y == 1).astype(float)
    initial = regularized_logistic_objective(
        y01, TX, np.zeros(TX.shape[1]), lambda_=1e-4
    )
    result = newton_logistic_reference(Y, TX, lambda_=1e-4)
    assert result["objective"] < initial
    assert result["history"] == sorted(result["history"], reverse=True)
