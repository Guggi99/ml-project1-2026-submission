"""Hyperparameter tuning helpers for logistic models.

The public pipeline deliberately stays small: one config produces one CV result.
This module adds the issue #12 search around that pipeline without changing any
of the six graded functions in :mod:`implementations`.

Two stages are kept separate:

* :func:`checkpoint_cross_validate` reuses a fitted fold while inspecting
  several gradient-descent checkpoints.  It is an exploratory convergence
  diagnostic, not an official logged run.
* :func:`regularization_grid` evaluates each lambda through
  :func:`src.pipeline.develop`, so preprocessing and the F1 threshold are still
  learned inside every training fold.

The selected configuration must finally go through ``run_experiment``.  The
root ``tune.py`` command orchestrates all three steps and creates a submission.
"""

import time

import numpy as np

from implementations import compute_logistic_loss, sigmoid
from src.pipeline import develop, fit_threshold
from src.preprocessing import fit_preprocessor, transform
from src.validation import (
    K_FOLDS,
    SEED,
    accuracy,
    f1_score,
    stratified_k_fold_indices,
)


def regularized_logistic_objective(y01, tx, w, lambda_=0.0):
    """Mean logistic loss plus ``lambda_ * ||w||^2``.

    The penalty convention matches ``implementations.reg_logistic_regression``.
    In particular, its gradient contribution is ``2 * lambda_ * w`` and the
    intercept is regularized too.
    """
    return float(compute_logistic_loss(y01, tx, w) + lambda_ * np.dot(w, w))


def fit_logistic_checkpoints(y, tx, gamma, checkpoints, lambda_=0.0):
    """Fit once and return weights and objectives at requested iterations.

    This uses the exact update of the graded logistic functions.  Keeping it in
    the experiment layer avoids changing their required signatures merely to
    expose intermediate weights.
    """
    tx = np.asarray(tx, dtype=float)
    y01 = (np.asarray(y) == 1).astype(float)
    checkpoints = tuple(sorted(set(int(step) for step in checkpoints)))
    if not checkpoints or checkpoints[0] < 1:
        raise ValueError("checkpoints must contain positive integers")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")

    w = np.zeros(tx.shape[1])
    snapshots = {}
    wanted = set(checkpoints)
    for step in range(1, checkpoints[-1] + 1):
        scores = tx @ w
        gradient = tx.T @ (sigmoid(scores) - y01) / len(y01)
        if lambda_:
            gradient += 2.0 * lambda_ * w
        w -= gamma * gradient
        if not np.all(np.isfinite(w)):
            raise FloatingPointError(
                f"logistic gradient descent diverged at step {step}"
            )
        if step in wanted:
            objective = regularized_logistic_objective(y01, tx, w, lambda_)
            if not np.isfinite(objective):
                raise FloatingPointError(
                    f"logistic objective became non-finite at step {step}"
                )
            snapshots[step] = {"w": w.copy(), "objective": objective}
    return snapshots


def estimate_logistic_step_limit(tx, iterations=30):
    """Estimate the conservative unregularized GD limit ``8 / L``.

    ``L`` is the largest eigenvalue of ``X.T @ X / N``.  Power iteration avoids
    constructing the expensive dense Gram matrix for the wide full dataset.
    The value is diagnostic rather than a proof, so candidates are still
    judged by finite objectives and held-out F1 on every fold.
    """
    tx = np.asarray(tx, dtype=float)
    if tx.ndim != 2 or not len(tx):
        raise ValueError("tx must be a non-empty matrix")
    v = np.ones(tx.shape[1], dtype=float)
    v /= np.linalg.norm(v)
    for _ in range(iterations):
        next_v = tx.T @ (tx @ v) / len(tx)
        norm = np.linalg.norm(next_v)
        if norm == 0:
            return np.inf
        v = next_v / norm
    largest = float(v @ (tx.T @ (tx @ v)) / len(tx))
    return np.inf if largest <= 0 else 8.0 / largest


def checkpoint_cross_validate(
    x,
    y,
    preprocessing,
    gammas,
    checkpoints,
    *,
    threshold="train_f1",
    k=K_FOLDS,
    seed=SEED,
):
    """Measure validation F1 at several GD steps while reusing every fold.

    Preprocessing, weights and the decision threshold see only a fold's
    training rows.  Each gamma is trained once to the largest checkpoint.
    Invalid/diverging gammas are returned separately instead of destroying a
    long search after other candidates have completed.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    gammas = tuple(float(gamma) for gamma in gammas)
    checkpoints = tuple(sorted(set(int(step) for step in checkpoints)))
    if not gammas or any(gamma <= 0 for gamma in gammas):
        raise ValueError("gammas must contain positive values")

    values = {
        (gamma, step): {
            "f1_folds": [],
            "acc_folds": [],
            "train_f1_folds": [],
            "objective_folds": [],
            "predicted_positive_folds": [],
        }
        for gamma in gammas
        for step in checkpoints
    }
    invalid = {}
    step_limits = []
    seconds_by_gamma = {gamma: 0.0 for gamma in gammas}

    for fold, (train_idx, val_idx) in enumerate(
        stratified_k_fold_indices(y, k=k, seed=seed), start=1
    ):
        x_train, y_train = x[train_idx], y[train_idx]
        x_val, y_val = x[val_idx], y[val_idx]
        params = fit_preprocessor(x_train, **preprocessing)
        tx_train = transform(x_train, params)
        tx_val = transform(x_val, params)
        step_limits.append(estimate_logistic_step_limit(tx_train))

        for gamma in gammas:
            if gamma in invalid:
                continue
            start = time.time()
            try:
                snapshots = fit_logistic_checkpoints(
                    y_train, tx_train, gamma, checkpoints
                )
                for step, snapshot in snapshots.items():
                    w = snapshot["w"]
                    train_scores = tx_train @ w
                    cutoff = fit_threshold(threshold, train_scores, y_train)
                    train_pred = np.where(train_scores > cutoff, 1, -1)
                    val_pred = np.where(tx_val @ w > cutoff, 1, -1)
                    item = values[(gamma, step)]
                    item["f1_folds"].append(f1_score(y_val, val_pred))
                    item["acc_folds"].append(accuracy(y_val, val_pred))
                    item["train_f1_folds"].append(f1_score(y_train, train_pred))
                    item["objective_folds"].append(snapshot["objective"])
                    item["predicted_positive_folds"].append(
                        float(np.mean(val_pred == 1))
                    )
            except FloatingPointError as error:
                invalid[gamma] = f"fold {fold}: {error}"
            finally:
                seconds_by_gamma[gamma] += time.time() - start

        del tx_train, tx_val

    candidates = []
    for (gamma, step), item in values.items():
        if gamma in invalid or len(item["f1_folds"]) != k:
            continue
        candidate = {
            "gamma": gamma,
            "max_iters": step,
            "lambda_": 0.0,
            "f1_folds": item["f1_folds"],
            "f1_mean": float(np.mean(item["f1_folds"])),
            "f1_std": float(np.std(item["f1_folds"])),
            "acc_mean": float(np.mean(item["acc_folds"])),
            "train_f1_mean": float(np.mean(item["train_f1_folds"])),
            "objective_mean": float(np.mean(item["objective_folds"])),
            "predicted_positive": float(np.mean(item["predicted_positive_folds"])),
            # One gamma path produces every checkpoint, so this is the time of
            # the complete path rather than a pretend per-checkpoint runtime.
            "path_seconds": round(seconds_by_gamma[gamma], 1),
        }
        candidates.append(candidate)

    candidates.sort(key=lambda row: (row["gamma"], row["max_iters"]))
    return {
        "candidates": candidates,
        "invalid_gammas": {str(key): value for key, value in invalid.items()},
        "estimated_gamma_limit_folds": step_limits,
        "estimated_gamma_limit_min": float(np.min(step_limits)),
        "k": k,
        "seed": seed,
    }


def model_config(gamma, max_iters, lambda_=0.0):
    """Return the pipeline model block for the selected parameters."""
    if lambda_ == 0:
        return {
            "name": "logistic_regression",
            "max_iters": int(max_iters),
            "gamma": float(gamma),
        }
    return {
        "name": "reg_logistic_regression",
        "lambda_": float(lambda_),
        "max_iters": int(max_iters),
        "gamma": float(gamma),
    }


def regularization_grid(
    base_config,
    x,
    y,
    *,
    gamma,
    max_iters,
    lambdas,
    on_result=None,
):
    """Evaluate lambdas through the standard ``develop`` CV runner."""
    rows = []
    for lambda_ in (float(value) for value in lambdas):
        config = {
            **base_config,
            "model": model_config(gamma, max_iters, lambda_),
        }
        result = develop(config, x, y)
        row = {
            "lambda_": lambda_,
            "gamma": float(gamma),
            "max_iters": int(max_iters),
            "f1_folds": result["f1_folds"],
            "f1_mean": result["f1_mean"],
            "f1_std": result["f1_std"],
            "acc_mean": result["acc_mean"],
            "seconds": result["seconds"],
        }
        rows.append(row)
        if on_result is not None:
            on_result(row)
    return rows


def select_near_best(candidates, tolerance=0.0005, preference=()):
    """Choose the cheapest candidate within ``tolerance`` of the best F1.

    This avoids selecting a much longer run for a numerically tiny CV change.
    ``preference`` contains keys whose smaller values are preferred, in order.
    """
    if not candidates:
        raise ValueError("cannot select from an empty candidate list")
    best_f1 = max(row["f1_mean"] for row in candidates)
    eligible = [row for row in candidates if row["f1_mean"] >= best_f1 - tolerance]
    return min(
        eligible,
        key=lambda row: tuple(row[key] for key in preference) + (-row["f1_mean"],),
    )


def newton_logistic_reference(
    y,
    tx,
    *,
    lambda_=1e-6,
    max_iters=20,
    tolerance=1e-6,
):
    """Damped Newton reference for a subsample, not a production model.

    Backtracking requires every accepted step to lower the same regularized
    objective used by gradient descent.  A small numerical damping term keeps
    collinear one-hot matrices solvable without changing the reported
    objective.
    """
    tx = np.asarray(tx, dtype=float)
    y01 = (np.asarray(y) == 1).astype(float)
    w = np.zeros(tx.shape[1])
    objective = regularized_logistic_objective(y01, tx, w, lambda_)
    history = [objective]
    converged = False

    for _ in range(max_iters):
        scores = tx @ w
        probabilities = sigmoid(scores)
        gradient = tx.T @ (probabilities - y01) / len(y01) + 2.0 * lambda_ * w
        curvature = probabilities * (1.0 - probabilities)
        hessian = (tx.T * curvature) @ tx / len(y01)
        hessian += (2.0 * lambda_ + 1e-10) * np.eye(tx.shape[1])
        step = np.linalg.solve(hessian, gradient)

        scale = 1.0
        accepted = False
        while scale >= 2.0**-20:
            proposal = w - scale * step
            proposal_objective = regularized_logistic_objective(
                y01, tx, proposal, lambda_
            )
            if proposal_objective < objective:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break

        w = proposal
        objective = proposal_objective
        history.append(objective)
        if np.max(np.abs(scale * step)) < tolerance:
            converged = True
            break

    return {
        "w": w,
        "objective": objective,
        "iterations": len(history) - 1,
        "converged": converged,
        "history": history,
    }
