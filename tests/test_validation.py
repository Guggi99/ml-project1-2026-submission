"""Checks on the CV harness — the one thing every reported score depends on."""

import numpy as np
import pytest

from src.validation import (
    K_FOLDS,
    SEED,
    accuracy,
    cross_validate,
    f1_score,
    stratified_k_fold_indices,
)

# 30 rows, 20% positive — the imbalance is roughly what the real labels show.
Y = np.array([1] * 6 + [-1] * 24)
X = np.arange(60).reshape(30, 2)


def test_folds_partition_the_data():
    folds = stratified_k_fold_indices(Y)
    val_all = np.concatenate([val for _, val in folds])

    assert len(folds) == K_FOLDS
    np.testing.assert_array_equal(np.sort(val_all), np.arange(len(Y)))


def test_train_and_val_are_disjoint_and_complete():
    for train, val in stratified_k_fold_indices(Y):
        assert set(train).isdisjoint(val)
        assert len(train) + len(val) == len(Y)


def test_class_ratio_is_kept_in_every_fold():
    for _, val in stratified_k_fold_indices(Y):
        # 6 positives over 5 folds: every fold gets 1 or 2.
        assert np.sum(Y[val] == 1) in (1, 2)


def test_same_seed_gives_same_split():
    a = stratified_k_fold_indices(Y, seed=SEED)
    b = stratified_k_fold_indices(Y, seed=SEED)
    for (tr_a, va_a), (tr_b, va_b) in zip(a, b):
        np.testing.assert_array_equal(tr_a, tr_b)
        np.testing.assert_array_equal(va_a, va_b)


def test_different_seed_gives_different_split():
    a = stratified_k_fold_indices(Y, seed=0)
    b = stratified_k_fold_indices(Y, seed=1)
    assert any(not np.array_equal(x[1], z[1]) for x, z in zip(a, b))


def test_k_below_two_is_rejected():
    with pytest.raises(ValueError):
        stratified_k_fold_indices(Y, k=1)


def test_f1_and_accuracy_on_a_hand_computed_case():
    y_true = np.array([1, 1, 1, -1, -1])
    y_pred = np.array([1, 1, -1, 1, -1])
    # tp=2, fp=1, fn=1 -> F1 = 2*2 / (2*2 + 1 + 1) = 0.666...
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(0.6)


def test_f1_is_zero_when_nothing_is_predicted_positive():
    y_true = np.array([1, -1, -1])
    assert f1_score(y_true, np.array([-1, -1, -1])) == 0.0


def test_f1_is_zero_rather_than_nan_when_undefined():
    y_true = np.array([-1, -1, -1])
    assert f1_score(y_true, np.array([-1, -1, -1])) == 0.0


def test_perfect_prediction_scores_one():
    y_true = np.array([1, -1, 1, -1])
    assert f1_score(y_true, y_true) == 1.0
    assert accuracy(y_true, y_true) == 1.0


def test_cross_validate_sees_only_training_rows_when_fitting():
    """fit_fn must never be handed a validation row — that is the leak we care about."""
    seen = []

    def fit_fn(x_train, y_train):
        seen.append(set(map(tuple, x_train)))
        return None

    def predict_fn(_model, x_val):
        return np.full(len(x_val), -1)

    folds = stratified_k_fold_indices(Y)
    cross_validate(X, Y, fit_fn, predict_fn)

    for (_, val_idx), train_rows in zip(folds, seen):
        val_rows = set(map(tuple, X[val_idx]))
        assert train_rows.isdisjoint(val_rows)


def test_cross_validate_reports_a_perfect_oracle():
    """An oracle that returns the true labels must score 1.0 on every fold."""
    lookup = {tuple(row): label for row, label in zip(X, Y)}

    result = cross_validate(
        X,
        Y,
        fit_fn=lambda x_train, y_train: None,
        predict_fn=lambda _m, x_val: np.array([lookup[tuple(r)] for r in x_val]),
    )

    assert result["f1_mean"] == pytest.approx(1.0)
    assert result["acc_mean"] == pytest.approx(1.0)
    assert result["f1_std"] == pytest.approx(0.0)
    assert len(result["f1_folds"]) == K_FOLDS
    assert result["k"] == K_FOLDS and result["seed"] == SEED


def test_cross_validate_reports_the_all_negative_baseline():
    """Predicting 'healthy' for everyone: high accuracy, zero F1 — the trap to beat."""
    result = cross_validate(
        X,
        Y,
        fit_fn=lambda x_train, y_train: None,
        predict_fn=lambda _m, x_val: np.full(len(x_val), -1),
    )

    assert result["f1_mean"] == 0.0
    # 6 positives over 5 folds do not divide evenly, so the fold means differ
    # slightly from the overall 0.8 negative rate.
    assert result["acc_mean"] == pytest.approx(0.8, abs=0.01)
