# ---------------------------------------------------------------------------
# VALIDATION, estimating how well a model predicts labels for unseen rows.
#
# This module provides a common evaluation procedure for comparing models:
# 1. Split the labelled data into k folds, keeping approximately the same
#    proportion of positive and negative examples in each fold (stratification).
# 2. Train on k - 1 folds and predict the labels of the remaining fold.
# 3. Measure those predictions using F1 and accuracy.
# 4. Repeat until every fold has been used for validation, then report the
#    average scores and how much they vary between folds.
#
# F1 measures how well the model detects the positive class (label 1).
# Accuracy is the fraction of all labels predicted correctly.
# The defaults are 5 folds and seed 42; the seed makes the split reproducible.
# For fair comparisons, use the same data, labels, fold count and seed.
# Preprocessing must be learned from each fold's training rows only, so that
# information from the validation rows does not leak into model training.
# ---------------------------------------------------------------------------

"""Shared evaluation utilities for model comparison.

Results recorded in ``results/log.csv`` should be produced by
``cross_validate`` with the default seed and fold count. This gives every
experiment the same evaluation protocol. Scores are directly comparable only
when they also use the same dataset, row selection and target labels.

Owner: toolbox + evaluation role.
"""

import numpy as np

SEED = 42
K_FOLDS = 5

POSITIVE = 1


# ---------------------------------------------------------------------------
# STRATIFIED K-FOLD INDICES, splitting rows while preserving class proportions.
#
# y contains one class label per row. k is the number of validation folds
# (at least 2), and seed controls the reproducible random shuffle.
# "Stratified" means distributing each class as evenly as possible across
# the folds, so their class proportions stay close to those of the full data.
# This matters when positive cases are rare: an ordinary random split could
# put too many of them in one fold and too few in another.
#
# For example, with 100 rows (20 positive, 80 negative) and k = 5:
# each validation fold has 4 positive and 16 negative rows;
# its training set contains the other 16 positive and 64 negative rows.
#
# The function builds these splits in two stages:
# 1. For each class, collect its row indices, shuffle them, and divide them
#    into k chunks. Add one chunk to each validation fold. When the class
#    count is not divisible by k, the first folds receive one extra index.
# 2. Combine the class chunks for each fold and sort the validation indices.
#    All remaining row indices form that fold's training set.
#
# The result is a list of k pairs: (train_idx, val_idx).
# These are row positions, not the feature values themselves; use them as
# x[train_idx], y[train_idx], x[val_idx] and y[val_idx] to select the data.
# Within each pair, training and validation rows do not overlap. Across all
# pairs, every row is used once for validation and k - 1 times for training.
# The same y, k and seed produce the same splits.
# If a class has fewer than k rows, some validation folds will lack that class.
# ---------------------------------------------------------------------------
def stratified_k_fold_indices(y, k=K_FOLDS, seed=SEED):
    """Return ``k`` reproducible ``(train_idx, val_idx)`` pairs.

    Each class is shuffled separately and divided as evenly as possible among
    the validation folds. Therefore, each fold approximately preserves the
    class ratio of ``y``; validation counts for a given class differ by at
    most one. This matters because MICHD cases are a small minority.

    Every row appears in exactly one validation fold and in the training part
    of every other fold.
    """
    y = np.asarray(y)
    if k < 2:
        raise ValueError("k must be at least 2")

    rng = np.random.default_rng(seed)
    n = y.shape[0]

    # Collect each fold's validation indices, one class at a time.
    val_per_fold = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        # array_split copes with a class count that is not a multiple of k:
        # the first folds take the extra element.
        for fold, chunk in enumerate(np.array_split(idx, k)):
            val_per_fold[fold].append(chunk)

    folds = []
    for chunks in val_per_fold:
        val_idx = np.sort(np.concatenate(chunks))
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        folds.append((np.flatnonzero(mask), val_idx))
    return folds


# ---------------------------------------------------------------------------
# F1 SCORE, measure of predictive performance.
# https://en.wikipedia.org/wiki/F-score
#
# TP = true positives:  predicted positive and actually positive
# FP = false positives: predicted positive but actually negative
# FN = false negatives: predicted negative but actually positive
#
# Precision = TP / (TP + FP)
# Recall    = TP / (TP + FN)
#
# F1 is the harmonic mean of precision and recall:
#
#              2 * Precision * Recall
# F1 = -----------------------------------------
#                 Precision + Recall
#
# Substituting the precision and recall formulas gives:
#
#                 2 * TP
# F1 = --------------------------------
#              2 * TP + FP + FN
#
# F1 ranges from 0 to 1. A high score requires both good precision and good
# recall. True negatives are not included because F1 focuses on detecting the
# positive class, which is particularly useful for an imbalanced dataset.
# ---------------------------------------------------------------------------
def f1_score(y_true, y_pred):
    """F1 score for the positive class; labels in {-1, 1}.

    Returns 0.0 when the score is undefined (nothing predicted positive and
    nothing actually positive), so a degenerate model scores 0 rather than
    raising in the middle of a sweep.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = np.sum((y_pred == POSITIVE) & (y_true == POSITIVE))
    fp = np.sum((y_pred == POSITIVE) & (y_true != POSITIVE))
    fn = np.sum((y_pred != POSITIVE) & (y_true == POSITIVE))

    denominator = 2 * tp + fp + fn
    if denominator == 0:
        return 0.0
    return float(2 * tp / denominator)


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# CROSS-VALIDATION, evaluating a training procedure across several data splits.
#
# x contains the raw feature rows; y contains their true labels (-1 or 1).
# A fold is one portion of the data reserved for validation in a given round.
# With k = 5, each round trains on roughly 80% of the rows and validates on
# the remaining 20%. Each row is used for validation exactly once.
#
# For each stratified split:
# 1. fit_fn(x_train, y_train) learns preprocessing and trains a fresh model
#    using only that round's training rows, then returns the fitted model.
# 2. predict_fn(model, x_val) applies the learned preprocessing and model to
#    the validation rows, returning predicted labels without retraining.
# 3. Compare these predictions with y_val and record F1 and accuracy.
#
# These functions are supplied by the caller, so the same evaluation code
# can be used for different models. They must respect the training/validation
# separation: cross_validate passes separate rows but cannot prevent a caller
# from using information learned outside the fold.
#
# For either metric, the reported average is:
#
#                  score_1 + score_2 + ... + score_k
# Mean score = ---------------------------------------
#                                k
#
# The standard deviation describes how much scores vary between folds;
# a smaller value means more consistent scores across these splits.
# The returned dictionary contains the means, standard deviations, individual
# fold scores, k and seed. It does not return a final model trained on all rows.
# ---------------------------------------------------------------------------
def cross_validate(x, y, fit_fn, predict_fn, k=K_FOLDS, seed=SEED):
    """Run k-fold CV; preprocessing is fitted inside each fold.

    ``fit_fn`` gets the raw training rows of the fold, so whatever it learns —
    medians, scales, category encodings, weights — never sees the validation
    rows. That is the point of this function: no preprocessing is fitted
    outside it.

    Args:
        fit_fn: ``fit_fn(x_train, y_train) -> model``
        predict_fn: ``predict_fn(model, x_val) -> labels in {-1, 1}``

    Returns:
        dict with mean and std of F1 and accuracy over the folds, plus the
        per-fold values under ``f1_folds`` / ``acc_folds``, so a single odd
        fold can be spotted instead of being hidden by the mean.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    f1_folds = []
    acc_folds = []
    for train_idx, val_idx in stratified_k_fold_indices(y, k=k, seed=seed):
        model = fit_fn(x[train_idx], y[train_idx])
        y_hat = predict_fn(model, x[val_idx])
        f1_folds.append(f1_score(y[val_idx], y_hat))
        acc_folds.append(accuracy(y[val_idx], y_hat))

    return {
        "f1_mean": float(np.mean(f1_folds)),
        "f1_std": float(np.std(f1_folds)),
        "acc_mean": float(np.mean(acc_folds)),
        "acc_std": float(np.std(acc_folds)),
        "f1_folds": f1_folds,
        "acc_folds": acc_folds,
        "k": k,
        "seed": seed,
    }
