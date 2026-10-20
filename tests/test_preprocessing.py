import numpy as np

from src.preprocessing import fit_preprocessor, transform


def test_fit_and_transform_drop_impute_standardise_without_mutating_input():
    x = np.array(
        [
            [1.0, 5.0, np.nan],
            [2.0, 5.0, 10.0],
            [3.0, 5.0, 12.0],
            [4.0, 5.0, 14.0],
        ]
    )
    original = x.copy()

    params = fit_preprocessor(x)
    tx = transform(x, params)

    np.testing.assert_array_equal(params["keep_cols"], np.array([0, 2]))
    assert tx.shape == (4, 3)  # intercept plus two retained columns
    np.testing.assert_allclose(tx[:, 0], 1.0)
    assert np.all(np.isfinite(tx))
    np.testing.assert_allclose(tx[:, 1:].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(tx[:, 1:].std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_equal(x, original)


def test_sentinel_decisions_are_learned_on_train_and_reused_on_test():
    coded = np.tile([1.0, 2.0], 50)
    coded[-2:] = [7.0, 9.0]
    genuine_count = np.tile(np.arange(8, dtype=float), 13)[:100]
    x_train = np.column_stack((coded, genuine_count))

    params = fit_preprocessor(x_train)

    assert params["sentinels"] == {0: (7, 9)}
    np.testing.assert_array_equal(params["keep_cols"], np.array([0, 1]))

    x_test = np.array([[7.0, 7.0], [9.0, 3.0], [1.0, 0.0]])
    tx = transform(x_test, params)

    # Sentinel values in column 0 use the training median. The genuine 7 in
    # column 1 remains an ordinary value.
    expected_imputed = (params["median"][0] - params["mean"][0]) / params["std"][0]
    np.testing.assert_allclose(tx[:2, 1], expected_imputed)
    expected_seven = (7.0 - params["mean"][1]) / params["std"][1]
    np.testing.assert_allclose(tx[0, 2], expected_seven)


def test_transform_uses_training_statistics_for_new_missing_values():
    x_train = np.array([[1.0], [2.0], [100.0]])
    params = fit_preprocessor(x_train)

    tx = transform(np.array([[np.nan], [1000.0]]), params)

    expected = (params["median"][0] - params["mean"][0]) / params["std"][0]
    assert tx[0, 1] == expected


def _mixed_training_matrix():
    """Column 0 continuous, column 1 coded {1, 2, 3} with one NaN."""
    continuous = np.linspace(0.0, 30.0, 12) ** 1.5
    coded = np.tile([1.0, 2.0, 3.0], 4)
    coded[5] = np.nan
    return np.column_stack((continuous, coded))


def test_default_options_match_the_plain_pipeline():
    x = _mixed_training_matrix()

    params = fit_preprocessor(x)
    tx = transform(x, params)

    imputed = np.where(np.isnan(x), params["median"], x)
    expected = np.c_[np.ones(len(x)), (imputed - params["mean"]) / params["std"]]
    np.testing.assert_allclose(tx, expected)


def test_transform_accepts_params_without_the_new_keys():
    x = _mixed_training_matrix()
    params = fit_preprocessor(x)
    old_params = {
        key: params[key] for key in ("keep_cols", "sentinels", "median", "mean", "std")
    }

    np.testing.assert_allclose(transform(x, old_params), transform(x, params))


def test_one_hot_encodes_coded_columns_with_a_missing_level():
    x = _mixed_training_matrix()

    params = fit_preprocessor(x, one_hot=True)
    tx = transform(x, params)

    np.testing.assert_array_equal(params["numeric_pos"], [0])
    np.testing.assert_array_equal(params["one_hot_pos"], [1])
    np.testing.assert_array_equal(params["levels"][0], [1.0, 2.0, 3.0])
    assert params["missing_level"] == [True]

    # intercept, one numeric column, indicators for 1, 2, 3 and "missing"
    assert tx.shape == (12, 6)
    np.testing.assert_array_equal(tx[5, 2:], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_array_equal(tx[:, 2:].sum(axis=1), 1.0)


def test_one_hot_gives_zeros_for_a_level_unseen_in_training():
    params = fit_preprocessor(_mixed_training_matrix(), one_hot=True)

    tx = transform(np.array([[10.0, 4.0], [10.0, 2.0]]), params)

    np.testing.assert_array_equal(tx[0, 2:], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(tx[1, 2:], [0.0, 1.0, 0.0, 0.0])


def test_polynomial_terms_are_standardised_powers_of_numeric_columns():
    x = _mixed_training_matrix()

    params = fit_preprocessor(x, one_hot=True, poly_degree=3)
    tx = transform(x, params)

    # intercept, numeric, its square and cube, then 4 indicators
    assert tx.shape == (12, 8)
    z = tx[:, 1]
    for column, power in ((2, 2), (3, 3)):
        expected = (z**power - (z**power).mean()) / (z**power).std()
        np.testing.assert_allclose(tx[:, column], expected, atol=1e-12)


def test_new_options_keep_transform_free_of_statistics_and_side_effects():
    x_train = _mixed_training_matrix()
    params = fit_preprocessor(x_train, one_hot=True, poly_degree=2)

    x_test = np.array([[np.nan, np.nan], [1e3, 1.0]])
    original = x_test.copy()
    first = transform(x_test, params)

    assert first.shape[1] == transform(x_train, params).shape[1]
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, transform(x_test, params))
    np.testing.assert_equal(x_test, original)


def test_missing_indicators_mark_training_learned_missing_columns():
    x_train = np.array(
        [
            [1.0, np.nan, 10.0],
            [2.0, 1.0, 20.0],
            [3.0, 2.0, np.nan],
            [4.0, 1.0, 40.0],
        ]
    )

    params = fit_preprocessor(x_train, missing_indicators=True)
    tx = transform(x_train, params)

    # Positions 1 and 2 within keep_cols contained NaN in the training rows.
    np.testing.assert_array_equal(params["missing_indicator_pos"], [1, 2])

    # Intercept + three standardised features + two binary missing flags.
    assert tx.shape == (4, 6)
    np.testing.assert_array_equal(
        tx[:, -2:],
        [
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
    )


def test_missing_indicators_do_not_duplicate_one_hot_missing_level():
    x = _mixed_training_matrix()

    params = fit_preprocessor(
        x,
        one_hot=True,
        missing_indicators=True,
    )

    # The only missing column is one-hot encoded and already has a dedicated
    # missing category, so no second flag is appended.
    np.testing.assert_array_equal(params["missing_indicator_pos"], [])
    assert transform(x, params).shape == (12, 6)
