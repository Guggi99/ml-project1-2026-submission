"""Tests of the registry-driven options (v2) of src/preprocessing.py.

The registry describes the 321 real columns, so the tests build small
synthetic matrices with that width and set the few columns each test needs.
"""

import numpy as np
import pytest

from src.preprocessing import (
    ENGINEERED_DEFAULT,
    ENGINEERED_GROUPS,
    apply_codebook,
    column_layout,
    fit_preprocessor,
    load_registry,
    transform,
)

REGISTRY = load_registry()
COL = {r["name"]: r["index"] for r in REGISTRY}
ALL_STEPS = dict(
    codes=True,
    flags=True,
    registry_one_hot=True,
    missing_indicators=True,
    select=True,
    nonlinear=True,
    extras=True,
    engineered=ENGINEERED_GROUPS,
    aspirin_interactions=True,
    standardize_all=True,
)


def synthetic(n=80, seed=0):
    """n rows x 321 columns of small coded answers, plus realistic key columns."""
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 4, size=(n, len(REGISTRY))).astype(float)
    x[:, COL["_AGE80"]] = rng.integers(18, 81, n)
    x[:, COL["_BMI5"]] = rng.uniform(16, 45, n)
    x[:, COL["_STATE"]] = rng.choice([1, 6, 20, 53], n)
    x[:, COL["PHYSHLTH"]] = rng.choice([88, 1, 5, 30, 77], n)
    x[:, COL["HAREHAB1"]] = np.nan
    x[:3, COL["HAREHAB1"]] = [1, 2, 7]
    x[:, COL["PA1MIN_"]] = rng.exponential(200, n)
    return x


def test_registry_covers_every_column_once():
    assert [r["index"] for r in REGISTRY] == list(range(321))
    assert {r["decision"] for r in REGISTRY} <= {"keep", "recode", "one-hot", "drop"}


def test_codebook_codes_and_skip_logic():
    x = synthetic()
    x[0, COL["PHYSHLTH"]] = 88  # none -> 0
    x[1, COL["PHYSHLTH"]] = 77  # don't know -> NaN
    x[2, COL["EMPLOY1"]] = 7  # retired: a real answer, untouched
    x[3, COL["CHECKUP1"]] = 8  # never -> 5
    x[4, [COL["BPHIGH4"], COL["BPMEDS"]]] = [3, np.nan]  # no high BP -> no meds
    x[5, [COL["BPHIGH4"], COL["BPMEDS"]]] = [7, np.nan]  # unknown stays unknown

    rules = fit_preprocessor(x, codes=True)["rules"]
    out = apply_codebook(x, rules)

    assert out[0, COL["PHYSHLTH"]] == 0
    assert np.isnan(out[1, COL["PHYSHLTH"]])
    assert out[2, COL["EMPLOY1"]] == 7
    assert out[3, COL["CHECKUP1"]] == 5
    assert out[4, COL["BPMEDS"]] == 2
    assert np.isnan(out[5, COL["BPMEDS"]])
    assert x[1, COL["PHYSHLTH"]] == 77  # the input itself is not modified


def test_harehab_asked_flag_survives_almost_empty_column():
    x = synthetic()
    params = fit_preprocessor(x, codes=True, flags=True)
    tx = transform(x, params)

    # flags come last: HAREHAB1 asked, CVDASPRN asked + yes, RDUCHART asked + yes
    np.testing.assert_array_equal(tx[:5, -5], [1, 1, 1, 0, 0])
    assert (
        COL["HAREHAB1"] not in params["keep_cols"]
    )  # 96 % missing -> v1 rule drops it


def test_all_steps_keep_the_fit_transform_contract():
    x_train = synthetic(seed=1)
    x_test = synthetic(n=7, seed=2)
    x_test[0, COL["_STATE"]] = 72  # a state never seen in training
    original = x_test.copy()

    params = fit_preprocessor(x_train, **ALL_STEPS)
    first = transform(x_test, params)

    assert first.shape[1] == transform(x_train, params).shape[1]
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, transform(x_test, params))
    np.testing.assert_equal(x_test, original)
    np.testing.assert_allclose(first[:, 0], 1.0)


def test_select_keeps_only_registry_columns():
    params = fit_preprocessor(synthetic(), **ALL_STEPS)
    drop = {r["index"] for r in REGISTRY if r["decision"] == "drop"}
    assert not drop.intersection(params["keep_cols"].tolist())
    assert COL["_STATE"] in params["keep_cols"]  # extras=True one-hot encodes it


def test_standardize_all_gives_zero_mean_unit_std_on_training_rows():
    x = synthetic()
    tx = transform(x, fit_preprocessor(x, **ALL_STEPS))
    features = tx[:, 1:]
    varying = features.std(axis=0) > 0
    np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(features[:, varying].std(axis=0), 1.0, atol=1e-9)


def test_registry_steps_need_codes():
    with pytest.raises(ValueError):
        fit_preprocessor(synthetic(), select=True)


def test_mixed_units_become_one_unit():
    x = synthetic(n=6)
    x[:, COL["WEIGHT2"]] = [180, 9080, 9999, 7777, 120, np.nan]
    x[:, COL["HEIGHT3"]] = [510, 9165, 9999, 600, 411, np.nan]
    x[:, COL["ALCDAY5"]] = [107, 204, 888, 777, 101, 230]
    x[:, COL["FRUIT1"]] = [102, 203, 301, 555, 999, 300]
    x[:, COL["EXERHMM1"]] = [130, 45, 200, 777, 800, np.nan]
    x[:, COL["FLSHTMY2"]] = [102014, 12015, 772014, 777777, 999999, np.nan]
    out = apply_codebook(x, fit_preprocessor(x, codes=True)["rules"])

    def col(name):
        return out[:, COL[name]]

    np.testing.assert_allclose(col("WEIGHT2")[:2], [180, 80 * 2.2046])
    assert np.isnan(col("WEIGHT2")[2:4]).all() and col("WEIGHT2")[4] == 120
    np.testing.assert_allclose(col("HEIGHT3")[[0, 1, 3, 4]], [70, 165 / 2.54, 72, 59])
    np.testing.assert_allclose(col("ALCDAY5")[[0, 1, 2, 4, 5]], [30, 4, 0, 30 / 7, 30])
    assert np.isnan(col("ALCDAY5")[3])
    np.testing.assert_allclose(
        col("FRUIT1")[[0, 1, 2, 3, 5]], [2, 3 / 7, 1 / 30, 0, 0.5 / 30]
    )
    np.testing.assert_allclose(col("EXERHMM1")[[0, 1, 2, 4]], [90, 45, 120, 480])
    np.testing.assert_allclose(col("FLSHTMY2")[:3], [2014.75, 2015.0, 2014.5])
    assert np.isnan(col("FLSHTMY2")[3:]).all()


def test_column_layout_names_every_output_column():
    x = synthetic()
    for options in ({}, dict(codes=True, flags=True), ALL_STEPS):
        params = fit_preprocessor(x, **options)
        layout = column_layout(params)
        assert len(layout) == transform(x, params).shape[1]
        assert layout[0] == ("intercept", -1, "intercept")


def test_engineered_columns_follow_their_formulas():
    x = synthetic(n=40)
    # row 0: a 70-year-old man with high blood pressure and diabetes, nothing else
    for name, value in [("SEX", 1), ("_AGE80", 70), ("BPHIGH4", 1), ("DIABETE3", 1)]:
        x[0, COL[name]] = value
    for name in ("TOLDHI2", "CVDSTRK3", "CHCKIDNY", "CHCCOPD1", "HAVARTH3", "ADDEPEV2"):
        x[0, COL[name]] = 2  # no
    x[0, COL["_SMOKER3"]] = 4  # never smoked
    x[0, COL["_TOTINDA"]] = 1  # active
    x[0, COL["_BMI5"]] = 25.0
    x[1, COL["SEX"]] = 2  # a woman
    groups = ("sex_age", "age_factors", "counts", "factor_pairs")
    params = fit_preprocessor(x, codes=True, engineered=groups)
    tx = transform(x, params)
    column = {name: k for k, (_, _, name) in enumerate(column_layout(params))}

    age = (70 - 50) / 10
    assert tx[0, column["male x age"]] == age
    assert tx[0, column["male x age^2"]] == age**2
    assert tx[0, column["age x high_bp"]] == age
    assert tx[0, column["high_bp x diabetes"]] == 1
    assert tx[0, column["# cardio"]] == 2 and tx[0, column["# cardio^2"]] == 4
    assert tx[1, column["male x age"]] == 0


def test_ordinal_levels_are_learned_on_training_rows():
    x_train = synthetic(seed=3)
    params = fit_preprocessor(x_train, codes=True, engineered=("ordinal_one_hot",))
    x_test = synthetic(n=5, seed=4)
    x_test[0, COL["GENHLTH"]] = 5  # a level absent from training (values 1..3)
    tx = transform(x_test, params)
    names = [name for _, _, name in column_layout(params)]
    assert "GENHLTH = 5" not in names
    genhlth = [k for k, name in enumerate(names) if name.startswith("GENHLTH = ")]
    np.testing.assert_array_equal(tx[0, genhlth], 0)  # unseen level: all zeros


def test_engineered_needs_codes_and_known_groups():
    with pytest.raises(ValueError):
        fit_preprocessor(synthetic(), engineered=True)
    with pytest.raises(ValueError):
        fit_preprocessor(synthetic(), codes=True, engineered=("not_a_group",))
    params = fit_preprocessor(
        synthetic(), codes=True, engineered=ENGINEERED_GROUPS[::-1]
    )
    assert params["engineered"] == ENGINEERED_GROUPS  # order fixed, not as written
    params = fit_preprocessor(synthetic(), codes=True, engineered=True)
    assert params["engineered"] == ENGINEERED_DEFAULT  # the six recommended groups
    assert not {"ordinal_one_hot", "sex_one_hot"} & set(ENGINEERED_DEFAULT)


def test_aspirin_interactions_follow_their_formulas():
    x = synthetic(n=40)
    # row 0: a 70-year-old man in the aspirin module, high BP, general health 4
    for name, value in [
        ("CVDASPRN", 1),
        ("SEX", 1),
        ("_AGE80", 70),
        ("BPHIGH4", 1),
        ("GENHLTH", 4),
        ("TOLDHI2", 2),
        ("DIABETE3", 3),
        ("BPMEDS", 1),
    ]:
        x[0, COL[name]] = value
    x[1, COL["CVDASPRN"]] = np.nan  # not in the module
    params = fit_preprocessor(x, codes=True, aspirin_interactions=True)
    tx = transform(x, params)
    layout = column_layout(params)
    column = {name: k for k, (_, _, name) in enumerate(layout)}
    assert sum(kind == "aspirin" for kind, _, _ in layout) == 7

    assert tx[0, column["aspirin answered x age"]] == (70 - 50) / 10
    assert tx[0, column["aspirin answered x male"]] == 1
    assert tx[0, column["aspirin answered x genhlth"]] == 4 - 3
    assert tx[0, column["aspirin answered x high_bp"]] == 1
    assert tx[0, column["aspirin answered x bp_meds"]] == 1
    assert tx[0, column["aspirin answered x high_chol"]] == 0
    aspirin = [k for k, (kind, _, _) in enumerate(layout) if kind == "aspirin"]
    np.testing.assert_array_equal(tx[1, aspirin], 0)  # outside the module: all 0


def test_aspirin_interactions_need_codes():
    with pytest.raises(ValueError):
        fit_preprocessor(synthetic(), aspirin_interactions=True)
